"""
Stage 3: Pulsatile forward PINN for incompressible Navier-Stokes flow.

The model solves the transient forward problem with a known pulsatile inlet
pressure:

    p_in(t) = A0 + A1 sin(2*pi*t/T)

The implementation uses:
- non-dimensional formulation,
- Xavier initialization,
- hard no-slip wall boundary conditions,
- hard inlet/outlet pressure ansatz,
- random uniform collocation points,
- time-window curriculum training,
- Adam with MultiStepLR,
- L-BFGS refinement.

COMSOL data are used for evaluation and plotting. They are not used as
full-field supervised training data.

Paths are resolved automatically relative to the repository root.
"""

import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


# =========================
# Configuration
# =========================

# Run mode
TRAIN_MODE = True
SAVE_CHECKPOINT = True
SAVE_PLOT_DATA = False

# Plot options
CREATE_PLOTS = True
SHOW_PLOTS = False
OVERWRITE_EXISTING_OUTPUTS = False

# Reproducibility
RANDOM_SEED = False
FIXED_SEED = 189869491

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]

COMSOL_DATA_PATH = REPO_ROOT / "data" / "comsol" / "stage3_stage4" / "NS_xy_pulsatile.txt"
CHECKPOINT_PATH = REPO_ROOT / "models" / "stage3_forward_model.pt"
PLOT_DATA_PATH = REPO_ROOT / "stage3_pulsatile_forward" / "stage3_plot_data.npz"
FIGURES_DIR = REPO_ROOT / "stage3_pulsatile_forward" / "figures"
RUN_NAME = "stage3_forward"
RUN_FIGURES_DIR = FIGURES_DIR

# Physical parameters
MU = 1.0
RHO = 1.0

P_IN_MEAN = 10.0
P_IN_AMPLITUDE = 5.0
P_OUT_DIM = 0.0
P_PERIOD = 0.5

T_INITIAL = 0.0
T_FINAL = 3.0
DT_EXPORT = 0.05

LENGTH = 12.0
HEIGHT = 4.0

# Network architecture
INPUT_DIM = 3
OUTPUT_DIM = 3
HIDDEN_LAYERS = 5
NEURONS = 64

# Training points
N_DOMAIN = 10000
N_INLET = 200
N_OUTLET = 200
N_INITIAL = 200

# Boundary-check points
N_WALL_UPPER = 500
N_WALL_LOWER = 500

# Stage 3 training strategy
TIME_WINDOWS_PHYS = [1.0, 2.0, 3.0]
EPOCHS_PER_WINDOW = [3000, 3000, 4000]
RESET_LR_EACH_WINDOW = True

if len(TIME_WINDOWS_PHYS) != len(EPOCHS_PER_WINDOW):
    raise ValueError(
        "TIME_WINDOWS_PHYS and EPOCHS_PER_WINDOW must have the same length."
    )

if any(np.diff(TIME_WINDOWS_PHYS) <= 0):
    raise ValueError("TIME_WINDOWS_PHYS must be strictly increasing.")

training_schedule = list(zip(TIME_WINDOWS_PHYS, EPOCHS_PER_WINDOW))
configured_total_adam_epochs = sum(stage_epochs for _, stage_epochs in training_schedule)

# Optimizer settings
LR_ADAM = 1e-3
LR_LBFGS = 1e-1
MAX_LBFGS_ITERATIONS = 1000

# =========================
# Reproducibility / Seed
# =========================

if RANDOM_SEED:
    seed = random.randint(0, 2**32 - 1)
else:
    seed = FIXED_SEED

print("=" * 60)
print(f"Seed used for this run: {seed}")
print("=" * 60)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def sync_time():
    """
    Accurate timing, especially when running on GPU.
    CUDA operations are asynchronous, so synchronize before reading time.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

# =========================
# Physical parameters
# =========================

mu = MU
rho = RHO

A0 = P_IN_MEAN
A1 = P_IN_AMPLITUDE
T = P_PERIOD

p_in_dim = P_IN_MEAN
p_out_dim = P_OUT_DIM
deltaP = p_in_dim - p_out_dim

t_initial = T_INITIAL
t_final = T_FINAL
cycles = (t_final - t_initial) / T

L = LENGTH
H = HEIGHT

# Characteristic values for non-dimensionalization
x_ref = L
y_ref = H
t_ref = T
u_ref = deltaP * H**2 / (8 * mu * L)
p_ref = rho * u_ref**2

alpha = y_ref / x_ref
Re = rho * u_ref * y_ref / mu
St = y_ref / (u_ref * t_ref)

print("Problem is solved in non-dimensional form:")
print(f"X_ref = {x_ref:.4e} m")
print(f"Y_ref = {y_ref:.4e} m")
print(f"t_ref = {t_ref:.4e} s")
print(f"P_ref = {p_ref:.4e} Pa")
print(f"U_ref = {u_ref:.4e} m/s")
print(f"Aspect ratio Y_ref/X_ref: alpha = {alpha:.4e}")
print(f"Reynolds Number = {Re:.4f}")
print(f"Strouhal Number = {St:.4f}")

# =========================
# Model: (y*, x*, t*) -> (u_y*, u_x*, p*)
# =========================

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation = nn.Tanh()

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])

            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

            self.layers.append(layer)

    def forward(self, y, x, t):
        inp = torch.cat([y, x, t], dim=1)

        out = inp

        for i in range(len(self.layers) - 1):
            out = self.activation(self.layers[i](out))

        raw = self.layers[-1](out)

        # Hard no-slip wall boundary conditions:
        # u_y = u_x = 0 at y*=0 and y*=1
        phi = y * (1.0 - y)

        u_y = raw[:, 0:1] * phi
        u_x = raw[:, 1:2] * phi

        # Hard pressure boundary conditions:
        # p(x=0,t) = p_in(t), p(x=L,t) = p_out

        # x_bar maps the current nondimensional x-domain [0, L/x_ref] to [0, 1].
        x_bar = x / (L / x_ref)

        t_dim_local = t * t_ref

        p_in_t = (
            A0 / p_ref
            + (A1 / p_ref) * torch.sin(2.0 * np.pi * t_dim_local / T)
        )

        p_out_star = torch.full_like(p_in_t, p_out_dim / p_ref)

        p = (
            (1.0 - x_bar) * p_in_t
            + x_bar * p_out_star
            + x_bar * (1.0 - x_bar) * raw[:, 2:3]
        )

        return torch.cat([u_y, u_x, p], dim=1)


# =========================
# Network
# =========================

layers = [INPUT_DIM] + [NEURONS] * HIDDEN_LAYERS + [OUTPUT_DIM]
print("Network Architecture:", layers)

model = PINN(layers).to(device)
print("Model on device:", next(model.parameters()).device)


# =========================
# Adam Optimizer
# =========================

optimizer_adam = torch.optim.Adam(model.parameters(), lr=LR_ADAM)

print(f"Adam optimizer with initial learning rate: {LR_ADAM:.1e}")

def make_adam_scheduler(optimizer, stage_epochs):
    milestones = [
        int(0.40 * stage_epochs),
        int(0.70 * stage_epochs),
        int(0.90 * stage_epochs),
    ]

    milestones = sorted(list(set([m for m in milestones if m > 0])))

    print(f"Adam learning rate scheduler: MultiStepLR, milestones={milestones}")

    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.5
    )


# =========================
# L-BFGS optimizer
# =========================

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=LR_LBFGS,
    max_iter=MAX_LBFGS_ITERATIONS,
    line_search_fn="strong_wolfe"
)

def closure():
    global lbfgs_iter

    optimizer_lbfgs.zero_grad()
    loss, _, _, _ = loss_function()
    loss.backward()

    loss_history_lbfgs.append(loss.item())
    lbfgs_iter_history.append(lbfgs_iter)

    lbfgs_iter += 1

    return loss

# =========================
# Load COMSOL reference data for evaluation (not used in training)
# =========================

# Wide format: columns = [x, y, (u_y, u_x, p)@t0, (u_y, u_x, p)@t1, ...]

if not COMSOL_DATA_PATH.exists():
    raise FileNotFoundError(
        f"COMSOL file not found:\n{COMSOL_DATA_PATH}"
    )

data = np.loadtxt(COMSOL_DATA_PATH, comments="%")
print(f"Loaded COMSOL data from: {COMSOL_DATA_PATH}")

n_nodes = data.shape[0]
n_cols = data.shape[1]

if (n_cols - 2) % 3 != 0:
    raise ValueError("Unexpected column count: expected 2 + 3*n_times.")

n_times = (n_cols - 2) // 3

# COMSOL export: t = 0, 0.05, ..., 3.0 s
t_values = DT_EXPORT * np.arange(n_times)

print(
    f"Nodes: {n_nodes}, time snapshots: {n_times}, "
    f"t in [{t_values[0]:.3f}, {t_values[-1]:.3f}] s"
)

# Spatial coordinates
x_dim = data[:, 0:1]
y_dim = data[:, 1:2]

# Field block -> (n_nodes, n_times, 3)
# field 0 = u_y, field 1 = u_x, field 2 = p
fields = data[:, 2:].reshape(n_nodes, n_times, 3)

u_y_wide = fields[:, :, 0]
u_x_wide = fields[:, :, 1]
p_wide = fields[:, :, 2]

# Long format: one row per (node, time)
x_long = np.repeat(x_dim, n_times, axis=1)
y_long = np.repeat(y_dim, n_times, axis=1)
t_long = np.tile(t_values, (n_nodes, 1))

x_dim_long = x_long.reshape(-1, 1)
y_dim_long = y_long.reshape(-1, 1)
t_dim_long = t_long.reshape(-1, 1)

u_y_dim_long = u_y_wide.reshape(-1, 1)
u_x_dim_long = u_x_wide.reshape(-1, 1)
p_dim_long = p_wide.reshape(-1, 1)

print(
    f"Long-format dataset: {x_dim_long.shape[0]} rows "
    f"(= {n_nodes} nodes x {n_times} times)"
)

# Non-dimensional inputs; dimensional fields for validation
x_data = torch.tensor(x_dim_long / x_ref, dtype=torch.float32, device=device)
y_data = torch.tensor(y_dim_long / y_ref, dtype=torch.float32, device=device)
t_data = torch.tensor(t_dim_long / t_ref, dtype=torch.float32, device=device)

u_y_data = torch.tensor(u_y_dim_long, dtype=torch.float32, device=device)
u_x_data = torch.tensor(u_x_dim_long, dtype=torch.float32, device=device)
p_data = torch.tensor(p_dim_long, dtype=torch.float32, device=device)

# =========================
# Training point builders
# =========================

def build_training_window(t_window_max, print_summary=True):
    """
    Rebuilds all time-dependent training tensors for the current time window.
    The model is not reinitialized.
    """

    global y_d, x_d, t_d
    global y_wall_upper, x_wall_upper, t_wall_upper
    global y_wall_lower, x_wall_lower, t_wall_lower
    global y_in, x_in, t_in
    global y_out, x_out, t_out

    t0_star = t_initial / t_ref
    t1_star = t_window_max / t_ref

    # -------------------------
    # Collocation points
    # -------------------------
    y_d_np = np.random.uniform(0.0, H / y_ref, (N_DOMAIN, 1))
    x_d_np = np.random.uniform(0.0, L / x_ref, (N_DOMAIN, 1))
    t_d_np = np.random.uniform(t0_star, t1_star, (N_DOMAIN, 1))

    y_d = torch.tensor(y_d_np, dtype=torch.float32, device=device, requires_grad=True)
    x_d = torch.tensor(x_d_np, dtype=torch.float32, device=device, requires_grad=True)
    t_d = torch.tensor(t_d_np, dtype=torch.float32, device=device, requires_grad=True)

    # -------------------------
    # Upper wall at y*=1
    # -------------------------
    y_wall_upper_np = np.ones((N_WALL_UPPER, 1)) * (H / y_ref)
    x_wall_upper_np = np.random.uniform(0.0, L / x_ref, (N_WALL_UPPER, 1))
    t_wall_upper_np = np.random.uniform(t0_star, t1_star, (N_WALL_UPPER, 1))

    y_wall_upper = torch.tensor(y_wall_upper_np, dtype=torch.float32, device=device)
    x_wall_upper = torch.tensor(x_wall_upper_np, dtype=torch.float32, device=device)
    t_wall_upper = torch.tensor(t_wall_upper_np, dtype=torch.float32, device=device)

    # -------------------------
    # Lower wall at y*=0
    # -------------------------
    y_wall_lower_np = np.zeros((N_WALL_LOWER, 1))
    x_wall_lower_np = np.random.uniform(0.0, L / x_ref, (N_WALL_LOWER, 1))
    t_wall_lower_np = np.random.uniform(t0_star, t1_star, (N_WALL_LOWER, 1))

    y_wall_lower = torch.tensor(y_wall_lower_np, dtype=torch.float32, device=device)
    x_wall_lower = torch.tensor(x_wall_lower_np, dtype=torch.float32, device=device)
    t_wall_lower = torch.tensor(t_wall_lower_np, dtype=torch.float32, device=device)

    # -------------------------
    # Inlet at x*=0
    # -------------------------
    y_in_np = np.random.uniform(0.0, H / y_ref, (N_INLET, 1))
    x_in_np = np.zeros((N_INLET, 1))
    t_in_np = np.random.uniform(t0_star, t1_star, (N_INLET, 1))

    y_in = torch.tensor(y_in_np, dtype=torch.float32, device=device)
    x_in = torch.tensor(x_in_np, dtype=torch.float32, device=device)
    t_in = torch.tensor(t_in_np, dtype=torch.float32, device=device)

    # -------------------------
    # Outlet at x*=L/x_ref
    # -------------------------
    y_out_np = np.random.uniform(0.0, H / y_ref, (N_OUTLET, 1))
    x_out_np = np.ones((N_OUTLET, 1)) * (L / x_ref)
    t_out_np = np.random.uniform(t0_star, t1_star, (N_OUTLET, 1))

    y_out = torch.tensor(y_out_np, dtype=torch.float32, device=device)
    x_out = torch.tensor(x_out_np, dtype=torch.float32, device=device)
    t_out = torch.tensor(t_out_np, dtype=torch.float32, device=device)

    if print_summary:
        print("\n" + "=" * 60)
        print(f"Built training window: t ∈ [{t_initial}, {t_window_max}] s")
        print(f"Collocation points used: {N_DOMAIN}")
        print(f"Collocation time range: [{t0_star:.4f}, {t1_star:.4f}] in nondimensional time")
        print("=" * 60)

def build_initial_condition_points():
    global y_ic, x_ic, t_ic
    global u_y_ic, u_x_ic, p_ic

    y_ic_np = np.random.uniform(0.0, H / y_ref, (N_INITIAL, 1))
    x_ic_np = np.random.uniform(0.0, L / x_ref, (N_INITIAL, 1))
    t_ic_np = np.zeros((N_INITIAL, 1))

    y_ic = torch.tensor(y_ic_np, dtype=torch.float32, device=device)
    x_ic = torch.tensor(x_ic_np, dtype=torch.float32, device=device)
    t_ic = torch.tensor(t_ic_np, dtype=torch.float32, device=device)

    y_dim_ic = y_ic * y_ref
    x_dim_ic = x_ic * x_ref

    u_x_ic = ((deltaP / (2.0 * mu * L)) * y_dim_ic * (H - y_dim_ic)) / u_ref
    u_y_ic = torch.zeros((N_INITIAL, 1), dtype=torch.float32, device=device)
    p_ic = (deltaP * (1.0 - x_dim_ic / L)) / p_ref

def p_in_star(t_star):
    t_dim = t_star * t_ref

    return (
        A0 / p_ref
        + (A1 / p_ref) * torch.sin(2.0 * np.pi * t_dim / T)
    )

# Preserve the historical sampling sequence used in the thesis runs.
# The full-window build advances the NumPy RNG before the initial-condition points are sampled.
build_training_window(t_final, print_summary=False)
build_initial_condition_points()

# =========================
# Navier-Stokes residual calculator
# =========================

def NS_res_calc(
    y, x, t,
    u_y, u_x,
    u_y_y, u_y_yy, u_y_x, u_y_xx, u_y_t,
    u_x_x, u_x_xx, u_x_y, u_x_yy, u_x_t,
    p_y, p_x
):
    residual_c = alpha * u_x_x + u_y_y

    NS_terms_y = {
        "time_y": St * u_y_t,
        "conv_y": u_x * alpha * u_y_x + u_y * u_y_y,
        "press_y": p_ref / (rho * u_ref**2) * p_y,
        "visc_y": (-1.0 / Re) * (alpha**2 * u_y_xx + u_y_yy),
    }

    NS_terms_x = {
        "time_x": St * u_x_t,
        "conv_x": u_x * alpha * u_x_x + u_y * u_x_y,
        "press_x": alpha * p_ref / (rho * u_ref**2) * p_x,
        "visc_x": (-1.0 / Re) * (alpha**2 * u_x_xx + u_x_yy),
    }

    residual_y = sum(NS_terms_y.values())
    residual_x = sum(NS_terms_x.values())

    return residual_c, residual_y, residual_x, NS_terms_y, NS_terms_x

# =========================
# Loss balancing weights
# =========================

weights = {
    "Cont": 1.0,
    "NS_y": 1.0,
    "NS_x": 1.0,
    "Inlet": 1.0,
    "Outlet": 1.0,
    "Initial": 1.0,
}

# =========================
# Histories creation
# =========================

loss_history = []
epoch_history = []

loss_history_lbfgs = []
lbfgs_iter_history = []
lbfgs_iter = 0

loss_term_history = {key: [] for key in weights}

lr_history = []
lr_change_history = []

# =========================
# Gradient calculator
# =========================

def grad_calc(y, x):
    grads = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

    return grads


# =========================
# Loss function
# =========================

def loss_function(return_diag=False):

    # ---- Physics loss calculation ----
    output = model(y_d, x_d, t_d)

    u_y_pred = output[:, 0:1]
    u_x_pred = output[:, 1:2]
    p_pred = output[:, 2:3]

    # Gradients
    u_y_y = grad_calc(u_y_pred, y_d)
    u_y_yy = grad_calc(u_y_y, y_d)
    u_y_x = grad_calc(u_y_pred, x_d)
    u_y_xx = grad_calc(u_y_x, x_d)
    u_y_t = grad_calc(u_y_pred, t_d)

    u_x_y = grad_calc(u_x_pred, y_d)
    u_x_yy = grad_calc(u_x_y, y_d)
    u_x_x = grad_calc(u_x_pred, x_d)
    u_x_xx = grad_calc(u_x_x, x_d)
    u_x_t = grad_calc(u_x_pred, t_d)

    p_y = grad_calc(p_pred, y_d)
    p_x = grad_calc(p_pred, x_d)

    residual_c, residual_y, residual_x, NS_terms_y, NS_terms_x = NS_res_calc(
        y_d, x_d, t_d,
        u_y_pred, u_x_pred,
        u_y_y, u_y_yy, u_y_x, u_y_xx, u_y_t,
        u_x_x, u_x_xx, u_x_y, u_x_yy, u_x_t,
        p_y, p_x
    )

    loss_c = torch.mean(residual_c**2)
    loss_y = torch.mean(residual_y**2)
    loss_x = torch.mean(residual_x**2)

    if return_diag:
        diag_y = {
            "|Time_y|": NS_terms_y["time_y"].abs().mean().item(),
            "|Conv_y|": NS_terms_y["conv_y"].abs().mean().item(),
            "|Press_y|": NS_terms_y["press_y"].abs().mean().item(),
            "|Visc_y|": NS_terms_y["visc_y"].abs().mean().item(),
            "Res_y_mean": residual_y.mean().item(),
            "|Res_y_max|": residual_y.abs().max().item(),
        }

        diag_x = {
            "|Time_x|": NS_terms_x["time_x"].abs().mean().item(),
            "|Conv_x|": NS_terms_x["conv_x"].abs().mean().item(),
            "|Press_x|": NS_terms_x["press_x"].abs().mean().item(),
            "|Visc_x|": NS_terms_x["visc_x"].abs().mean().item(),
            "Res_x_mean": residual_x.mean().item(),
            "|Res_x_max|": residual_x.abs().max().item(),
        }
    else:
        diag_y = diag_x = None

    # ---- Inlet loss ----
    # Pressure is imposed by hard ansatz.
    # Only the normal velocity u_y is penalized at inlet.
    output_in = model(y_in, x_in, t_in)

    u_y_in_pred = output_in[:, 0:1]

    loss_in = torch.mean(u_y_in_pred**2)

    # ---- Outlet loss ----
    # Pressure is imposed by hard ansatz.
    # Only the normal velocity u_y is penalized at outlet.
    output_out = model(y_out, x_out, t_out)

    u_y_out_pred = output_out[:, 0:1]

    loss_out = torch.mean(u_y_out_pred**2)

    # ---- Initial condition loss ----
    output_ic = model(y_ic, x_ic, t_ic)

    u_y_ic_pred = output_ic[:, 0:1]
    u_x_ic_pred = output_ic[:, 1:2]
    p_ic_pred = output_ic[:, 2:3]

    loss_ic = torch.mean(
        (u_y_ic_pred - u_y_ic)**2
        + (u_x_ic_pred - u_x_ic)**2
        + (p_ic_pred - p_ic)**2
    )

    # ---- Total loss ----
    loss_terms = {
        "Cont": loss_c,
        "NS_y": loss_y,
        "NS_x": loss_x,
        "Inlet": loss_in,
        "Outlet": loss_out,
        "Initial": loss_ic,
    }

    loss = sum(weights[key] * loss_terms[key] for key in loss_terms)

    return loss, loss_terms, diag_y, diag_x

# =========================
# Output utilities
# =========================

def num_to_tag(value, decimals=4):
    s = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s

def sanitize_filename(name):
    filename_aliases = {
        "|v|": "v_mag",
        "|v|_comparison": "v_mag_comparison",
        "$p$_comparison": "p_comparison",
    }

    safe_name = filename_aliases.get(str(name), str(name))

    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        safe_name = safe_name.replace(char, "_")

    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")

    safe_name = safe_name.strip("._ ")

    return safe_name

def stop_if_output_exists(path):
    path = Path(path)

    if path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        raise FileExistsError(
            f"Output already exists and would be overwritten:\n"
            f"{path}\n\n"
            f"Delete the existing output or set OVERWRITE_EXISTING_OUTPUTS = True."
        )

def check_existing_outputs_before_run():
    if TRAIN_MODE and SAVE_CHECKPOINT:
        stop_if_output_exists(CHECKPOINT_PATH)

    if SAVE_PLOT_DATA:
        stop_if_output_exists(PLOT_DATA_PATH)

    if (
        CREATE_PLOTS
        and RUN_FIGURES_DIR.exists()
        and any(RUN_FIGURES_DIR.iterdir())
        and not OVERWRITE_EXISTING_OUTPUTS
    ):
        raise FileExistsError(
            f"Plot folder already exists and is not empty:\n"
            f"{RUN_FIGURES_DIR}\n\n"
            f"Delete the existing folder or set "
            f"OVERWRITE_EXISTING_OUTPUTS = True."
        )

def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

# =========================
# Checkpoint save / load
# =========================

def build_run_config(final_stage_name=None):
    return {
        "stage": "stage3_pulsatile_forward",
        "formulation": "nondimensional",
        "weight_initialization": "xavier_normal",
        "activation": "tanh",
        "hard_bcs": True,
        "hard_pressure": True,
        "collocation_sampling": "random_uniform",

        "run_name": RUN_NAME,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "figures_dir": str(RUN_FIGURES_DIR),
        "plot_data_path": str(PLOT_DATA_PATH),

        "seed": seed,
        "seed_mode": "random" if RANDOM_SEED else "fixed",

        "physical_parameters": {
            "mu": mu,
            "rho": rho,
            "A0": A0,
            "A1": A1,
            "T": T,
            "p_out_dim": p_out_dim,
            "t_initial": t_initial,
            "t_final": t_final,
            "cycles": cycles,
            "L": L,
            "H": H,
        },

        "scales": {
            "x_ref": x_ref,
            "y_ref": y_ref,
            "t_ref": t_ref,
            "u_ref": u_ref,
            "p_ref": p_ref,
            "alpha": alpha,
            "Re": Re,
            "St": St,
        },

        "architecture": {
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "hidden_layers": HIDDEN_LAYERS,
            "neurons": NEURONS,
            "layers": layers,
        },

        "training_points": {
            "N_DOMAIN": N_DOMAIN,
            "N_WALL_UPPER": N_WALL_UPPER,
            "N_WALL_LOWER": N_WALL_LOWER,
            "N_INLET": N_INLET,
            "N_OUTLET": N_OUTLET,
            "N_INITIAL": N_INITIAL,
        },

        "training": {
            "training_schedule": training_schedule,
            "total_adam_epochs": configured_total_adam_epochs,
            "time_windows_phys": TIME_WINDOWS_PHYS,
            "epochs_per_window": EPOCHS_PER_WINDOW,
            "LR_ADAM": LR_ADAM,
            "LR_LBFGS": LR_LBFGS,
            "max_lbfgs_iterations": MAX_LBFGS_ITERATIONS,
            "run_lbfgs": True,
            "RESET_LR_EACH_WINDOW": RESET_LR_EACH_WINDOW,
        },

        "loss_weights": weights,
        "final_stage_name": final_stage_name,
    }


def save_stage3_checkpoint(final_stage_name, adam_time, lbfgs_time, optimizer_time, overhead_time, total_time):
    checkpoint = {
        "model_state_dict": model.state_dict(),

        "network_state": {
            key: value for key, value in model.state_dict().items()
            if key.startswith("layers.")
        },

        "layers": layers,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_layers": HIDDEN_LAYERS,
        "neurons": NEURONS,
        "activation": "tanh",

        "loss_history": loss_history,
        "epoch_history": epoch_history,
        "loss_history_lbfgs": loss_history_lbfgs,
        "lbfgs_iter_history": lbfgs_iter_history,
        "loss_term_history": loss_term_history,

        "weights": weights,
        "lr_history": lr_history,
        "lr_change_history": lr_change_history,

        "seed": seed,

        "adam_time": adam_time,
        "lbfgs_time": lbfgs_time,
        "optimizer_time": optimizer_time,
        "overhead_time": overhead_time,
        "total_time": total_time,

        "run_config": build_run_config(final_stage_name),

        "run_name": RUN_NAME,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "figures_dir": str(RUN_FIGURES_DIR),

        "hard_bcs": True,
        "hard_pressure": True,
        "training_schedule": training_schedule,
        "total_adam_epochs": configured_total_adam_epochs,
        "time_windows_phys": TIME_WINDOWS_PHYS,
        "epochs_per_window": EPOCHS_PER_WINDOW,
        "RESET_LR_EACH_WINDOW": RESET_LR_EACH_WINDOW,
        "run_lbfgs": True,

        "LR_ADAM": LR_ADAM,
        "LR_LBFGS": LR_LBFGS,
        "max_lbfgs_iterations": MAX_LBFGS_ITERATIONS,

        "adam_scheduler": "MultiStepLR",
        "final_stage_name": final_stage_name,

        "environment": {
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
        },
    }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stop_if_output_exists(CHECKPOINT_PATH)

    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nCheckpoint saved to: {CHECKPOINT_PATH}")


def load_stage3_checkpoint():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Run first with TRAIN_MODE = True and SAVE_CHECKPOINT = True."
        )

    checkpoint = load_torch_checkpoint(CHECKPOINT_PATH)

    if checkpoint.get("layers") != layers:
        raise ValueError(
            "Loaded Stage 3 checkpoint architecture does not match "
            "the current model.\n"
            f"Checkpoint layers: {checkpoint.get('layers')}\n"
            f"Current layers:    {layers}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Stage 3 checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\nCheckpoint loaded from: {CHECKPOINT_PATH}")

    return checkpoint

# =========================
# Adam + L-BFGS training or checkpoint loading
# =========================

check_existing_outputs_before_run()

if TRAIN_MODE:

    # =========================
    # Initial network state before any Adam optimizer step
    # =========================

    first_time_window, _ = training_schedule[0]
    build_training_window(first_time_window, print_summary=False)

    print("\n" + "=" * 60)
    print("Initial network state")
    print("=" * 60)

    loss, loss_terms, diag_y, diag_x = loss_function(return_diag=True)

    loss_history.append(loss.item())
    epoch_history.append(0)

    for key in weights:
        loss_term_history[key].append(loss_terms[key].item())

    weighted_loss_terms = {
        key: weights[key] * loss_terms[key].item()
        for key in weights
    }

    weighted_loss_total = sum(weighted_loss_terms.values()) + 1e-12

    loss_term_percentages = {
        key: 100.0 * weighted_loss_terms[key] / weighted_loss_total
        for key in weights
    }

    print(f"Epoch 0, Total Loss: {loss.item():.4e}")

    print("\n--- Loss terms ---")
    for key in weights:
        print(f"{key:12s}: {loss_terms[key].item():.4e}  ({loss_term_percentages[key]:.2f}%)")

    print("\n--- NS_y diagnostics ---")
    for key, value in diag_y.items():
        print(f"{key:12s}: {value:+.4e}")

    print("\n--- NS_x diagnostics ---")
    for key, value in diag_x.items():
        print(f"{key:12s}: {value:+.4e}")

    # =========================
    # Adam training
    # =========================

    total_start_time = sync_time()
    adam_start_time = sync_time()

    print("\n" + "=" * 60)
    print("Starting Adam optimizer...")
    print("=" * 60)

    print("\n" + "=" * 60)
    print(f"Planned total Adam epochs: {configured_total_adam_epochs}")
    print("=" * 60)

    global_epoch = 0

    for stage_id, (t_window_max, stage_epochs) in enumerate(training_schedule, start=1):

        print("\n" + "=" * 60)
        print(f"Adam stage {stage_id}/{len(training_schedule)}")
        print(f"Training window: t <= {t_window_max} s")
        print(f"Stage Adam epochs: {stage_epochs}")
        print("=" * 60)

        if stage_id > 1:
            build_training_window(t_window_max)

        if RESET_LR_EACH_WINDOW:
            for group in optimizer_adam.param_groups:
                group["lr"] = LR_ADAM

        scheduler = make_adam_scheduler(optimizer_adam, stage_epochs)
        previous_lr = optimizer_adam.param_groups[0]["lr"]

        for local_epoch in range(1, stage_epochs + 1):

            global_epoch += 1

            lr_used = optimizer_adam.param_groups[0]["lr"]

            optimizer_adam.zero_grad()

            do_print = (global_epoch % 500 == 0)

            loss, loss_terms, diag_y, diag_x = loss_function(return_diag=do_print)
            loss.backward()

            optimizer_adam.step()

            scheduler.step()

            current_lr = optimizer_adam.param_groups[0]["lr"]

            lr_history.append(lr_used)

            if current_lr != previous_lr:
                lr_change_history.append((global_epoch, previous_lr, current_lr))
                print(
                    f"[LR CHANGE] Global epoch {global_epoch}: "
                    f"{previous_lr:.4e} -> {current_lr:.4e}"
                )
                previous_lr = current_lr

            loss_history.append(loss.item())
            epoch_history.append(global_epoch)

            for key in weights:
                loss_term_history[key].append(loss_terms[key].item())

            weighted_loss_terms = {
                key: weights[key] * loss_terms[key].item()
                for key in weights
            }

            weighted_loss_total = sum(weighted_loss_terms.values()) + 1e-12

            loss_term_percentages = {
                key: 100.0 * weighted_loss_terms[key] / weighted_loss_total
                for key in weights
            }

            if global_epoch % 500 == 0:

                print(
                    f"Stage {stage_id}/{len(training_schedule)}, "
                    f"Epoch {global_epoch}/{configured_total_adam_epochs}, "
                    f"Stage progress {local_epoch}/{stage_epochs}, "
                    f"Total Loss: {loss.item():.4e}"
                )

                print("\n--- Loss terms ---")
                for key in weights:
                    print(f"{key:12s}: {loss_terms[key].item():.4e}  ({loss_term_percentages[key]:.2f}%)")

                print("\n--- NS_y diagnostics ---")
                for key, value in diag_y.items():
                    print(f"{key:12s}: {value:+.4e}")

                print("\n--- NS_x diagnostics ---")
                for key, value in diag_x.items():
                    print(f"{key:12s}: {value:+.4e}")

                print(f"\nAdam learning rate for next epoch: {current_lr:.4e}")
                print("-" * 60)

    if global_epoch != configured_total_adam_epochs:
        raise RuntimeError(
            f"Adam epoch counter mismatch: "
            f"global_epoch={global_epoch}, expected={configured_total_adam_epochs}"
        )

    adam_end_time = sync_time()
    adam_time = adam_end_time - adam_start_time

    print("\n" + "=" * 60)
    print(f"Adam training time: {adam_time:.2f} s ({adam_time/60:.2f} min)")
    print("=" * 60)

    print("\n=== Adam learning rate changes ===")
    if len(lr_change_history) == 0:
        print("No learning rate changes occurred during Adam training.")
    else:
        for epoch, old_lr, new_lr in lr_change_history:
            print(f"Epoch {epoch}: {old_lr:.4e} -> {new_lr:.4e}")

    # =========================
    # L-BFGS training
    # =========================

    print("\n" + "=" * 60)
    print("Starting L-BFGS optimizer...")
    print("=" * 60)

    lbfgs_start_time = sync_time()

    optimizer_lbfgs.step(closure)

    lbfgs_end_time = sync_time()
    lbfgs_time = lbfgs_end_time - lbfgs_start_time

    print("\n" + "=" * 60)
    print(f"L-BFGS training time: {lbfgs_time:.2f} s ({lbfgs_time/60:.2f} min)")
    print("=" * 60)

    print(f"L-BFGS closure evaluations recorded: {len(loss_history_lbfgs)}")

    final_stage_name = "after L-BFGS"

    loss, loss_terms, diag_y, diag_x = loss_function(return_diag=True)

    weighted_loss_terms = {
        key: weights[key] * loss_terms[key].item()
        for key in weights
    }

    weighted_loss_total = sum(weighted_loss_terms.values()) + 1e-12

    loss_term_percentages = {
        key: 100.0 * weighted_loss_terms[key] / weighted_loss_total
        for key in weights
    }

    print(f"\n--- Final loss report {final_stage_name} ---")
    print(f"Total Loss: {loss.item():.6e}")

    print("\n--- Final loss terms ---")
    for key in weights:
        print(f"{key:12s}: {loss_terms[key].item():.4e}  ({loss_term_percentages[key]:.2f}%)")

    print("\n--- Final NS_y diagnostics ---")
    for key, value in diag_y.items():
        print(f"{key:12s}: {value:+.4e}")

    print("\n--- Final NS_x diagnostics ---")
    for key, value in diag_x.items():
        print(f"{key:12s}: {value:+.4e}")

    total_end_time = sync_time()
    total_time = total_end_time - total_start_time

    optimizer_time = adam_time + lbfgs_time
    overhead_time = max(0.0, total_time - optimizer_time)

    print("\n" + "=" * 60)
    print("Training time summary")
    print("=" * 60)
    print(f"Adam time        : {adam_time:.2f} s ({adam_time/60:.2f} min)")
    print(f"L-BFGS time      : {lbfgs_time:.2f} s ({lbfgs_time/60:.2f} min)")
    print(f"Optimizer time   : {optimizer_time:.2f} s ({optimizer_time/60:.2f} min)")
    print(f"Overhead time    : {overhead_time:.2f} s ({overhead_time/60:.2f} min)")
    print(f"Total train time : {total_time:.2f} s ({total_time/60:.2f} min)")
    print("=" * 60)

    if SAVE_CHECKPOINT:
        save_stage3_checkpoint(
            final_stage_name=final_stage_name,
            adam_time=adam_time,
            lbfgs_time=lbfgs_time,
            optimizer_time=optimizer_time,
            overhead_time=overhead_time,
            total_time=total_time
        )

else:

    print("\n" + "=" * 60)
    print("Loading trained model and skipping training...")
    print("=" * 60)

    checkpoint = load_stage3_checkpoint()

    loss_history = checkpoint.get("loss_history", [])
    epoch_history = checkpoint.get("epoch_history", [])
    loss_history_lbfgs = checkpoint.get("loss_history_lbfgs", [])
    lbfgs_iter_history = checkpoint.get("lbfgs_iter_history", [])
    loss_term_history = checkpoint.get("loss_term_history", loss_term_history)
    weights = checkpoint.get("weights", weights)
    lr_history = checkpoint.get("lr_history", [])
    lr_change_history = checkpoint.get("lr_change_history", [])

    adam_time = checkpoint.get("adam_time", 0.0)
    lbfgs_time = checkpoint.get("lbfgs_time", 0.0)
    optimizer_time = checkpoint.get("optimizer_time", 0.0)
    overhead_time = checkpoint.get("overhead_time", 0.0)
    total_time = checkpoint.get("total_time", 0.0)

    final_stage_name = checkpoint.get("final_stage_name", "loaded checkpoint")

    loaded_run_config = checkpoint.get("run_config", {})

    print("\nLoaded run metadata:")
    print(f"Run name          : {checkpoint.get('run_name', RUN_NAME)}")
    print(f"Final saved stage : {final_stage_name}")
    print(f"Hard BCs          : {checkpoint.get('hard_bcs', True)}")
    print(f"Hard pressure     : {checkpoint.get('hard_pressure', True)}")
    print(f"Time windows      : {checkpoint.get('time_windows_phys', TIME_WINDOWS_PHYS)}")
    print(f"Epochs per window : {checkpoint.get('epochs_per_window', EPOCHS_PER_WINDOW)}")
    print(f"Total Adam epochs : {checkpoint.get('total_adam_epochs', configured_total_adam_epochs)}")
    print(f"Original seed     : {checkpoint.get('seed', seed)}")

    if total_time > 0:
        print("\nLoaded timing information:")
        print(f"Adam time        : {adam_time:.2f} s ({adam_time/60:.2f} min)")
        print(f"L-BFGS time      : {lbfgs_time:.2f} s ({lbfgs_time/60:.2f} min)")
        print(f"Optimizer time   : {optimizer_time:.2f} s ({optimizer_time/60:.2f} min)")
        print(f"Overhead time    : {overhead_time:.2f} s ({overhead_time/60:.2f} min)")
        print(f"Total train time : {total_time:.2f} s ({total_time/60:.2f} min)")

    if len(loaded_run_config) > 0:
        print("\nFull loaded run_config:")
        for key, value in loaded_run_config.items():
            print(f"{key:32s}: {value}")

# =========================
# Evaluation
# =========================

model.eval()

with torch.no_grad():
    output_data = model(y_data, x_data, t_data)

    u_y_pred = output_data[:, 0:1] * u_ref
    u_x_pred = output_data[:, 1:2] * u_ref
    p_pred = output_data[:, 2:3] * p_ref


# =========================
# Error metrics
# =========================
def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true)**2))

def l2_error(pred, true):
    return torch.norm(pred - true, p=2)

def relative_l2(pred, true):
    return l2_error(pred, true) / (torch.norm(true, p=2) + 1e-12)

# u_y in m/s
rmse_u_y_dim = rmse(u_y_pred, u_y_data)
l2_u_y_dim = l2_error(u_y_pred, u_y_data)

# u_x in m/s
rmse_u_x_dim = rmse(u_x_pred, u_x_data)
l2_u_x_dim = l2_error(u_x_pred, u_x_data)
rel_l2_u_x_dim = relative_l2(u_x_pred, u_x_data)

# Velocity magnitude |v| = sqrt(u_y^2 + u_x^2)
v_mag_pred = torch.sqrt(u_y_pred**2 + u_x_pred**2)
v_mag_data = torch.sqrt(u_y_data**2 + u_x_data**2)

rmse_v_mag_dim = rmse(v_mag_pred, v_mag_data)
l2_v_mag_dim = l2_error(v_mag_pred, v_mag_data)
rel_l2_v_mag_dim = relative_l2(v_mag_pred, v_mag_data)

# p in Pa
rmse_p_dim = rmse(p_pred, p_data)
l2_p_dim = l2_error(p_pred, p_data)
rel_l2_p_dim = relative_l2(p_pred, p_data)

print("\n=== Field errors ===")

print(f"RMSE u_y (m/s):              {rmse_u_y_dim.item():.6e}")
print(f"L2 error u_y (m/s):          {l2_u_y_dim.item():.6e}")
print("Relative L2 error u_y:       not representative because the L2 norm of the reference field is close to zero.")

print("-" * 60)

print(f"RMSE u_x (m/s):              {rmse_u_x_dim.item():.6e}")
print(f"L2 error u_x (m/s):          {l2_u_x_dim.item():.6e}")
print(f"Relative L2 error u_x:       {rel_l2_u_x_dim.item():.6e} ({100 * rel_l2_u_x_dim.item():.4f}%)")

print("-" * 60)

print(f"RMSE |v| (m/s):              {rmse_v_mag_dim.item():.6e}")
print(f"L2 error |v| (m/s):          {l2_v_mag_dim.item():.6e}")
print(f"Relative L2 error |v|:       {rel_l2_v_mag_dim.item():.6e} ({100 * rel_l2_v_mag_dim.item():.4f}%)")

print("-" * 60)

print(f"RMSE p (Pa):                 {rmse_p_dim.item():.6e}")
print(f"L2 error p (Pa):             {l2_p_dim.item():.6e}")
print(f"Relative L2 error p:         {rel_l2_p_dim.item():.6e} ({100 * rel_l2_p_dim.item():.4f}%)")

# =========================
# Per-time-slice error
# =========================

t_np = t_data.detach().cpu().numpy().flatten()
uniq_t = np.unique(t_np)

print("\n=== Per-time-slice error ===")
print(f"{'t (s)':>7} | {'u_x rel':>9} | {'p rel':>9} | {'p RMSE (Pa)':>11}")
print("-" * 48)

for tj in uniq_t:
    idx = torch.tensor(np.where(np.isclose(t_np, tj))[0], device=device)

    eu = 100.0 * relative_l2(u_x_pred[idx], u_x_data[idx]).item()
    ep = 100.0 * relative_l2(p_pred[idx], p_data[idx]).item()
    p_rmse = rmse(p_pred[idx], p_data[idx]).item()

    print(f"{tj*t_ref:7.2f} | {eu:8.3f}% | {ep:8.3f}% | {p_rmse:9.4f}")

# =========================
# Boundary checks
# =========================

with torch.no_grad():
    output_wall_upper = model(y_wall_upper, x_wall_upper, t_wall_upper)

    u_y_wall_pred_upper = output_wall_upper[:, 0:1] * u_ref
    u_x_wall_pred_upper = output_wall_upper[:, 1:2] * u_ref

    output_wall_lower = model(y_wall_lower, x_wall_lower, t_wall_lower)

    u_y_wall_pred_lower = output_wall_lower[:, 0:1] * u_ref
    u_x_wall_pred_lower = output_wall_lower[:, 1:2] * u_ref

    output_in = model(y_in, x_in, t_in)
    output_out = model(y_out, x_out, t_out)

    p_in_pred = output_in[:, 2:3] * p_ref
    p_out_pred = output_out[:, 2:3] * p_ref

    p_in_target = p_in_star(t_in) * p_ref
    err_in = torch.abs(p_in_pred - p_in_target)

    u_y_in_pred = output_in[:, 0:1] * u_ref
    u_y_out_pred = output_out[:, 0:1] * u_ref

print("\n=== Boundary checks ===")
print(f"Max |u_y_wall_upper| (m/s)  :   {torch.max(torch.abs(u_y_wall_pred_upper)).item():.6e}")
print(f"Mean |u_y_wall_upper| (m/s) :   {torch.mean(torch.abs(u_y_wall_pred_upper)).item():.6e}")
print(f"Max |u_x_wall_upper| (m/s)  :   {torch.max(torch.abs(u_x_wall_pred_upper)).item():.6e}")
print(f"Mean |u_x_wall_upper| (m/s) :   {torch.mean(torch.abs(u_x_wall_pred_upper)).item():.6e}")
print("-" * 60)
print(f"Max |u_y_wall_lower| (m/s)  :   {torch.max(torch.abs(u_y_wall_pred_lower)).item():.6e}")
print(f"Mean |u_y_wall_lower| (m/s) :   {torch.mean(torch.abs(u_y_wall_pred_lower)).item():.6e}")
print(f"Max |u_x_wall_lower| (m/s)  :   {torch.max(torch.abs(u_x_wall_pred_lower)).item():.6e}")
print(f"Mean |u_x_wall_lower| (m/s) :   {torch.mean(torch.abs(u_x_wall_pred_lower)).item():.6e}")
print("-" * 60)
print(f"Max predicted p_outlet (Pa) :   {torch.max(torch.abs(p_out_pred)).item():.6e}")
print(f"Mean predicted p_outlet (Pa):   {torch.mean(p_out_pred).item():.6e}")
print(f"Target p_outlet (Pa)        :   {p_out_dim:.6e}")
print("-" * 60)
print(f"Max |p_inlet - target| (Pa) :   {err_in.max().item():.6e}")
print(f"Mean |p_inlet - target| (Pa):   {err_in.mean().item():.6e}")
print(f"Inlet target range (Pa)     :   [{p_in_target.min().item():.3f}, {p_in_target.max().item():.3f}]")
print("-" * 60)
print(f"Max predicted u_y_inlet (m/s)   :   {torch.max(torch.abs(u_y_in_pred)).item():.6e}")
print(f"Mean predicted u_y_inlet (m/s)  :   {torch.mean(u_y_in_pred).item():.6e}")
print(f"Max predicted u_y_outlet (m/s)  :   {torch.max(torch.abs(u_y_out_pred)).item():.6e}")
print(f"Mean predicted u_y_outlet (m/s) :   {torch.mean(u_y_out_pred).item():.6e}")

print("=" * 60)
print(f"Seed used for this run: {seed}")
print("=" * 60)

# =========================
# Convert to numpy for plots
# =========================

y_plot = (y_data * y_ref).detach().cpu().numpy().flatten()
x_plot = (x_data * x_ref).detach().cpu().numpy().flatten()
t_plot = (t_data * t_ref).detach().cpu().numpy().flatten()

u_y_pred_np = u_y_pred.detach().cpu().numpy().flatten()
u_x_pred_np = u_x_pred.detach().cpu().numpy().flatten()
p_pred_np = p_pred.detach().cpu().numpy().flatten()

u_y_data_np = u_y_data.detach().cpu().numpy().flatten()
u_x_data_np = u_x_data.detach().cpu().numpy().flatten()
p_data_np = p_data.detach().cpu().numpy().flatten()

v_mag_pred_np = v_mag_pred.detach().cpu().numpy().flatten()
v_mag_data_np = v_mag_data.detach().cpu().numpy().flatten()

u_y_pred_wide = u_y_pred_np.reshape(n_nodes, n_times)
u_x_pred_wide = u_x_pred_np.reshape(n_nodes, n_times)
p_pred_wide = p_pred_np.reshape(n_nodes, n_times)
v_mag_pred_wide = v_mag_pred_np.reshape(n_nodes, n_times)

v_mag_data_wide = v_mag_data_np.reshape(n_nodes, n_times)

abs_err_u_y = np.abs(u_y_data_np - u_y_pred_np)
abs_err_u_x = np.abs(u_x_data_np - u_x_pred_np)
abs_err_p = np.abs(p_data_np - p_pred_np)
abs_err_v_mag = np.abs(v_mag_data_np - v_mag_pred_np)


# =========================
# Save plot data
# =========================

if SAVE_PLOT_DATA:
    PLOT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    stop_if_output_exists(PLOT_DATA_PATH)

    np.savez(
        PLOT_DATA_PATH,

        x_plot=x_plot,
        y_plot=y_plot,
        t_plot=t_plot,

        u_y_pred_np=u_y_pred_np,
        u_x_pred_np=u_x_pred_np,
        p_pred_np=p_pred_np,

        u_y_data_np=u_y_data_np,
        u_x_data_np=u_x_data_np,
        p_data_np=p_data_np,

        v_mag_pred_np=v_mag_pred_np,
        v_mag_data_np=v_mag_data_np,

        u_y_pred_wide=u_y_pred_wide,
        u_x_pred_wide=u_x_pred_wide,
        p_pred_wide=p_pred_wide,
        v_mag_pred_wide=v_mag_pred_wide,
        v_mag_data_wide=v_mag_data_wide,

        t_values=t_values,
        x_dim=x_dim,
        y_dim=y_dim,

        abs_err_u_y=abs_err_u_y,
        abs_err_u_x=abs_err_u_x,
        abs_err_p=abs_err_p,
        abs_err_v_mag=abs_err_v_mag,

        seed=seed,
        n_evaluation_points=u_x_data.numel(),

        rmse_u_y=rmse_u_y_dim.item(),
        l2_u_y=l2_u_y_dim.item(),

        rmse_u_x=rmse_u_x_dim.item(),
        l2_u_x=l2_u_x_dim.item(),
        relative_l2_u_x=rel_l2_u_x_dim.item(),

        rmse_v_mag=rmse_v_mag_dim.item(),
        l2_v_mag=l2_v_mag_dim.item(),
        relative_l2_v_mag=rel_l2_v_mag_dim.item(),

        rmse_p=rmse_p_dim.item(),
        l2_p=l2_p_dim.item(),
        relative_l2_p=rel_l2_p_dim.item(),
    )

    print(f"\nPlot data saved to: {PLOT_DATA_PATH}")

# =========================
# Plotting helpers
# =========================

t_slice_desired = 2.60
t_idx = int(np.argmin(np.abs(t_values - t_slice_desired)))
t_slice = t_values[t_idx]

sl = np.arange(n_nodes) * n_times + t_idx

print(f"Field comparison plots at t = {t_slice:.3f} s ({sl.size} nodes)")

triang_slice = tri.Triangulation(x_plot[sl], y_plot[sl])

plot_dir = RUN_FIGURES_DIR


def save_fig(fig, name, dpi=300):
    plot_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(name)
    path = plot_dir / f"{safe_name}.png"

    stop_if_output_exists(path)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


def show_or_close(fig):
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def safe_percent_error(abs_error, true_values, fallback_scale=None, floor_fraction=1e-3, clip_max=100.0):
    """
    Computes a safer relative percentage error.

    If true_values are close to zero, pointwise relative error explodes.
    This function uses a small floor based on the field scale.
    """

    true_values = np.asarray(true_values)
    abs_error = np.asarray(abs_error)

    max_true = np.nanmax(np.abs(true_values))

    if fallback_scale is not None:
        max_fallback = np.nanmax(np.abs(fallback_scale))
        scale = max(max_true, max_fallback)
    else:
        scale = max_true

    if scale <= 1e-14:
        scale = 1.0

    denominator_floor = floor_fraction * scale
    denominator = np.maximum(np.abs(true_values), denominator_floor)

    rel_percent = 100.0 * abs_error / denominator

    return np.clip(rel_percent, 0.0, clip_max)


def get_common_limits(a, b):
    combined = np.concatenate([
        np.asarray(a).flatten(),
        np.asarray(b).flatten()
    ])

    vmin = np.nanmin(combined)
    vmax = np.nanmax(combined)

    if np.isclose(vmin, vmax):
        eps = 1e-12 if abs(vmin) < 1e-12 else 0.01 * abs(vmin)
        vmin -= eps
        vmax += eps

    return vmin, vmax


def get_error_limits(err):
    err = np.asarray(err).flatten()
    vmin = 0.0
    vmax = np.nanpercentile(err, 99)

    if vmax <= 0:
        vmax = np.nanmax(err)

    if vmax <= 0:
        vmax = 1.0

    return vmin, vmax


def setup_axis(ax):
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, H)
    ax.set_aspect("equal", adjustable="box")


def engineering_scale(values):
    vmax = np.nanmax(np.abs(values))

    if (not np.isfinite(vmax)) or vmax <= 0:
        return 1.0, 0

    exponent = int(np.floor(np.log10(vmax)))
    exponent_eng = 3 * int(np.floor(exponent / 3))
    scale = 10.0 ** exponent_eng

    return scale, exponent_eng


def format_scaled_colorbar_label(base_symbol, exponent, unit):
    if exponent == 0:
        return rf"{base_symbol} ({unit})"
    else:
        return rf"{base_symbol} ($\times 10^{{{exponent}}}$ {unit})"


def relative_l2_numpy(pred, ref):
    pred = np.asarray(pred).flatten()
    ref = np.asarray(ref).flatten()

    return 100.0 * np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12)

def plot_field_comparison(
    field_name,
    unit,
    comsol_values,
    pinn_values,
    absolute_error,
    relative_error_percent
):
    field_vmin, field_vmax = get_common_limits(comsol_values, pinn_values)
    abs_vmin, abs_vmax = get_error_limits(absolute_error)

    rel_vmin = 0.0
    rel_vmax = np.nanpercentile(relative_error_percent, 99)

    if rel_vmax <= 0:
        rel_vmax = np.nanmax(relative_error_percent)

    if rel_vmax <= 0:
        rel_vmax = 1.0

    fig, axs = plt.subplots(
        2, 2,
        figsize=(14, 8),
        constrained_layout=True
    )

    fig.suptitle(f"{field_name} comparison for t = {t_slice:.3f} s", fontsize=14)

    cf1 = axs[0, 0].tricontourf(
        triang_slice,
        comsol_values,
        levels=60,
        vmin=field_vmin,
        vmax=field_vmax
    )
    fig.colorbar(cf1, ax=axs[0, 0], label=f"COMSOL {field_name} ({unit})")
    axs[0, 0].set_title(f"COMSOL {field_name}")
    setup_axis(axs[0, 0])

    cf2 = axs[0, 1].tricontourf(
        triang_slice,
        pinn_values,
        levels=60,
        vmin=field_vmin,
        vmax=field_vmax
    )
    fig.colorbar(cf2, ax=axs[0, 1], label=f"PINN {field_name} ({unit})")
    axs[0, 1].set_title(f"PINN {field_name}")
    setup_axis(axs[0, 1])

    cf3 = axs[1, 0].tricontourf(
        triang_slice,
        absolute_error,
        levels=60,
        vmin=abs_vmin,
        vmax=abs_vmax
    )
    fig.colorbar(cf3, ax=axs[1, 0], label=f"|COMSOL - PINN| ({unit})")
    axs[1, 0].set_title(f"Absolute error in {field_name}")
    setup_axis(axs[1, 0])

    cf4 = axs[1, 1].tricontourf(
        triang_slice,
        relative_error_percent,
        levels=60,
        vmin=rel_vmin,
        vmax=rel_vmax
    )
    fig.colorbar(cf4, ax=axs[1, 1], label="Relative error (%)")
    axs[1, 1].set_title(f"Relative error in {field_name} (%)")
    setup_axis(axs[1, 1])

    safe_name = sanitize_filename(field_name)

    save_fig(fig, f"field_{safe_name}_t{num_to_tag(t_slice, decimals=3)}")
    show_or_close(fig)


def plot_three_panel_vertical(
    save_name,
    field_symbol,
    unit,
    comsol_values,
    pinn_values,
    absolute_error,
    cmap="viridis",
    error_exponent=None
):
    field_vmin, field_vmax = get_common_limits(comsol_values, pinn_values)

    if error_exponent is None:
        err_scale, err_exp = engineering_scale(absolute_error)
    else:
        err_exp = error_exponent
        err_scale = 10.0 ** err_exp

    absolute_error_scaled = absolute_error / err_scale

    abs_vmin = 0.0
    abs_vmax = np.nanpercentile(absolute_error_scaled, 99)

    if abs_vmax <= 0:
        abs_vmax = np.nanmax(absolute_error_scaled)

    if abs_vmax <= 0:
        abs_vmax = 1.0

    fig = plt.figure(figsize=(8.5, 12), constrained_layout=True)

    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.0, 0.035],
        height_ratios=[1, 1, 1],
        wspace=0.08,
        hspace=0.18
    )

    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])

    ax2 = fig.add_subplot(gs[1, 0])
    cax2 = fig.add_subplot(gs[1, 1])

    ax3 = fig.add_subplot(gs[2, 0])
    cax3 = fig.add_subplot(gs[2, 1])

    cf1 = ax1.tricontourf(
        triang_slice,
        comsol_values,
        levels=60,
        vmin=field_vmin,
        vmax=field_vmax,
        cmap=cmap
    )

    cb1 = fig.colorbar(cf1, cax=cax1)
    cb1.set_label(rf"{field_symbol} ({unit})")
    cb1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax1.set_title("COMSOL solution", fontsize=15)
    setup_axis(ax1)

    cf2 = ax2.tricontourf(
        triang_slice,
        pinn_values,
        levels=60,
        vmin=field_vmin,
        vmax=field_vmax,
        cmap=cmap
    )

    cb2 = fig.colorbar(cf2, cax=cax2)
    cb2.set_label(rf"{field_symbol} ({unit})")
    cb2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax2.set_title("PINN solution", fontsize=15)
    setup_axis(ax2)

    cf3 = ax3.tricontourf(
        triang_slice,
        absolute_error_scaled,
        levels=60,
        vmin=abs_vmin,
        vmax=abs_vmax,
        cmap=cmap
    )

    cb3 = fig.colorbar(cf3, cax=cax3)
    cb3.set_label(format_scaled_colorbar_label(r"$|\Delta|$", err_exp, unit))
    cb3.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax3.set_title("Absolute deviation", fontsize=15)
    setup_axis(ax3)

    save_fig(fig, save_name)
    show_or_close(fig)


def plot_pressure_time_series_at_point(xp=5.0, yp=2.0):
    t_series = np.linspace(0.0, t_final, 400).reshape(-1, 1)

    x_s = torch.tensor(
        np.full_like(t_series, xp) / x_ref,
        dtype=torch.float32,
        device=device
    )

    y_s = torch.tensor(
        np.full_like(t_series, yp) / y_ref,
        dtype=torch.float32,
        device=device
    )

    t_s = torch.tensor(
        t_series / t_ref,
        dtype=torch.float32,
        device=device
    )

    model.eval()
    with torch.no_grad():
        p_series_pinn = (
            model(y_s, x_s, t_s)[:, 2:3] * p_ref
        ).cpu().numpy().flatten()

    node = int(np.argmin((x_dim.flatten() - xp)**2 + (y_dim.flatten() - yp)**2))

    print(
        f"Nearest COMSOL node to ({xp:.1f},{yp:.1f}): "
        f"x={x_dim.flatten()[node]:.3f}, y={y_dim.flatten()[node]:.3f}"
    )

    p_series_comsol = p_wide[node, :]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        t_series.flatten(),
        p_series_pinn,
        color="C0",
        linewidth=1.8,
        label="PINN"
    )

    ax.plot(
        t_values,
        p_series_comsol,
        "o",
        color="k",
        ms=3,
        alpha=0.6,
        label="COMSOL"
    )

    ax.set_xlabel("t (s)")
    ax.set_ylabel("p (Pa)")
    ax.set_title("Pressure at point (x=5, y=2)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    save_fig(fig, "pressure_at_5_2")
    show_or_close(fig)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def sinusoidal_phase(t_array, signal_array, period):
    t_array = np.asarray(t_array).flatten()
    signal_array = np.asarray(signal_array).flatten()

    omega = 2.0 * np.pi / period

    X = np.column_stack([
        np.ones_like(t_array),
        np.sin(omega * t_array),
        np.cos(omega * t_array)
    ])

    coeffs, _, _, _ = np.linalg.lstsq(X, signal_array, rcond=None)

    c = coeffs[0]
    a = coeffs[1]
    b = coeffs[2]

    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(b, a)

    return phase, amplitude, c


def phase_difference_velocity_pressure(t_array, u_array, p_array, period):
    phi_u, amp_u, mean_u = sinusoidal_phase(t_array, u_array, period)
    phi_p, amp_p, mean_p = sinusoidal_phase(t_array, p_array, period)

    dphi = wrap_to_pi(phi_u - phi_p)
    dphi_deg = np.degrees(dphi)

    omega = 2.0 * np.pi / period
    dt_peak = -dphi / omega

    return {
        "phi_u_rad": phi_u,
        "phi_p_rad": phi_p,
        "dphi_rad": dphi,
        "dphi_deg": dphi_deg,
        "dt_peak": dt_peak,
        "amp_u": amp_u,
        "amp_p": amp_p,
        "mean_u": mean_u,
        "mean_p": mean_p,
    }


def plot_phase_comparison_at_point(xp=5.0, yp=2.0):
    t_series = np.linspace(0.0, t_final, 400).reshape(-1, 1)

    x_s = torch.tensor(
        np.full_like(t_series, xp) / x_ref,
        dtype=torch.float32,
        device=device
    )

    y_s = torch.tensor(
        np.full_like(t_series, yp) / y_ref,
        dtype=torch.float32,
        device=device
    )

    t_s = torch.tensor(
        t_series / t_ref,
        dtype=torch.float32,
        device=device
    )

    model.eval()
    with torch.no_grad():
        out_s = model(y_s, x_s, t_s)

        u_x_series_pinn = (
            out_s[:, 1:2] * u_ref
        ).cpu().numpy().flatten()

        p_series_pinn = (
            out_s[:, 2:3] * p_ref
        ).cpu().numpy().flatten()

    node = int(np.argmin((x_dim.flatten() - xp)**2 + (y_dim.flatten() - yp)**2))

    print(
        f"Nearest COMSOL node to ({xp:.1f},{yp:.1f}): "
        f"x={x_dim.flatten()[node]:.3f}, y={y_dim.flatten()[node]:.3f}"
    )

    u_x_series_comsol = u_x_wide[node, :]
    p_series_comsol = p_wide[node, :]

    phase_pinn = phase_difference_velocity_pressure(
        t_array=t_series.flatten(),
        u_array=u_x_series_pinn,
        p_array=p_series_pinn,
        period=T
    )

    phase_comsol = phase_difference_velocity_pressure(
        t_array=t_values,
        u_array=u_x_series_comsol,
        p_array=p_series_comsol,
        period=T
    )

    print("\n=== Phase difference at (x=5, y=2) ===")
    print("Definition: Δφ = φ_u_x - φ_p")
    print("Positive Δφ means u_x leads pressure p.")
    print("-" * 60)
    print(
        f"PINN  : Δφ = {phase_pinn['dphi_rad']:+.6f} rad "
        f"({phase_pinn['dphi_deg']:+.3f} deg), "
        f"peak time shift u_x - p = {phase_pinn['dt_peak']:+.6f} s"
    )
    print(
        f"COMSOL: Δφ = {phase_comsol['dphi_rad']:+.6f} rad "
        f"({phase_comsol['dphi_deg']:+.3f} deg), "
        f"peak time shift u_x - p = {phase_comsol['dt_peak']:+.6f} s"
    )
    print("-" * 60)
    print(
        f"PINN amplitudes  : amp(u_x) = {phase_pinn['amp_u']:.6e} m/s, "
        f"amp(p) = {phase_pinn['amp_p']:.6e} Pa"
    )
    print(
        f"COMSOL amplitudes: amp(u_x) = {phase_comsol['amp_u']:.6e} m/s, "
        f"amp(p) = {phase_comsol['amp_p']:.6e} Pa"
    )
    print("=" * 60)

    fig, ax_v = plt.subplots(figsize=(11, 6))
    ax_p = ax_v.twinx()

    l1, = ax_v.plot(
        t_series.flatten(),
        u_x_series_pinn,
        color="C0",
        lw=1.8,
        label=r"$u_x$ PINN"
    )

    l2, = ax_v.plot(
        t_values,
        u_x_series_comsol,
        "o",
        color="C0",
        ms=3,
        alpha=0.5,
        label=r"$u_x$ COMSOL"
    )

    ax_v.set_xlabel("t (s)")
    ax_v.set_ylabel(r"$u_x$ (m/s)", color="C0")
    ax_v.tick_params(axis="y", labelcolor="C0")

    l3, = ax_p.plot(
        t_series.flatten(),
        p_series_pinn,
        color="C2",
        lw=1.8,
        label="p PINN"
    )

    l4, = ax_p.plot(
        t_values,
        p_series_comsol,
        "s",
        color="C2",
        ms=3,
        alpha=0.5,
        label="p COMSOL"
    )

    ax_p.set_ylabel("p (Pa)", color="C2")
    ax_p.tick_params(axis="y", labelcolor="C2")

    ax_v.set_title(r"Velocity-pressure phase comparison at $(x=5,\ y=2)$")
    ax_v.grid(alpha=0.3)
    ax_v.legend(handles=[l1, l2, l3, l4], loc="best", fontsize=9)

    plt.tight_layout()
    save_fig(fig, "phase_comparison_at_5_2")
    show_or_close(fig)

def plot_velocity_pressure_profiles_stacked(
    times_to_plot=(2.50, 2.60, 2.70, 2.85),
    x_profile_location=6.0,
    y_pressure_location=2.0,
    n_profile_points=300,
    n_pressure_points=400,
    n_comsol_markers_velocity=18,
    n_comsol_markers_pressure=22
):
    x_flat = x_dim.flatten()
    y_flat = y_dim.flatten()

    triang_full = tri.Triangulation(x_flat, y_flat)

    def interp_comsol(values_at_nodes, x_query, y_query):
        interpolator = tri.LinearTriInterpolator(triang_full, values_at_nodes)
        values_interp = interpolator(x_query, y_query)

        if np.ma.isMaskedArray(values_interp):
            values_interp = values_interp.filled(np.nan)

        return np.asarray(values_interp, dtype=float)

    y_line = np.linspace(0.0, H, n_profile_points)
    x_line_for_velocity = np.full_like(y_line, x_profile_location)

    x_line = np.linspace(0.0, L, n_pressure_points)
    y_line_for_pressure = np.full_like(x_line, y_pressure_location)

    idx_markers_velocity = np.linspace(
        0,
        n_profile_points - 1,
        n_comsol_markers_velocity,
        dtype=int
    )

    idx_markers_pressure = np.linspace(
        0,
        n_pressure_points - 1,
        n_comsol_markers_pressure,
        dtype=int
    )

    time_indices = [int(np.argmin(np.abs(t_values - tt))) for tt in times_to_plot]
    times_used = [t_values[idx] for idx in time_indices]

    print("\n=== Stacked velocity / pressure profiles ===")
    print(f"Velocity profile evaluated at x = {x_profile_location:.3f} m")
    print(f"Pressure distribution evaluated at y = {y_pressure_location:.3f} m")
    print(f"Requested times: {times_to_plot}")
    print(f"Used COMSOL times: {[round(t, 3) for t in times_used]}")
    print("=" * 60)

    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(11, 10),
        sharex=False
    )

    time_handles = []

    for i, (idx_t, used_t) in enumerate(zip(time_indices, times_used)):

        color = colors[i % len(colors)]

        ux_comsol = interp_comsol(
            values_at_nodes=u_x_wide[:, idx_t],
            x_query=x_line_for_velocity,
            y_query=y_line
        )

        x_torch = torch.tensor(
            (x_line_for_velocity / x_ref).reshape(-1, 1),
            dtype=torch.float32,
            device=device
        )

        y_torch = torch.tensor(
            (y_line / y_ref).reshape(-1, 1),
            dtype=torch.float32,
            device=device
        )

        t_torch = torch.tensor(
            np.full((n_profile_points, 1), used_t / t_ref),
            dtype=torch.float32,
            device=device
        )

        model.eval()
        with torch.no_grad():
            ux_pinn = (
                model(y_torch, x_torch, t_torch)[:, 1:2] * u_ref
            ).cpu().numpy().flatten()

        ax1.plot(
            ux_pinn,
            y_line,
            color=color,
            linewidth=2.2,
            linestyle="-",
            zorder=2
        )

        ax1.plot(
            ux_comsol[idx_markers_velocity],
            y_line[idx_markers_velocity],
            linestyle="None",
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            zorder=3
        )

        time_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.5,
                label=rf"t = {used_t:.2f} s"
            )
        )

    ax1.set_title("Velocity profile at x = 6 m", fontsize=15)
    ax1.set_xlabel(r"$u_x$ (m/s)")
    ax1.set_ylabel("y (m)")
    ax1.set_ylim(0.0, H)
    ax1.grid(alpha=0.3)

    for i, (idx_t, used_t) in enumerate(zip(time_indices, times_used)):

        color = colors[i % len(colors)]

        p_comsol = interp_comsol(
            values_at_nodes=p_wide[:, idx_t],
            x_query=x_line,
            y_query=y_line_for_pressure
        )

        x_torch = torch.tensor(
            (x_line / x_ref).reshape(-1, 1),
            dtype=torch.float32,
            device=device
        )

        y_torch = torch.tensor(
            (y_line_for_pressure / y_ref).reshape(-1, 1),
            dtype=torch.float32,
            device=device
        )

        t_torch = torch.tensor(
            np.full((n_pressure_points, 1), used_t / t_ref),
            dtype=torch.float32,
            device=device
        )

        model.eval()
        with torch.no_grad():
            p_pinn = (
                model(y_torch, x_torch, t_torch)[:, 2:3] * p_ref
            ).cpu().numpy().flatten()

        ax2.plot(
            x_line,
            p_pinn,
            color=color,
            linewidth=2.2,
            linestyle="-",
            zorder=2
        )

        ax2.plot(
            x_line[idx_markers_pressure],
            p_comsol[idx_markers_pressure],
            linestyle="None",
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            zorder=3
        )

    ax2.set_title("Pressure distribution at y = 2 m", fontsize=15)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("p (Pa)")
    ax2.set_xlim(0.0, L)
    ax2.grid(alpha=0.3)

    style_handles = [
        Line2D(
            [0], [0],
            color="k",
            lw=2.5,
            linestyle="-",
            label="PINN"
        ),
        Line2D(
            [0], [0],
            color="k",
            linestyle="None",
            marker="o",
            markersize=5,
            markerfacecolor="k",
            markeredgecolor="k",
            label="COMSOL"
        )
    ]

    fig.legend(
        handles=time_handles + style_handles,
        loc="lower center",
        ncol=len(time_handles) + len(style_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save_fig(fig, "stacked_velocity_pressure_profiles_selected_times")
    show_or_close(fig)


def plot_per_time_slice_error():
    rel_l2_vmag_time = []
    rel_l2_p_time = []

    for j, _ in enumerate(t_values):

        ev = relative_l2_numpy(
            pred=v_mag_pred_wide[:, j],
            ref=v_mag_data_wide[:, j]
        )

        ep = relative_l2_numpy(
            pred=p_pred_wide[:, j],
            ref=p_wide[:, j]
        )

        rel_l2_vmag_time.append(ev)
        rel_l2_p_time.append(ep)

    rel_l2_vmag_time = np.asarray(rel_l2_vmag_time)
    rel_l2_p_time = np.asarray(rel_l2_p_time)

    print("\n=== Per-time-slice relative L2 error summary ===")
    print(
        f"|v|: min = {rel_l2_vmag_time.min():.4f}%, "
        f"max = {rel_l2_vmag_time.max():.4f}%, "
        f"mean = {rel_l2_vmag_time.mean():.4f}%"
    )
    print(
        f"p  : min = {rel_l2_p_time.min():.4f}%, "
        f"max = {rel_l2_p_time.max():.4f}%, "
        f"mean = {rel_l2_p_time.mean():.4f}%"
    )
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10.5, 6))

    ax.plot(
        t_values,
        rel_l2_vmag_time,
        color="tab:blue",
        linewidth=2.0,
        label="|v|"
    )

    ax.plot(
        t_values,
        rel_l2_p_time,
        color="tab:red",
        linewidth=2.0,
        label="p"
    )

    for i, boundary_time in enumerate(TIME_WINDOWS_PHYS[:-1]):

        ax.axvline(
            boundary_time,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="Time-window boundary" if i == 0 else None
        )

    ax.grid(alpha=0.3)

    ax.set_title(r"Time evolution of relative $L_2$ error", fontsize=15)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"Relative $L_2$ error (%)")
    ax.set_xlim(0.0, t_final)
    ax.set_xticks(np.arange(0.0, t_final + 0.001, 0.5))

    ax.legend(
        loc="best",
        frameon=True,
        fontsize=11
    )

    plt.tight_layout()
    save_fig(fig, "per_time_slice_relative_L2_error_vmag_pressure")
    show_or_close(fig)

# =========================
# Safer relative errors (%)
# =========================

rel_err_u_y_percent = safe_percent_error(
    abs_error=abs_err_u_y,
    true_values=u_y_data_np,
    fallback_scale=v_mag_data_np,
    floor_fraction=1e-3,
    clip_max=100.0
)

rel_err_u_x_percent = safe_percent_error(
    abs_error=abs_err_u_x,
    true_values=u_x_data_np,
    fallback_scale=v_mag_data_np,
    floor_fraction=1e-3,
    clip_max=100.0
)

rel_err_v_mag_percent = safe_percent_error(
    abs_error=abs_err_v_mag,
    true_values=v_mag_data_np,
    fallback_scale=v_mag_data_np,
    floor_fraction=1e-3,
    clip_max=100.0
)

rel_err_p_percent = safe_percent_error(
    abs_error=abs_err_p,
    true_values=p_data_np,
    fallback_scale=p_data_np,
    floor_fraction=1e-3,
    clip_max=100.0
)


# =========================
# Plots
# =========================

if CREATE_PLOTS:

    # =========================
    # Field comparison plots
    # =========================

    plot_field_comparison(
        field_name="u_y",
        unit="m/s",
        comsol_values=u_y_data_np[sl],
        pinn_values=u_y_pred_np[sl],
        absolute_error=abs_err_u_y[sl],
        relative_error_percent=rel_err_u_y_percent[sl]
    )

    plot_field_comparison(
        field_name="u_x",
        unit="m/s",
        comsol_values=u_x_data_np[sl],
        pinn_values=u_x_pred_np[sl],
        absolute_error=abs_err_u_x[sl],
        relative_error_percent=rel_err_u_x_percent[sl]
    )

    plot_field_comparison(
        field_name="|v|",
        unit="m/s",
        comsol_values=v_mag_data_np[sl],
        pinn_values=v_mag_pred_np[sl],
        absolute_error=abs_err_v_mag[sl],
        relative_error_percent=rel_err_v_mag_percent[sl]
    )

    plot_field_comparison(
        field_name="p",
        unit="Pa",
        comsol_values=p_data_np[sl],
        pinn_values=p_pred_np[sl],
        absolute_error=abs_err_p[sl],
        relative_error_percent=rel_err_p_percent[sl]
    )

    # =========================
    # Time-series and phase plots
    # =========================

    plot_pressure_time_series_at_point(
        xp=5.0,
        yp=2.0
    )

    plot_phase_comparison_at_point(
        xp=5.0,
        yp=2.0
    )

    # =========================
    # Total loss evolution plot: Adam + L-BFGS
    # =========================

    if len(loss_history) > 0 and len(epoch_history) > 0:

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            epoch_history,
            loss_history,
            label="Adam loss",
            linewidth=2.0
        )

        window_change_epochs = np.cumsum(EPOCHS_PER_WINDOW)[:-1]

        for i, change_epoch in enumerate(window_change_epochs):
            ax.axvline(
                change_epoch,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label="Time-window boundary" if i == 0 else None
            )

        if len(loss_history_lbfgs) > 0:

            last_adam_epoch = epoch_history[-1]

            lbfgs_plot_epochs = np.concatenate([
                np.array([last_adam_epoch]),
                last_adam_epoch + np.arange(1, len(loss_history_lbfgs) + 1)
            ])

            lbfgs_plot_loss = np.concatenate([
                np.array([loss_history[-1]]),
                np.array(loss_history_lbfgs)
            ])

            ax.plot(
                lbfgs_plot_epochs,
                lbfgs_plot_loss,
                label="L-BFGS loss",
                color="red",
                linewidth=2.0
            )

            ax.axvline(
                last_adam_epoch,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label="Adam to L-BFGS transition"
            )

        ax.set_xlabel("Training step")
        ax.set_ylabel("Total loss")
        ax.set_title("Total loss evolution: Adam + L-BFGS")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()
        save_fig(fig, "total_loss_evolution_adam_lbfgs")
        show_or_close(fig)

    # =========================
    # Individual loss terms evolution plot
    # =========================

    if len(epoch_history) > 0 and len(loss_term_history["Cont"]) > 0:

        fig, ax = plt.subplots(figsize=(10, 6))

        for key, values in loss_term_history.items():
            ax.plot(
                epoch_history,
                values,
                label=key,
                linewidth=1.4,
                alpha=0.9
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss term")
        ax.set_title("Individual loss terms evolution during Adam training")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9, ncol=2)

        plt.tight_layout()
        save_fig(fig, "individual_loss_terms_adam")
        show_or_close(fig)

    # =========================
    # Adam learning rate evolution plot
    # =========================

    adam_epoch_history = epoch_history[1:]

    if len(lr_history) == len(adam_epoch_history) and len(lr_history) > 0:

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            adam_epoch_history,
            lr_history,
            label="Adam learning rate",
            linewidth=2.0
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Adam learning rate evolution")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()
        save_fig(fig, "adam_learning_rate_evolution")
        show_or_close(fig)

    else:

        raise RuntimeError(
            "Cannot create Adam learning rate plot because "
            f"Adam epoch history length ({len(adam_epoch_history)}) differs from "
            f"lr_history length ({len(lr_history)})."
        )

    # =========================
    # Final three-panel comparison plots
    # =========================

    plot_three_panel_vertical(
        save_name=f"vertical_three_panel_vmag_t{num_to_tag(t_slice, decimals=3)}",
        field_symbol=r"|v|",
        unit="m/s",
        comsol_values=v_mag_data_np[sl],
        pinn_values=v_mag_pred_np[sl],
        absolute_error=abs_err_v_mag[sl],
        cmap="viridis",
        error_exponent=-3
    )

    plot_three_panel_vertical(
        save_name=f"vertical_three_panel_pressure_t{num_to_tag(t_slice, decimals=3)}",
        field_symbol=r"$p$",
        unit="Pa",
        comsol_values=p_data_np[sl],
        pinn_values=p_pred_np[sl],
        absolute_error=abs_err_p[sl],
        cmap="viridis"
    )

    # =========================
    # Velocity and pressure profile plots
    # =========================

    plot_velocity_pressure_profiles_stacked(
        times_to_plot=(2.50, 2.60, 2.70, 2.85),
        x_profile_location=6.0,
        y_pressure_location=2.0
    )

    # =========================
    # Per-time-slice error plot
    # =========================

    plot_per_time_slice_error()
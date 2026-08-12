"""
Stage 1: Stationary forward PINN for incompressible Navier-Stokes flow.

The model is trained using physics-informed losses only. COMSOL data are used
only for evaluation and plotting, not during training.

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
from matplotlib.ticker import FormatStrFormatter

# =========================
# Configuration
# =========================

# Run mode
TRAIN_MODE = True
SAVE_CHECKPOINT = False
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

COMSOL_DATA_PATH = REPO_ROOT / "data" / "comsol" / "stage1_stage2" / "NS_xy_stationary.txt"
CHECKPOINT_PATH = REPO_ROOT / "models" / "stage1_forward_model.pt"
PLOT_DATA_PATH = REPO_ROOT / "stage1_stationary_forward" / "stage1_plot_data.npz"
FIGURES_DIR = REPO_ROOT / "stage1_stationary_forward" / "figures"

# Physical parameters
MU = 1.0
RHO = 1.0
P_IN_DIM = 10.0
P_OUT_DIM = 0.0
LENGTH = 12.0
HEIGHT = 4.0

# Network architecture
INPUT_DIM = 2
OUTPUT_DIM = 3
HIDDEN_LAYERS = 5
NEURONS = 64

# Training points
N_DOMAIN = 2000
N_WALL_UPPER = 500
N_WALL_LOWER = 500
N_INLET = 200
N_OUTLET = 200

# Optimizer settings
ADAM_EPOCHS = 5000
LR_ADAM = 1e-3

MAX_LBFGS_ITERATIONS = 300
LR_LBFGS = 1e-1

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
p_in_dim = P_IN_DIM
p_out_dim = P_OUT_DIM

deltaP = p_in_dim - p_out_dim

L = LENGTH
H = HEIGHT

# Characteristic values for non-dimensionalization
x_ref = L
y_ref = H
u_ref = deltaP * H**2 / (8 * mu * L)
alpha = y_ref / x_ref
p_ref = rho * u_ref**2
Re = rho * u_ref * y_ref / mu

print("Problem is solved in non-dimensional form:")
print(f"X_ref = {x_ref:.4e} m")
print(f"Y_ref = {y_ref:.4e} m")
print(f"P_ref = {p_ref:.4e} Pa")
print(f"U_ref = {u_ref:.4e} m/s")
print(f"Aspect ratio Y_ref/X_ref: alpha = {alpha:.4e}")
print(f"Reynolds Number = {Re:.4f}")

# =========================
# Model: (y*, x*) -> (u_y*, u_x*, p*)
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

    def forward(self, y, x):
        out = torch.cat([y, x], dim=1)

        for i in range(len(self.layers) - 1):
            out = self.activation(self.layers[i](out))

        return self.layers[-1](out)

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

print(f"Adam optimizer with fixed learning rate: {LR_ADAM:.1e}")

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
if not COMSOL_DATA_PATH.exists():
    raise FileNotFoundError(
        f"COMSOL file not found:\n{COMSOL_DATA_PATH}"
    )

data = np.loadtxt(COMSOL_DATA_PATH, comments="%")
print(f"Loaded COMSOL data from: {COMSOL_DATA_PATH}")

# Dimensional data
y_dim   = data[:, 1:2]      # m
x_dim   = data[:, 0:1]      # m
u_y_dim = data[:, 2:3]      # m/s
u_x_dim = data[:, 3:4]      # m/s
p_dim   = data[:, 4:5]      # Pa

# Conversion to torch tensors
y_data      = torch.tensor(y_dim / y_ref, dtype=torch.float32, device=device)
x_data      = torch.tensor(x_dim / x_ref, dtype=torch.float32, device=device)
u_y_data    = torch.tensor(u_y_dim, dtype=torch.float32, device=device)
u_x_data    = torch.tensor(u_x_dim, dtype=torch.float32, device=device)
p_data      = torch.tensor(p_dim, dtype=torch.float32, device=device)

# =========================
# Collocation points
# =========================

N_d = N_DOMAIN

y_d_np = np.random.uniform(0.0, H / y_ref, (N_d, 1))
x_d_np = np.random.uniform(0.0, L / x_ref, (N_d, 1))

y_d = torch.tensor(
    y_d_np,
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

x_d = torch.tensor(
    x_d_np,
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

print("Random uniform collocation point selection")
print(f"Random collocation points used: {N_d}")

# =========================
# Navier-Stokes residual calculator
# =========================
def NS_res_calc(
    y, x,
    u_y, u_x,
    u_y_y, u_y_yy, u_y_x, u_y_xx,
    u_x_x, u_x_xx, u_x_y, u_x_yy,
    p_y, p_x
):
    residual_c = alpha * u_x_x + u_y_y

    NS_terms_y = {
        "conv_y": u_x * alpha * u_y_x + u_y * u_y_y,
        "press_y": p_ref / (rho * u_ref**2) * p_y,
        "visc_y": (-1 / Re) * (alpha**2 * u_y_xx + u_y_yy),
    }

    NS_terms_x = {
        "conv_x": u_x * alpha * u_x_x + u_y * u_x_y,
        "press_x": alpha * p_ref / (rho * u_ref**2) * p_x,
        "visc_x": (-1 / Re) * (alpha**2 * u_x_xx + u_x_yy),
    }

    residual_y = sum(NS_terms_y.values())
    residual_x = sum(NS_terms_x.values())

    return residual_c, residual_y, residual_x, NS_terms_y, NS_terms_x

# =========================
# Upper wall BC at y*=1
# =========================
N_wall_upper = N_WALL_UPPER

y_wall_upper_np = np.ones((N_wall_upper, 1)) * (H / y_ref)
x_wall_upper_np = np.random.uniform(0.0, L / x_ref, (N_wall_upper, 1))

y_wall_upper = torch.tensor(y_wall_upper_np, dtype=torch.float32, device=device)
x_wall_upper = torch.tensor(x_wall_upper_np, dtype=torch.float32, device=device)

# =========================
# Lower wall BC at y*=0
# =========================
N_wall_lower = N_WALL_LOWER

y_wall_lower_np = np.zeros((N_wall_lower, 1))
x_wall_lower_np = np.random.uniform(0.0, L / x_ref, (N_wall_lower, 1))

y_wall_lower = torch.tensor(y_wall_lower_np, dtype=torch.float32, device=device)
x_wall_lower = torch.tensor(x_wall_lower_np, dtype=torch.float32, device=device)

# =========================
# Inlet at x*=0
# =========================
N_in = N_INLET

y_in_np = np.random.uniform(0.0, H / y_ref, (N_in,1))
x_in_np = np.zeros((N_in,1))

y_in = torch.tensor(y_in_np, dtype=torch.float32, device=device)
x_in = torch.tensor(x_in_np, dtype=torch.float32, device=device)

p_in = torch.full((N_in,1), p_in_dim / p_ref, dtype=torch.float32, device=device)

# =========================
# Outlet at x*=L/x_ref
# =========================
N_out = N_OUTLET

y_out_np = np.random.uniform(0.0, H / y_ref, (N_out,1))
x_out_np = np.ones((N_out,1)) * (L / x_ref)

y_out = torch.tensor(y_out_np, dtype=torch.float32, device=device)
x_out = torch.tensor(x_out_np, dtype=torch.float32, device=device)

p_out = torch.full((N_out,1), p_out_dim / p_ref, dtype=torch.float32, device=device)

# =========================
# Loss balancing weights
# =========================

weights = {
    "Cont"      : 1.0,
    "NS_y"      : 1.0,
    "NS_x"      : 1.0,
    "Upper wall": 1.0,
    "Lower wall": 1.0,
    "Inlet"     : 1.0,
    "Outlet"    : 1.0,
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

    # Model Output on Collocation Points:
    output = model(y_d, x_d)   # (u_y*, u_x*, p*)

    u_y_pred = output[:,0:1]   # u_y*
    u_x_pred = output[:,1:2]   # u_x*
    p_pred = output[:,2:3]     # p*

    # Gradient Calculation (dimensionless):
    u_y_y = grad_calc(u_y_pred, y_d)
    u_y_yy = grad_calc(u_y_y, y_d)
    u_y_x = grad_calc(u_y_pred, x_d)
    u_y_xx = grad_calc(u_y_x, x_d)

    u_x_y = grad_calc(u_x_pred, y_d)
    u_x_yy = grad_calc(u_x_y, y_d)
    u_x_x = grad_calc(u_x_pred, x_d)
    u_x_xx = grad_calc(u_x_x, x_d)

    p_y = grad_calc(p_pred, y_d)
    p_x = grad_calc(p_pred, x_d)

    residual_c, residual_y, residual_x, NS_terms_y, NS_terms_x = NS_res_calc(
        y_d, x_d,
        u_y_pred, u_x_pred,
        u_y_y, u_y_yy, u_y_x, u_y_xx,
        u_x_x, u_x_xx, u_x_y, u_x_yy,
        p_y, p_x
        )

    loss_c = torch.mean(residual_c**2)
    loss_y = torch.mean(residual_y**2)
    loss_x = torch.mean(residual_x**2)

    if return_diag:
        diag_y = {
            "|Conv_y|"      : NS_terms_y["conv_y"].abs().mean().item(),
            "|Press_y|"     : NS_terms_y["press_y"].abs().mean().item(),
            "|Visc_y|"      : NS_terms_y["visc_y"].abs().mean().item(),
            "Res_y_mean"    : residual_y.mean().item(),
            "|Res_y_max|"   : residual_y.abs().max().item()
        }

        diag_x = {
            "|Conv_x|"      : NS_terms_x["conv_x"].abs().mean().item(),
            "|Press_x|"     : NS_terms_x["press_x"].abs().mean().item(),
            "|Visc_x|"      : NS_terms_x["visc_x"].abs().mean().item(),
            "Res_x_mean"    : residual_x.mean().item(),
            "|Res_x_max|"   : residual_x.abs().max().item()
        }
    else:
        diag_y = diag_x = None

    # ---- Upper wall BC loss calculation ----
    output_wall_upper = model(y_wall_upper, x_wall_upper)

    u_y_wall_pred_upper = output_wall_upper[:,0:1]
    u_x_wall_pred_upper = output_wall_upper[:,1:2]

    loss_wall_upper = torch.mean(u_y_wall_pred_upper**2 + u_x_wall_pred_upper**2)

    # ---- Lower wall BC loss calculation ----
    output_wall_lower = model(y_wall_lower, x_wall_lower)

    u_y_wall_pred_lower = output_wall_lower[:,0:1]
    u_x_wall_pred_lower = output_wall_lower[:,1:2]

    loss_wall_lower = torch.mean(u_y_wall_pred_lower**2 + u_x_wall_pred_lower**2)

    # ---- Inlet loss calculation ----
    output_in = model(y_in, x_in)

    p_in_pred = output_in[:,2:3]

    u_y_in_pred = output_in[:, 0:1]
    loss_in = torch.mean((p_in_pred - p_in)**2 + u_y_in_pred**2)

    # ---- Outlet loss calculation ----
    output_out = model(y_out, x_out)

    p_out_pred = output_out[:,2:3]

    u_y_out_pred = output_out[:, 0:1]
    loss_out = torch.mean((p_out_pred - p_out)**2 + u_y_out_pred**2)

    # ---- Total loss ----

    loss_terms = {
        "Cont"      : loss_c,
        "NS_y"      : loss_y,
        "NS_x"      : loss_x,
        "Upper wall": loss_wall_upper,
        "Lower wall": loss_wall_lower,
        "Inlet"     : loss_in,
        "Outlet"    : loss_out
    }

    loss = sum(weights[key] * loss_terms[key] for key in loss_terms)

    return loss, loss_terms, diag_y, diag_x

# =========================
# Output utilities
# =========================

def sanitize_filename(name):
    filename_aliases = {
        "|v|_comparison": "v_mag_comparison",
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
            f"Output file already exists and would be overwritten:\n"
            f"{path}\n\n"
            f"Delete the existing file or set OVERWRITE_EXISTING_OUTPUTS = True."
        )


def expected_figure_paths():
    figure_names = [
        "u_y_comparison",
        "u_x_comparison",
        "|v|_comparison",
        "p_comparison",
        "total_loss_evolution_adam_lbfgs",
        "individual_loss_terms_adam",
        "vertical_three_panel_vmag_stationary",
        "vertical_three_panel_pressure_stationary",
    ]

    return [FIGURES_DIR / f"{sanitize_filename(name)}.png" for name in figure_names]


def check_existing_outputs_before_run():
    if TRAIN_MODE and SAVE_CHECKPOINT:
        stop_if_output_exists(CHECKPOINT_PATH)

    if SAVE_PLOT_DATA:
        stop_if_output_exists(PLOT_DATA_PATH)

    if CREATE_PLOTS:
        for path in expected_figure_paths():
            stop_if_output_exists(path)


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

# =========================
# Checkpoint save / load
# =========================

def save_stage1_checkpoint():
    checkpoint = {
        "model_state_dict": model.state_dict(),

        "layers": layers,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_layers": HIDDEN_LAYERS,
        "neurons": NEURONS,

        "seed": seed,
        "seed_mode": "random" if RANDOM_SEED else "fixed",
        "weight_initialization": "xavier_normal",
        "formulation": "nondimensional",
        "collocation_sampling": "random_uniform",
        "optimizer_strategy": "Adam + L-BFGS",

        "environment": {
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
        },

        "physical_parameters": {
            "mu": mu,
            "rho": rho,
            "p_in_dim": p_in_dim,
            "p_out_dim": p_out_dim,
            "deltaP": deltaP,
            "L": L,
            "H": H
        },

        "scales": {
            "x_ref": x_ref,
            "y_ref": y_ref,
            "u_ref": u_ref,
            "p_ref": p_ref,
            "alpha": alpha,
            "Re": Re
        },

        "training_settings": {
            "adam_epochs": ADAM_EPOCHS,
            "learning_rate_adam": LR_ADAM,
            "learning_rate_lbfgs": LR_LBFGS,
            "max_lbfgs_iterations": MAX_LBFGS_ITERATIONS,
            "weights": weights,
        },

        "loss_history": loss_history,
        "epoch_history": epoch_history,
        "loss_history_lbfgs": loss_history_lbfgs,
        "lbfgs_iter_history": lbfgs_iter_history,
        "loss_term_history": loss_term_history,
    }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stop_if_output_exists(CHECKPOINT_PATH)

    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nCheckpoint saved to: {CHECKPOINT_PATH}")


def load_stage1_checkpoint():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint '{CHECKPOINT_PATH}' not found. "
            "Run first with TRAIN_MODE = True."
        )

    checkpoint = load_torch_checkpoint(CHECKPOINT_PATH)

    if checkpoint.get("layers") != layers:
        print("\nWARNING: Loaded checkpoint architecture differs from current layers.")
        print(f"Checkpoint layers: {checkpoint.get('layers')}")
        print(f"Current layers   : {layers}")

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

    for epoch in range(1, ADAM_EPOCHS + 1):

        optimizer_adam.zero_grad()

        do_print = (epoch % 500 == 0)
        loss, loss_terms, diag_y, diag_x = loss_function(return_diag=do_print)
        loss.backward()

        optimizer_adam.step()

        loss_history.append(loss.item())
        epoch_history.append(epoch)

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

        if epoch % 500 == 0:

            print(f"Epoch {epoch}, Total Loss: {loss.item():.4e}")

            print("\n--- Loss terms ---")
            for key in weights:
                print(f"{key:12s}: {loss_terms[key].item():.4e}  ({loss_term_percentages[key]:.2f}%)")

            print("\n--- NS_y diagnostics ---")
            for key, value in diag_y.items():
                print(f"{key:12s}: {value:+.4e}")

            print("\n--- NS_x diagnostics ---")
            for key, value in diag_x.items():
                print(f"{key:12s}: {value:+.4e}")

            print("-" * 60)

    adam_end_time = sync_time()
    adam_time = adam_end_time - adam_start_time

    print("\n" + "=" * 60)
    print(f"Adam training time: {adam_time:.2f} s ({adam_time/60:.2f} min)")
    print("=" * 60)

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
        save_stage1_checkpoint()


else:

    print("\n" + "=" * 60)
    print("Loading trained model and skipping training...")
    print("=" * 60)

    checkpoint = load_stage1_checkpoint()

    # Restore histories for loss plots
    loss_history = checkpoint.get("loss_history", [])
    epoch_history = checkpoint.get("epoch_history", [])
    loss_history_lbfgs = checkpoint.get("loss_history_lbfgs", [])
    lbfgs_iter_history = checkpoint.get("lbfgs_iter_history", [])
    loss_term_history = checkpoint.get("loss_term_history", loss_term_history)

    adam_time = 0.0
    lbfgs_time = 0.0
    optimizer_time = 0.0
    overhead_time = 0.0
    total_time = 0.0


# =========================
# Evaluation
# =========================
model.eval()

with torch.no_grad():
    output_data = model(y_data, x_data)

    u_y_pred = output_data[:, 0:1] * u_ref
    u_x_pred = output_data[:, 1:2] * u_ref
    p_pred   = output_data[:, 2:3] * p_ref

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
# Boundary checks
# =========================
with torch.no_grad():
    # Upper Wall predictions
    output_wall_upper = model(y_wall_upper, x_wall_upper)

    u_y_wall_pred_upper = output_wall_upper[:, 0:1] * u_ref
    u_x_wall_pred_upper = output_wall_upper[:, 1:2] * u_ref

    # Lower Wall predictions
    output_wall_lower = model(y_wall_lower, x_wall_lower)

    u_y_wall_pred_lower = output_wall_lower[:, 0:1] * u_ref
    u_x_wall_pred_lower = output_wall_lower[:, 1:2] * u_ref

    # Pressure inlet/outlet
    output_in = model(y_in, x_in)
    output_out = model(y_out, x_out)

    p_in_pred = output_in[:, 2:3] * p_ref
    p_out_pred = output_out[:, 2:3] * p_ref

    # Velocity inlet/outlet
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
print(f"Max |p_inlet| (Pa)          :   {torch.max(torch.abs(p_in_pred)).item():.6e}")
print(f"Mean p_inlet (Pa)           :   {torch.mean(p_in_pred).item():.6e}")
print(f"Max |p_outlet| (Pa)         :   {torch.max(torch.abs(p_out_pred)).item():.6e}")
print(f"Mean p_outlet (Pa)          :   {torch.mean(p_out_pred).item():.6e}")
print(f"Target p_inlet (Pa)         :   {p_in_dim:.6e}")
print(f"Target p_outlet (Pa)        :   {p_out_dim:.6e}")
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

u_y_pred_np = u_y_pred.detach().cpu().numpy().flatten()
u_x_pred_np = u_x_pred.detach().cpu().numpy().flatten()
p_pred_np   = p_pred.detach().cpu().numpy().flatten()

u_y_data_np = u_y_data.detach().cpu().numpy().flatten()
u_x_data_np = u_x_data.detach().cpu().numpy().flatten()
p_data_np   = p_data.detach().cpu().numpy().flatten()

v_mag_pred_np = v_mag_pred.detach().cpu().numpy().flatten()
v_mag_data_np = v_mag_data.detach().cpu().numpy().flatten()

# Absolute errors
abs_err_u_y = np.abs(u_y_data_np - u_y_pred_np)
abs_err_u_x = np.abs(u_x_data_np - u_x_pred_np)
abs_err_p   = np.abs(p_data_np - p_pred_np)
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

        u_y_pred_np=u_y_pred_np,
        u_x_pred_np=u_x_pred_np,
        p_pred_np=p_pred_np,

        u_y_data_np=u_y_data_np,
        u_x_data_np=u_x_data_np,
        p_data_np=p_data_np,

        v_mag_pred_np=v_mag_pred_np,
        v_mag_data_np=v_mag_data_np,

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
        relative_l2_p=rel_l2_p_dim.item()
    )

    print(f"\nPlot data saved to: {PLOT_DATA_PATH}")

# =========================
# Plotting helpers
# =========================

triang = tri.Triangulation(x_plot, y_plot)

plot_dir = FIGURES_DIR

if CREATE_PLOTS:
    plot_dir.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name, dpi=300):
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
    """
    Returns a scaling factor in engineering notation.
    """
    vmax = np.nanmax(np.abs(values))

    if (not np.isfinite(vmax)) or vmax <= 0:
        return 1.0, 0

    exponent = int(np.floor(np.log10(vmax)))
    exponent_eng = 3 * int(np.floor(exponent / 3))
    scale = 10.0 ** exponent_eng

    return scale, exponent_eng


def format_scaled_colorbar_label(base_symbol, exponent, unit):
    """
    Example:
    exponent = -3, unit = 'm/s'
    -> |Δ| (×10^{-3} m/s)
    """
    if exponent == 0:
        return rf"{base_symbol} ({unit})"
    else:
        return rf"{base_symbol} ($\times 10^{{{exponent}}}$ {unit})"


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
    """
    Creates a vertical 3-panel figure:
    1) COMSOL solution
    2) PINN solution
    3) Absolute deviation
    """

    field_vmin, field_vmax = get_common_limits(comsol_values, pinn_values)

    # Scale absolute error colorbar.
    # Example: error_exponent = -3 gives |Δ| (×10^{-3} m/s).
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
        triang,
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
        triang,
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
        triang,
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

def plot_field_comparison(
    field_name,
    unit,
    comsol_values,
    pinn_values,
    absolute_error,
    relative_error_percent
):
    """
    Creates one 2x2 figure:
    top-left: COMSOL
    top-right: PINN
    bottom-left: absolute error
    bottom-right: relative % error
    """

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

    fig.suptitle(f"{field_name} comparison", fontsize=14)

    cf1 = axs[0, 0].tricontourf(
        triang,
        comsol_values,
        levels=60,
        vmin=field_vmin,
        vmax=field_vmax
    )
    fig.colorbar(cf1, ax=axs[0, 0], label=f"COMSOL {field_name} ({unit})")
    axs[0, 0].set_title(f"COMSOL {field_name}")
    setup_axis(axs[0, 0])

    cf2 = axs[0, 1].tricontourf(
        triang,
        pinn_values,
        levels=60,
        vmin=field_vmin,
        vmax=field_vmax
    )
    fig.colorbar(cf2, ax=axs[0, 1], label=f"PINN {field_name} ({unit})")
    axs[0, 1].set_title(f"PINN {field_name}")
    setup_axis(axs[0, 1])

    cf3 = axs[1, 0].tricontourf(
        triang,
        absolute_error,
        levels=60,
        vmin=abs_vmin,
        vmax=abs_vmax
    )
    fig.colorbar(cf3, ax=axs[1, 0], label=f"|COMSOL - PINN| ({unit})")
    axs[1, 0].set_title(f"Absolute error in {field_name}")
    setup_axis(axs[1, 0])

    cf4 = axs[1, 1].tricontourf(
        triang,
        relative_error_percent,
        levels=60,
        vmin=rel_vmin,
        vmax=rel_vmax
    )
    fig.colorbar(cf4, ax=axs[1, 1], label="Relative error (%)")
    axs[1, 1].set_title(f"Relative error in {field_name} (%)")
    setup_axis(axs[1, 1])

    save_fig(fig, f"{field_name}_comparison")
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
        comsol_values=u_y_data_np,
        pinn_values=u_y_pred_np,
        absolute_error=abs_err_u_y,
        relative_error_percent=rel_err_u_y_percent
    )

    plot_field_comparison(
        field_name="u_x",
        unit="m/s",
        comsol_values=u_x_data_np,
        pinn_values=u_x_pred_np,
        absolute_error=abs_err_u_x,
        relative_error_percent=rel_err_u_x_percent
    )

    plot_field_comparison(
        field_name="|v|",
        unit="m/s",
        comsol_values=v_mag_data_np,
        pinn_values=v_mag_pred_np,
        absolute_error=abs_err_v_mag,
        relative_error_percent=rel_err_v_mag_percent
    )

    plot_field_comparison(
        field_name="p",
        unit="Pa",
        comsol_values=p_data_np,
        pinn_values=p_pred_np,
        absolute_error=abs_err_p,
        relative_error_percent=rel_err_p_percent
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
                linewidth=2.0
            )

            ax.axvline(
                last_adam_epoch,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label="Adam to L-BFGS transition"
            )

        ax.set_xlabel("Training step")
        ax.set_ylabel("Total loss")
        ax.set_title("Total loss evolution")
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
    # Final three-panel comparison plots
    # =========================

    plot_three_panel_vertical(
        save_name="vertical_three_panel_vmag_stationary",
        field_symbol=r"|v|",
        unit="m/s",
        comsol_values=v_mag_data_np,
        pinn_values=v_mag_pred_np,
        absolute_error=abs_err_v_mag,
        cmap="viridis",
        error_exponent=-3
    )

    plot_three_panel_vertical(
        save_name="vertical_three_panel_pressure_stationary",
        field_symbol=r"$p$",
        unit="Pa",
        comsol_values=p_data_np,
        pinn_values=p_pred_np,
        absolute_error=abs_err_p,
        cmap="viridis",
        error_exponent=-2
    )
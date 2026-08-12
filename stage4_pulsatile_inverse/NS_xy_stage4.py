"""
Stage 4: Pulsatile inverse PINN for incompressible Navier-Stokes flow.

The model solves the transient inverse problem using sparse pressure
observations from the pulsatile-flow reference solution.

The inverse problem estimates the inlet-pressure parameters according to the
selected experiment family:
- F1: A1 unknown, T known,
- F2: A1 known, T unknown,
- F3: A1 unknown, T unknown.

The implementation supports runs with or without transfer learning from the
trained Stage 3 forward model. The run name, model filename, and figures
subfolder are generated automatically from the selected experiment settings.

COMSOL data are used for evaluation, plotting, and sparse pressure observations.
They are not used as full-field supervised training data.

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
from scipy.interpolate import griddata


# =========================
# Configuration
# =========================

# Run mode
TRAIN_MODE = True
SAVE_CHECKPOINT = True

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
MODEL_ROOT = REPO_ROOT / "models"
FIGURES_ROOT = REPO_ROOT / "stage4_pulsatile_inverse" / "figures"

# Pretrained Stage 3 model/checkpoint.
# This checkpoint is used only for neural-network layers.
STAGE3_CHECKPOINT_PATH = MODEL_ROOT / "stage3_forward_model.pt"

# Optional checkpoint path used when TRAIN_MODE = False.
# None uses the path generated from the current configuration.
LOAD_MODEL_PATH = None

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

# Inverse problem setup
# "F1": A1 unknown, T known
# "F2": A1 known,   T unknown
# "F3": A1 unknown, T unknown
EXPERIMENT_FAMILY = "F2"

# Transfer learning from Stage 3
USE_TRANSFER_LEARNING = True

# Amount of pressure-observation time points:
# "all" : 61 COMSOL time snapshots
# "half": 30 COMSOL time snapshots
# "five": 5 COMSOL time snapshots
OBSERVATION_CASE = "all"

# Available pressure observation points
OBSERVATION_POINT_CATALOG = {
    "x2_y2": (2.0, 2.0),
    "x3_y2": (3.0, 2.0),
    "x4_y2": (4.0, 2.0),
    "x5_y2": (5.0, 2.0),
}

# Selected pressure observation points
SELECTED_OBSERVATION_POINTS = ["x2_y2"]

# Five selected pressure observation times
BASE_T_OBS_UNIQUE_PHYS = np.array(
    [0.10, 0.25, 0.40, 1.25, 2.40]
).reshape(-1, 1)

# Inference initial values
A0_INITIAL = 8.0
A1_INITIAL = 3.0
P_PERIOD_INITIAL = 0.8

# Training profile:
# "single_fast"       : single window, 300 Adam epochs, 500 L-BFGS iterations
# "curriculum_medium" : [700, 300, 300, 500] Adam epochs, 500 L-BFGS iterations
# "curriculum_hard"   : [1000, 500, 500, 500] Adam epochs, 1000 L-BFGS iterations
# "custom"            : uses the CUSTOM_* settings below
TRAINING_PROFILE = "custom"

# Custom training profile
CUSTOM_TIME_WINDOWS_PHYS = [0.5, 1.0, 2.0, 3.0]
CUSTOM_EPOCHS_PER_WINDOW = [700, 300, 300, 300]
CUSTOM_MAX_LBFGS_ITERATIONS = 500

# Optimizer settings
LR_ADAM = 1e-3
LR_LBFGS = 1e-1
RESET_LR_EACH_WINDOW = True

# Inverse-parameter learning rates by training stage
INVERSE_LR_BY_STAGE = [1e-2, 5e-3, 1e-3, 1e-3]

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

p_in_dim = P_IN_MEAN
p_out_dim = P_OUT_DIM
deltaP = p_in_dim - p_out_dim

A0_true = P_IN_MEAN
A1_true = P_IN_AMPLITUDE
T_true = P_PERIOD

A0_initial = A0_INITIAL
A1_initial = A1_INITIAL
T_initial = P_PERIOD_INITIAL

t_initial = T_INITIAL
t_final = T_FINAL
cycles = (t_final - t_initial) / T_true

dt_export = DT_EXPORT

L = LENGTH
H = HEIGHT

# Characteristic values for non-dimensionalization
x_ref = L
y_ref = H
t_ref = T_true
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
# Training profile settings
# =========================

if TRAINING_PROFILE == "single_fast":

    time_windows_phys = [3.0]
    epochs_per_window = [300]

    run_lbfgs = True
    max_lbfgs_iterations = 500


elif TRAINING_PROFILE == "curriculum_medium":

    time_windows_phys = [0.5, 1.0, 2.0, 3.0]
    epochs_per_window = [700, 300, 300, 500]

    run_lbfgs = True
    max_lbfgs_iterations = 500


elif TRAINING_PROFILE == "curriculum_hard":

    time_windows_phys = [0.5, 1.0, 2.0, 3.0]
    epochs_per_window = [1000, 500, 500, 500]

    run_lbfgs = True
    max_lbfgs_iterations = 1000


elif TRAINING_PROFILE == "custom":

    time_windows_phys = list(CUSTOM_TIME_WINDOWS_PHYS)
    epochs_per_window = list(CUSTOM_EPOCHS_PER_WINDOW)

    run_lbfgs = True
    max_lbfgs_iterations = CUSTOM_MAX_LBFGS_ITERATIONS


else:
    raise ValueError(
        "Wrong TRAINING_PROFILE. Use 'single_fast', "
        "'curriculum_medium', 'curriculum_hard' or 'custom'."
    )


if len(time_windows_phys) != len(epochs_per_window):
    raise ValueError(
        "time_windows_phys and epochs_per_window must have the same length."
    )

if any(np.diff(time_windows_phys) <= 0):
    raise ValueError("time_windows_phys must be strictly increasing.")

training_schedule = list(zip(time_windows_phys, epochs_per_window))

configured_total_adam_epochs = sum(
    stage_epochs for _, stage_epochs in training_schedule
)

# =========================
# Choices check for inverse problem
# =========================

if EXPERIMENT_FAMILY == "F1":
    infer_A1 = True
    infer_T = False
    inverse_case_name = "A1_unknown_T_known"

elif EXPERIMENT_FAMILY == "F2":
    infer_A1 = False
    infer_T = True
    inverse_case_name = "A1_known_T_unknown"

elif EXPERIMENT_FAMILY == "F3":
    infer_A1 = True
    infer_T = True
    inverse_case_name = "A1_unknown_T_unknown"

else:
    raise ValueError("Wrong EXPERIMENT_FAMILY. Use 'F1', 'F2' or 'F3'.")

if OBSERVATION_CASE not in ["all", "half", "five"]:
    raise ValueError("Wrong observation case input.")

if len(SELECTED_OBSERVATION_POINTS) == 0:
    raise ValueError("At least one observation point must be selected.")

for key in SELECTED_OBSERVATION_POINTS:
    if key not in OBSERVATION_POINT_CATALOG:
        raise ValueError(
            f"Unknown observation point key: {key}. "
            f"Available keys: {list(OBSERVATION_POINT_CATALOG.keys())}"
        )

observation_points_phys = np.array(
    [OBSERVATION_POINT_CATALOG[key] for key in SELECTED_OBSERVATION_POINTS],
    dtype=float
)

n_obs_points = observation_points_phys.shape[0]


# =========================
# Run name / paths
# =========================

def num_to_tag(value, decimals=4):
    """
    Converts numbers to filename-safe strings.
    Examples:
    0.4  -> 0p4
    0.05 -> 0p05
    3.0  -> 3
    """
    s = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def observation_time_count():
    if OBSERVATION_CASE == "all":
        return int(round((t_final - t_initial) / dt_export)) + 1

    if OBSERVATION_CASE == "half":
        n_total_times = int(round((t_final - t_initial) / dt_export)) + 1
        return len(np.arange(n_total_times)[1::2])

    if OBSERVATION_CASE == "five":
        return len(BASE_T_OBS_UNIQUE_PHYS)

    raise ValueError("Wrong OBSERVATION_CASE input.")


family_tag = EXPERIMENT_FAMILY
tl_tag = "TL" if USE_TRANSFER_LEARNING else "noTL"

adam_tag = "adam" + "_".join(str(int(e)) for e in epochs_per_window)

lbfgs_iterations = max_lbfgs_iterations if run_lbfgs else 0
lbfgs_tag = f"lbfgs{int(lbfgs_iterations)}"

spatial_points_tag = "_".join(SELECTED_OBSERVATION_POINTS)
spatial_tag = f"sp{n_obs_points}_{spatial_points_tag}"

temporal_tag = f"tm{observation_time_count()}"

initial_tags = [f"A0init{num_to_tag(A0_initial)}"]

if infer_A1:
    initial_tags.append(f"A1init{num_to_tag(A1_initial)}")

if infer_T:
    initial_tags.append(f"Tinit{num_to_tag(T_initial)}")

run_name_parts = [
    family_tag,
    tl_tag,
    adam_tag,
    lbfgs_tag,
    spatial_tag,
    temporal_tag,
] + initial_tags

run_name = "_".join(run_name_parts)

model_path = MODEL_ROOT / f"stage4_{run_name}_model.pt"
plot_dir = FIGURES_ROOT / run_name

if (not TRAIN_MODE) and (LOAD_MODEL_PATH is not None):
    model_path = Path(LOAD_MODEL_PATH)

    loaded_file_name = model_path.stem

    if loaded_file_name.startswith("stage4_"):
        run_name = loaded_file_name.replace("stage4_", "", 1)

        if run_name.endswith("_model"):
            run_name = run_name[:-len("_model")]

    elif loaded_file_name.startswith("NS_xy_stage4_"):
        run_name = loaded_file_name.replace("NS_xy_stage4_", "", 1)

    else:
        run_name = loaded_file_name

    plot_dir = FIGURES_ROOT / run_name

print("=" * 60)
print("Stage 4 setup")
print("=" * 60)
print(f"Run name                     : {run_name}")

if TRAIN_MODE:
    print(f"Training schedule            : {training_schedule}")
    print(f"Total Adam epochs            : {configured_total_adam_epochs}")
else:
    print("Mode                         : checkpoint evaluation")

print(f"Model path                   : {model_path}")
print(f"Figures directory            : {plot_dir}")
print("=" * 60)

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

        self.A0 = nn.Parameter(torch.tensor([A0_initial / p_ref], dtype=torch.float32))

        A1_start = A1_initial if infer_A1 else A1_true
        T_start  = T_initial  if infer_T  else T_true

        self.A1_log = nn.Parameter(
            torch.log(torch.tensor([A1_start / p_ref], dtype=torch.float32)),
            requires_grad=infer_A1
        )

        self.T_log = nn.Parameter(
            torch.log(torch.tensor([T_start / t_ref], dtype=torch.float32)),
            requires_grad=infer_T
        )

    def A1_star(self):
        return torch.exp(self.A1_log)

    def T_star(self):
        return torch.exp(self.T_log)

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

        p_in_t = self.A0 + self.A1_star() * torch.sin(
            2.0 * np.pi * t / self.T_star()
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

# =========================
# Transfer Learning utilities
# =========================

def load_network_layers_only(model, pretrained_model_path, device):
    """
    Loads only the neural-network weights and biases from the Stage 3 checkpoint.
    The inverse parameters A0, A1 and T retain their Stage 4 initial values.
    """

    pretrained_model_path = Path(pretrained_model_path)

    if not pretrained_model_path.exists():
        raise FileNotFoundError(
            f"Pretrained Stage 3 checkpoint not found:\n"
            f"{pretrained_model_path}"
        )

    try:
        checkpoint = torch.load(
            pretrained_model_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            pretrained_model_path,
            map_location=device
        )

    if not isinstance(checkpoint, dict) or "network_state" not in checkpoint:
        raise KeyError(
            "Stage 3 checkpoint does not contain 'network_state'. "
            "Generate it with the current NS_xy_stage3.py script."
        )

    if checkpoint.get("layers") != layers:
        raise ValueError(
            "The Stage 3 checkpoint architecture does not match Stage 4. "
            f"Checkpoint layers: {checkpoint.get('layers')}; "
            f"current layers: {layers}."
        )

    network_state = checkpoint["network_state"]
    current_state = model.state_dict()

    expected_layer_keys = {
        key for key in current_state
        if key.startswith("layers.")
    }

    if set(network_state.keys()) != expected_layer_keys:
        raise ValueError(
            "The Stage 3 network_state does not contain exactly the "
            "expected Stage 4 neural-network layer parameters."
        )

    current_state.update(network_state)
    model.load_state_dict(current_state, strict=True)

    print("\n" + "=" * 60)
    print("Transfer learning initialization")
    print("=" * 60)
    print(f"Loaded pretrained network from: {pretrained_model_path}")
    print("Transferred parameters: neural-network weights and biases only")
    print("A0, A1 and T retain their Stage 4 initial values.")
    print("=" * 60)

if TRAIN_MODE and USE_TRANSFER_LEARNING:
    load_network_layers_only(
        model=model,
        pretrained_model_path=STAGE3_CHECKPOINT_PATH,
        device=device
    )

print("Model on device:", next(model.parameters()).device)

# =========================
# Adam Optimizers
# =========================

inverse_param_names = ["A0", "A1_log", "T_log"]

network_params = [
    p for name, p in model.named_parameters()
    if name not in inverse_param_names
]

# -------------------------
# Inverse-parameter learning rates
# -------------------------

if len(INVERSE_LR_BY_STAGE) == 0:
    raise ValueError("INVERSE_LR_BY_STAGE must contain at least one learning rate.")

if any(lr <= 0.0 for lr in INVERSE_LR_BY_STAGE):
    raise ValueError("All values in INVERSE_LR_BY_STAGE must be positive.")


def get_inverse_lr_for_stage(stage_id):
    """
    Returns the common inverse-parameter learning rate for the current stage.

    If the number of stages exceeds the configured list length,
    the final learning rate is reused.
    """
    idx = min(stage_id - 1, len(INVERSE_LR_BY_STAGE) - 1)
    return INVERSE_LR_BY_STAGE[idx]

optimizer_adam = torch.optim.Adam(network_params, lr=LR_ADAM)

inverse_param_groups = []

if model.A0.requires_grad:
    inverse_param_groups.append(
        {
            "params": [model.A0],
            "lr": get_inverse_lr_for_stage(stage_id=1),
            "name": "A0",
        }
    )

if model.A1_log.requires_grad:
    inverse_param_groups.append(
        {
            "params": [model.A1_log],
            "lr": get_inverse_lr_for_stage(stage_id=1),
            "name": "A1_log",
        }
    )

if model.T_log.requires_grad:
    inverse_param_groups.append(
        {
            "params": [model.T_log],
            "lr": get_inverse_lr_for_stage(stage_id=1),
            "name": "T_log",
        }
    )

if len(inverse_param_groups) > 0:
    optimizer_inverse_params = torch.optim.Adam(inverse_param_groups)
else:
    optimizer_inverse_params = None


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
    max_iter=max_lbfgs_iterations,
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

    A0_history_lbfgs.append(model.A0.item() * p_ref)
    A1_history_lbfgs.append(model.A1_star().item() * p_ref)
    T_history_lbfgs.append(model.T_star().item() * t_ref)

    return loss

# =========================
# Load COMSOL reference data for sparse observations and evaluation
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
    raise ValueError(
        "Unexpected column count: expected 2 + 3*n_times."
    )

n_times = (n_cols - 2) // 3

# COMSOL export times: 0, 0.05, ..., 3.0 s
t_values = dt_export * np.arange(n_times)

print(f"Nodes: {n_nodes}, time snapshots: {n_times}, "
      f"t in [{t_values[0]:.3f}, {t_values[-1]:.3f}] s")

# Spatial coordinates (identical for every time)
x_dim = data[:, 0:1]     # (n_nodes, 1)  m
y_dim = data[:, 1:2]     # (n_nodes, 1)  m

# Field block -> (n_nodes, n_times, 3)
# field 0 = u_y, field 1 = u_x, field 2 = p
fields = data[:, 2:].reshape(n_nodes, n_times, 3)
u_y_wide = fields[:, :, 0]     # (n_nodes, n_times)  m/s
u_x_wide = fields[:, :, 1]     # (n_nodes, n_times)  m/s
p_wide   = fields[:, :, 2]     # (n_nodes, n_times)  Pa

# Reshape to long format: one row per (node, time)
x_long = np.repeat(x_dim, n_times, axis=1)          # (n_nodes, n_times)
y_long = np.repeat(y_dim, n_times, axis=1)
t_long = np.tile(t_values, (n_nodes, 1))            # (n_nodes, n_times)

# Flatten consistently (C-order: node-major, then time)
x_dim_long   = x_long.reshape(-1, 1)
y_dim_long   = y_long.reshape(-1, 1)
t_dim_long   = t_long.reshape(-1, 1)
u_y_dim_long = u_y_wide.reshape(-1, 1)
u_x_dim_long = u_x_wide.reshape(-1, 1)
p_dim_long   = p_wide.reshape(-1, 1)

print(f"Long-format dataset: {x_dim_long.shape[0]} rows "
      f"(= {n_nodes} nodes x {n_times} times)")

# Non-dimensionalize inputs; keep reference fields dimensional
# for sparse observation extraction and evaluation.
x_data   = torch.tensor(x_dim_long / x_ref, dtype=torch.float32, device=device)
y_data   = torch.tensor(y_dim_long / y_ref, dtype=torch.float32, device=device)
t_data   = torch.tensor(t_dim_long / t_ref, dtype=torch.float32, device=device)

u_y_data = torch.tensor(u_y_dim_long, dtype=torch.float32, device=device)   # m/s
u_x_data = torch.tensor(u_x_dim_long, dtype=torch.float32, device=device)   # m/s
p_data   = torch.tensor(p_dim_long,   dtype=torch.float32, device=device)   # Pa

# =========================
# Training point builders
# =========================

# -------------------------
# Base pressure observation times
# -------------------------

if OBSERVATION_CASE == "all":
    base_t_obs_unique_phys = t_values.reshape(-1, 1)

elif OBSERVATION_CASE == "half":
    base_t_obs_unique_phys = t_values[1::2].reshape(-1, 1)

elif OBSERVATION_CASE == "five":
    base_t_obs_unique_phys = BASE_T_OBS_UNIQUE_PHYS

else:
    raise ValueError("Wrong observation case input.")


# ---- Validate observation points ----
for xp, yp in observation_points_phys:
    if not (0.0 <= xp <= L and 0.0 <= yp <= H):
        raise ValueError(
            f"Observation point is outside the domain: "
            f"x={xp}, y={yp}, domain=[0,{L}] x [0,{H}]"
        )


def build_observation_window(t_window_max):
    """
    Builds sparse pressure observations for the current training window.
    """

    global t_obs_unique_phys, n_obs_times
    global x_obs_phys, y_obs_phys, t_obs_phys, p_obs_phys
    global x_obs, y_obs, t_obs, p_obs
    global p_obs_phys_matrix

    mask = base_t_obs_unique_phys.flatten() <= t_window_max + 1e-12
    t_obs_unique_phys = base_t_obs_unique_phys[mask].reshape(-1, 1)

    if len(t_obs_unique_phys) == 0:
        raise ValueError(
            f"No observation times found for t_window_max={t_window_max}."
        )

    n_obs_times = len(t_obs_unique_phys)

    x_obs_phys_list = []
    y_obs_phys_list = []
    t_obs_phys_list = []
    p_obs_phys_list = []

    p_obs_phys_matrix = np.zeros((n_obs_points, n_obs_times))

    for i, (xp, yp) in enumerate(observation_points_phys):

        for j, t_phys in enumerate(t_obs_unique_phys.flatten()):

            t_idx_obs = int(np.argmin(np.abs(t_values - t_phys)))

            p_val = griddata(
                np.hstack([x_dim, y_dim]),
                p_wide[:, t_idx_obs],
                np.array([[xp, yp]]),
                method="linear"
            )[0]

            if np.isnan(p_val):
                raise ValueError(
                    f"griddata returned NaN for observation point "
                    f"(x={xp}, y={yp}) at t={t_phys}."
                )

            x_obs_phys_list.append(xp)
            y_obs_phys_list.append(yp)
            t_obs_phys_list.append(t_phys)
            p_obs_phys_list.append(p_val)

            p_obs_phys_matrix[i, j] = p_val

    x_obs_phys = np.array(x_obs_phys_list).reshape(-1, 1)
    y_obs_phys = np.array(y_obs_phys_list).reshape(-1, 1)
    t_obs_phys = np.array(t_obs_phys_list).reshape(-1, 1)
    p_obs_phys = np.array(p_obs_phys_list).reshape(-1, 1)

    x_obs = torch.tensor(x_obs_phys / x_ref, dtype=torch.float32, device=device)
    y_obs = torch.tensor(y_obs_phys / y_ref, dtype=torch.float32, device=device)
    t_obs = torch.tensor(t_obs_phys / t_ref, dtype=torch.float32, device=device)
    p_obs = torch.tensor(p_obs_phys / p_ref, dtype=torch.float32, device=device)

    print("\nObservation window")
    print(f"t <= {t_window_max:.3f} s")
    print(f"Number of observation times: {n_obs_times}")
    print(f"Total pressure data points : {len(p_obs_phys)}")


def build_training_window(t_window_max):
    """
    Rebuilds all time-dependent training tensors for the current time window.
    The model is NOT reinitialized.
    """

    global x_d, y_d, t_d
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

    t_d = torch.tensor(
        t_d_np,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

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

    # -------------------------
    # Pressure observations
    # -------------------------
    build_observation_window(t_window_max)

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
    return model.A0 + model.A1_star() * torch.sin(
        2.0 * np.pi * t_star / model.T_star()
    )

# Preserve the historical sampling sequence used in the thesis runs.
# The first training window is rebuilt when Adam training begins.
initial_observation_window = training_schedule[0][0]

build_training_window(initial_observation_window)
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
# Inverse problem info
# =========================

def fmt_num(v, decimals=2):
    v = float(v)
    if v == int(v):
        return str(int(v))
    return f"{v:.{decimals}f}"

def fmt_vec(arr, decimals=2):
    return "[" + ", ".join(fmt_num(v, decimals) for v in np.asarray(arr).flatten()) + "]"

print("\n" + "=" * 60)
print("Inverse run setup")
print("=" * 60)
print(f"Run name          : {run_name}")
print(f"Inverse case      : {inverse_case_name}")
print(f"Transfer learning: {USE_TRANSFER_LEARNING}")

if USE_TRANSFER_LEARNING:
    print(f"Pretrained model  : {STAGE3_CHECKPOINT_PATH}")
print("Pressure ansatz     : hard inlet/outlet pressure ansatz")
print("Wall BCs            : hard no-slip wall boundary conditions")
print("Collocation         : random uniform")
print(f"Experiment family  : {EXPERIMENT_FAMILY}")
print(f"Training profile   : {TRAINING_PROFILE}")
print(f"Inverse LR schedule: {INVERSE_LR_BY_STAGE}")
print(f"Time windows       : {time_windows_phys}")
print(f"Epochs per window  : {epochs_per_window}")
print(f"Run L-BFGS         : {run_lbfgs}")
print(f"L-BFGS iterations  : {max_lbfgs_iterations if run_lbfgs else 0}")
print(f"Infer A0           : True")
print(f"Infer A1           : {infer_A1}")
print(f"Infer T            : {infer_T}")

def fmt_points(points, decimals=2):
    return "[" + ", ".join(
        f"(x={fmt_num(x, decimals)}, y={fmt_num(y, decimals)})"
        for x, y in points
    ) + "]"

sensitivities = 1.0 - observation_points_phys[:, 0] / L

print(f"Observation case   : {OBSERVATION_CASE}")
print(f"Observation points : {fmt_points(observation_points_phys)} m")
print(f"Observation keys   : {SELECTED_OBSERVATION_POINTS}")
print(f"Inlet sensitivities: {fmt_vec(sensitivities)}")
print(f"Number of points   : {n_obs_points}")
print(f"Number of times    : {n_obs_times}")
print(f"Total p data       : {len(p_obs_phys)}")
print(f"Observation times  : {fmt_vec(t_obs_unique_phys)} s")
print("Measured p per point (Pa):")
for i, key in enumerate(SELECTED_OBSERVATION_POINTS):
    xp, yp = observation_points_phys[i]
    print(
        f"  {key} at (x={xp:g}, y={yp:g}): "
        f"{fmt_vec(p_obs_phys_matrix[i, :])}"
    )
print("=" * 60)

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
    "Data": 1.0,
}

# =========================
# Histories creation
# =========================

loss_history = []
epoch_history = []

loss_term_history = {key: [] for key in weights}

lr_history = []
lr_change_history = []
previous_lr = optimizer_adam.param_groups[0]["lr"]

A0_history = []
A1_history = []
T_history = []

A0_history_lbfgs = []
A1_history_lbfgs = []
T_history_lbfgs = []

loss_history_lbfgs = []
lbfgs_iter_history = []
lbfgs_iter = 0

adam_stage_time_history = []
adam_stage_window_history = []
adam_stage_epoch_history = []

# =========================
# Gradient calculator
# =========================

def grad_calc(y,x):
    grads = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True)[0]
    return grads

# =========================
# Loss function
# =========================
def loss_function(return_diag=False):

    # ---- Physics loss calculation ----

    # Model Output on Collocation Points:
    output = model(y_d, x_d, t_d)   # (y, x, t) -> (u_y, u_x, p)

    u_y_pred = output[:, 0:1]  # u_y*
    u_x_pred = output[:, 1:2]  # u_x*
    p_pred = output[:, 2:3]    # p*

    # Gradient calculation in nondimensional coordinates
    u_y_y = grad_calc(u_y_pred, y_d)
    u_y_yy = grad_calc(u_y_y, y_d)
    u_y_x = grad_calc(u_y_pred, x_d)
    u_y_xx = grad_calc(u_y_x, x_d)

    u_x_y = grad_calc(u_x_pred, y_d)
    u_x_yy = grad_calc(u_x_y, y_d)
    u_x_x = grad_calc(u_x_pred, x_d)
    u_x_xx = grad_calc(u_x_x, x_d)

    u_x_t = grad_calc(u_x_pred, t_d)
    u_y_t = grad_calc(u_y_pred, t_d)

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
            "|Time_y|"      : NS_terms_y["time_y"].abs().mean().item(),
            "|Conv_y|"      : NS_terms_y["conv_y"].abs().mean().item(),
            "|Press_y|"     : NS_terms_y["press_y"].abs().mean().item(),
            "|Visc_y|"      : NS_terms_y["visc_y"].abs().mean().item(),
            "Res_y_mean"    : residual_y.mean().item(),
            "|Res_y_max|"   : residual_y.abs().max().item()
        }

        diag_x = {
            "|Time_x|"      : NS_terms_x["time_x"].abs().mean().item(),
            "|Conv_x|"      : NS_terms_x["conv_x"].abs().mean().item(),
            "|Press_x|"     : NS_terms_x["press_x"].abs().mean().item(),
            "|Visc_x|"      : NS_terms_x["visc_x"].abs().mean().item(),
            "Res_x_mean"    : residual_x.mean().item(),
            "|Res_x_max|"   : residual_x.abs().max().item()
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

    # ---- Initial Condition loss calculation ----
    output_ic = model(y_ic, x_ic, t_ic)

    u_y_ic_pred = output_ic[:, 0:1]
    u_x_ic_pred = output_ic[:, 1:2]
    p_ic_pred   = output_ic[:,2:3]

    loss_ic = torch.mean(((u_y_ic_pred - u_y_ic)**2) + ((u_x_ic_pred - u_x_ic)**2) + ((p_ic_pred - p_ic)**2))

    # ---- Observation Data loss calculation ----
    output_data_obs = model(y_obs, x_obs, t_obs)

    p_data_pred = output_data_obs[:, 2:3]

    loss_data = torch.mean((p_data_pred - p_obs)**2)

    # ---- Total loss ----

    loss_terms = {
        "Cont": loss_c,
        "NS_y": loss_y,
        "NS_x": loss_x,
        "Inlet": loss_in,
        "Outlet": loss_out,
        "Initial": loss_ic,
        "Data": loss_data,
    }

    loss = sum(weights[key] * loss_terms[key] for key in loss_terms)

    return loss, loss_terms, diag_y, diag_x

# =========================
# Output utilities
# =========================

def stop_if_output_exists(path):
    path = Path(path)

    if path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        raise FileExistsError(
            f"Output already exists and would be overwritten:\n"
            f"{path}\n\n"
            f"Delete the existing output or set OVERWRITE_EXISTING_OUTPUTS = True."
        )

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

def check_existing_outputs_before_run():
    if TRAIN_MODE and SAVE_CHECKPOINT:
        stop_if_output_exists(model_path)

    if CREATE_PLOTS and plot_dir.exists() and any(plot_dir.iterdir()) and not OVERWRITE_EXISTING_OUTPUTS:
        raise FileExistsError(
            f"Plot folder already exists and is not empty:\n"
            f"{plot_dir}\n\n"
            f"Delete the existing folder or set OVERWRITE_EXISTING_OUTPUTS = True."
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
        "stage": "stage4_pulsatile_inverse",
        "formulation": "nondimensional",
        "weight_initialization": "xavier_normal",
        "activation": "tanh",
        "hard_bcs": True,
        "hard_pressure": True,
        "collocation_sampling": "random_uniform",

        "run_name": run_name,
        "model_path": str(model_path),
        "figures_dir": str(plot_dir),

        "EXPERIMENT_FAMILY": EXPERIMENT_FAMILY,
        "inverse_case_name": inverse_case_name,
        "infer_A0": True,
        "infer_A1": infer_A1,
        "infer_T": infer_T,
        "TRAINING_PROFILE": TRAINING_PROFILE,

        "USE_TRANSFER_LEARNING": USE_TRANSFER_LEARNING,
        "pretrained_model_path": str(STAGE3_CHECKPOINT_PATH) if USE_TRANSFER_LEARNING else None,

        "OBSERVATION_CASE": OBSERVATION_CASE,
        "SELECTED_OBSERVATION_POINTS": SELECTED_OBSERVATION_POINTS,
        "observation_points_phys": observation_points_phys,
        "n_obs_points": n_obs_points,
        "n_obs_times": n_obs_times if "n_obs_times" in globals() else None,

        "family_tag": family_tag,
        "tl_tag": tl_tag,
        "adam_tag": adam_tag,
        "lbfgs_tag": lbfgs_tag,
        "spatial_tag": spatial_tag,
        "temporal_tag": temporal_tag,
        "initial_tags": initial_tags,
        "run_name_parts": run_name_parts,

        "seed": seed,
        "seed_mode": "random" if RANDOM_SEED else "fixed",

        "physical_parameters": {
            "mu": mu,
            "rho": rho,
            "A0_true": A0_true,
            "A1_true": A1_true,
            "T_true": T_true,
            "p_out_dim": p_out_dim,
            "t_initial": t_initial,
            "t_final": t_final,
            "L": L,
            "H": H,
            "cycles": cycles,
        },

        "initial_values": {
            "A0_initial": A0_initial,
            "A1_initial": A1_initial,
            "T_initial": T_initial,
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
            "TRAINING_PROFILE": TRAINING_PROFILE,
            "INVERSE_LR_BY_STAGE": list(INVERSE_LR_BY_STAGE),
            "training_schedule": training_schedule,
            "total_adam_epochs": configured_total_adam_epochs,
            "time_windows_phys": time_windows_phys,
            "epochs_per_window": epochs_per_window,
            "LR_ADAM": LR_ADAM,
            "LR_LBFGS": LR_LBFGS,
            "max_lbfgs_iterations": max_lbfgs_iterations,
            "run_lbfgs": run_lbfgs,
            "RESET_LR_EACH_WINDOW": RESET_LR_EACH_WINDOW,
        },

        "loss_weights": weights,
        "final_stage_name": final_stage_name,
    }

def save_stage4_checkpoint(
    final_stage_name,
    adam_time,
    lbfgs_time,
    optimizer_time,
    overhead_time,
    total_time
):
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

        "A0_history": A0_history,
        "A1_history": A1_history,
        "T_history": T_history,
        "A0_history_lbfgs": A0_history_lbfgs,
        "A1_history_lbfgs": A1_history_lbfgs,
        "T_history_lbfgs": T_history_lbfgs,

        "adam_stage_time_history": adam_stage_time_history,
        "adam_stage_window_history": adam_stage_window_history,
        "adam_stage_epoch_history": adam_stage_epoch_history,
        "adam_time": adam_time,
        "lbfgs_time": lbfgs_time,
        "optimizer_time": optimizer_time,
        "overhead_time": overhead_time,
        "total_time": total_time,

        "A0_after_adam": A0_after_adam,
        "A1_after_adam": A1_after_adam,
        "T_after_adam": T_after_adam,
        "A0_final": A0_after_lbfgs,
        "A1_final": A1_after_lbfgs,
        "T_final": T_after_lbfgs,
        "final_stage_name": final_stage_name,

        "EXPERIMENT_FAMILY": EXPERIMENT_FAMILY,
        "inverse_case_name": inverse_case_name,
        "infer_A0": True,
        "infer_A1": infer_A1,
        "infer_T": infer_T,
        "TRAINING_PROFILE": TRAINING_PROFILE,
        "OBSERVATION_CASE": OBSERVATION_CASE,

        "USE_TRANSFER_LEARNING": USE_TRANSFER_LEARNING,
        "pretrained_model_path": (
            str(STAGE3_CHECKPOINT_PATH)
            if USE_TRANSFER_LEARNING else None
        ),

        "training_schedule": training_schedule,
        "total_adam_epochs": configured_total_adam_epochs,
        "time_windows_phys": time_windows_phys,
        "epochs_per_window": epochs_per_window,
        "RESET_LR_EACH_WINDOW": RESET_LR_EACH_WINDOW,
        "run_lbfgs": run_lbfgs,

        "LR_ADAM": LR_ADAM,
        "INVERSE_LR_BY_STAGE": list(INVERSE_LR_BY_STAGE),
        "LR_LBFGS": LR_LBFGS,
        "max_lbfgs_iterations": max_lbfgs_iterations,

        "run_name": run_name,
        "model_path": str(model_path),
        "plot_dir": str(plot_dir),

        "hard_bcs": True,
        "hard_pressure": True,
        "adam_scheduler": "MultiStepLR",

        "x_ref": x_ref,
        "y_ref": y_ref,
        "t_ref": t_ref,
        "u_ref": u_ref,
        "p_ref": p_ref,
        "Re": Re,
        "St": St,

        "run_config": build_run_config(final_stage_name),

        "family_tag": family_tag,
        "tl_tag": tl_tag,
        "adam_tag": adam_tag,
        "lbfgs_tag": lbfgs_tag,
        "spatial_tag": spatial_tag,
        "temporal_tag": temporal_tag,
        "initial_tags": initial_tags,
        "run_name_parts": run_name_parts,

        "t_obs_phys": t_obs_phys,
        "p_obs_phys": p_obs_phys,
        "A0_true": A0_true,
        "A1_true": A1_true,
        "T_true": T_true,
        "A0_initial": A0_initial,
        "A1_initial": A1_initial,
        "T_initial": T_initial,

        "SELECTED_OBSERVATION_POINTS": SELECTED_OBSERVATION_POINTS,
        "OBSERVATION_POINT_CATALOG": OBSERVATION_POINT_CATALOG,
        "observation_points_phys": observation_points_phys,
        "n_obs_points": n_obs_points,
        "n_obs_times": n_obs_times,
        "t_obs_unique_phys": t_obs_unique_phys,
        "p_obs_phys_matrix": p_obs_phys_matrix,

        "environment": {
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    stop_if_output_exists(model_path)

    torch.save(checkpoint, model_path)
    print(f"\nCheckpoint saved to: {model_path}")


def load_stage4_checkpoint():
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{model_path}\n\n"
            f"Run first with TRAIN_MODE = True and SAVE_CHECKPOINT = True."
        )

    checkpoint = load_torch_checkpoint(model_path)

    if checkpoint.get("layers") != layers:
        raise ValueError(
            "Loaded Stage 4 checkpoint architecture does not match "
            "the current model.\n"
            f"Checkpoint layers: {checkpoint.get('layers')}\n"
            f"Current layers:    {layers}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Stage 4 checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\nCheckpoint loaded from: {model_path}")

    return checkpoint

# =========================
# Adam + L-BFGS training or checkpoint loading
# =========================

if TRAIN_MODE:
    check_existing_outputs_before_run()

    # =========================
    # Adam training
    # =========================

    total_start_time = sync_time()
    adam_start_time = sync_time()

    print("\n" + "=" * 60)
    print("Starting Adam optimizer...")
    print("=" * 60)

    global_epoch = 0

    print("\n" + "=" * 60)
    print(f"Planned total Adam epochs: {configured_total_adam_epochs}")
    print("=" * 60)

    for stage_id, (t_window_max, stage_epochs) in enumerate(training_schedule, start=1):

        print("\n" + "=" * 60)
        print(f"Adam stage {stage_id}/{len(training_schedule)}")
        print(f"Training on t ∈ [0, {t_window_max}] s")
        print(f"Stage Adam epochs: {stage_epochs}")
        print("=" * 60)

        build_training_window(t_window_max)

        if RESET_LR_EACH_WINDOW:

            # Reset network Adam LR
            for group in optimizer_adam.param_groups:
                group["lr"] = LR_ADAM

        # Apply the configured inverse-parameter LR for the current stage.
        if optimizer_inverse_params is not None:

            print("\nInverse-parameter learning rates for this stage:")

            for group in optimizer_inverse_params.param_groups:

                param_name = group["name"]
                new_lr = get_inverse_lr_for_stage(stage_id)

                group["lr"] = new_lr

                print(f"  {param_name}: lr = {new_lr:.4e}")

        scheduler = make_adam_scheduler(optimizer_adam, stage_epochs)
        previous_lr = optimizer_adam.param_groups[0]["lr"]

        stage_adam_start_time = sync_time()

        for local_epoch in range(1, stage_epochs + 1):

            global_epoch += 1

            lr_used = optimizer_adam.param_groups[0]["lr"]

            optimizer_adam.zero_grad()

            if optimizer_inverse_params is not None:
                optimizer_inverse_params.zero_grad()

            do_print = (
                global_epoch % 500 == 0
                or local_epoch == stage_epochs
            )

            loss, loss_terms, diag_y, diag_x = loss_function(return_diag=do_print)
            loss.backward()

            optimizer_adam.step()

            if optimizer_inverse_params is not None:
                optimizer_inverse_params.step()

            scheduler.step()

            current_lr = optimizer_adam.param_groups[0]["lr"]

            lr_history.append(lr_used)

            if current_lr != previous_lr:
                lr_change_history.append(
                    (global_epoch, previous_lr, current_lr)
                )

                print(
                    f"[LR CHANGE] Global epoch {global_epoch}: "
                    f"{previous_lr:.4e} -> {current_lr:.4e}"
                )

                previous_lr = current_lr

            loss_history.append(loss.item())
            epoch_history.append(global_epoch)

            A0_history.append(model.A0.item() * p_ref)
            A1_history.append(model.A1_star().item() * p_ref)
            T_history.append(model.T_star().item() * t_ref)

            for key in weights:
                loss_term_history[key].append(loss_terms[key].item())

            weighted_loss_terms = {
                key: weights[key] * loss_terms[key].item()
                for key in weights
            }

            weighted_loss_total = (
                sum(weighted_loss_terms.values()) + 1e-12
            )

            loss_term_percentages = {
                key: 100.0
                * weighted_loss_terms[key]
                / weighted_loss_total
                for key in weights
            }

            if do_print:

                print(
                    f"Stage {stage_id}/{len(training_schedule)}, "
                    f"Epoch {global_epoch}/{configured_total_adam_epochs}, "
                    f"Stage progress {local_epoch}/{stage_epochs}, "
                    f"Total Loss: {loss.item():.4e}"
                )

                print("\n--- Loss terms ---")
                for key in weights:
                    print(
                        f"{key:12s}: {loss_terms[key].item():.4e}  "
                        f"({loss_term_percentages[key]:.2f}%)"
                    )

                print("\n--- NS_y diagnostics ---")
                for key, value in diag_y.items():
                    print(f"{key:12s}: {value:+.4e}")

                print("\n--- NS_x diagnostics ---")
                for key, value in diag_x.items():
                    print(f"{key:12s}: {value:+.4e}")

                print(f"\nAdam network learning rate for next epoch: {current_lr:.4e}")

                print(
                    f"[INV] A0 = {model.A0.item()*p_ref:.4f} Pa | "
                    f"A1 = {model.A1_star().item()*p_ref:.4f} Pa "
                    f"({'trainable' if infer_A1 else 'fixed'}) | "
                    f"T = {model.T_star().item()*t_ref:.6f} s "
                    f"({'trainable' if infer_T else 'fixed'})"
                )

                print("-" * 60)

        stage_adam_end_time = sync_time()
        stage_adam_time = stage_adam_end_time - stage_adam_start_time

        adam_stage_time_history.append(stage_adam_time)
        adam_stage_window_history.append(t_window_max)
        adam_stage_epoch_history.append(stage_epochs)

        print("\n" + "=" * 60)
        print(f"Adam stage {stage_id} time")
        print("=" * 60)
        print(f"Window t <= {t_window_max} s")
        print(f"Stage epochs : {stage_epochs}")
        print(f"Stage time   : {stage_adam_time:.2f} s ({stage_adam_time/60:.2f} min)")
        print("=" * 60)

    adam_end_time = sync_time()
    adam_time = adam_end_time - adam_start_time

    if global_epoch != configured_total_adam_epochs:
        raise RuntimeError(
            f"Adam epoch counter mismatch: "
            f"global_epoch={global_epoch}, expected={configured_total_adam_epochs}"
        )

    A0_after_adam = model.A0.item() * p_ref
    A1_after_adam = model.A1_star().item() * p_ref
    T_after_adam = model.T_star().item() * t_ref

    print("\n--- Inverse check after Adam ---")
    print(f"A0 after Adam = {A0_after_adam:.4f} Pa | target {A0_true} | dev = {100*abs(A0_after_adam-A0_true)/abs(A0_true):.3f}%")
    print(f"A1 after Adam = {A1_after_adam:.4f} Pa | target {A1_true} | dev = {100*abs(A1_after_adam-A1_true)/abs(A1_true):.3f}%")
    print(f"T  after Adam = {T_after_adam:.6f} s | target {T_true} | dev = {100*abs(T_after_adam-T_true)/abs(T_true):.3f}%")

    print("\n=== Adam learning rate changes ===")

    if len(lr_change_history) == 0:
        print("No learning rate changes occurred during Adam training.")
    else:
        for epoch, old_lr, new_lr in lr_change_history:
            print(f"Epoch {epoch}: {old_lr:.4e} -> {new_lr:.4e}")

    # =========================
    # L-BFGS training
    # =========================

    # Rebuild the full time window before the final report and optional L-BFGS.
    build_training_window(t_final)

    if run_lbfgs:

        print("\n" + "=" * 60)
        print("Starting L-BFGS optimizer on full time window...")
        print("=" * 60)

        lbfgs_start_time = sync_time()

        model.train()
        optimizer_lbfgs.step(closure)

        lbfgs_end_time = sync_time()
        lbfgs_time = lbfgs_end_time - lbfgs_start_time

        print("\n" + "=" * 60)
        print(f"L-BFGS training time: {lbfgs_time:.2f} s ({lbfgs_time/60:.2f} min)")
        print("=" * 60)

        loss, loss_terms, diag_y, diag_x = loss_function(return_diag=True)

        A0_after_lbfgs = model.A0.item() * p_ref
        A1_after_lbfgs = model.A1_star().item() * p_ref
        T_after_lbfgs  = model.T_star().item() * t_ref

        final_stage_name = "after L-BFGS"

    else:

        lbfgs_time = 0.0

        print("\n" + "=" * 60)
        print("L-BFGS skipped.")
        print("Final report is based on the model after Adam training.")
        print("=" * 60)

        loss, loss_terms, diag_y, diag_x = loss_function(return_diag=True)

        A0_after_lbfgs = model.A0.item() * p_ref
        A1_after_lbfgs = model.A1_star().item() * p_ref
        T_after_lbfgs  = model.T_star().item() * t_ref

        final_stage_name = "after Adam"


    # =========================
    # Final loss report
    # =========================

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
        print(
            f"{key:12s}: {loss_terms[key].item():.4e}  "
            f"({loss_term_percentages[key]:.2f}%)"
        )

    print(f"\n--- Final NS_y diagnostics ---")
    for key, value in diag_y.items():
        print(f"{key:12s}: {value:+.4e}")

    print(f"\n--- Final NS_x diagnostics ---")
    for key, value in diag_x.items():
        print(f"{key:12s}: {value:+.4e}")


    # =========================
    # Final inverse check
    # =========================

    print(f"\n--- Inverse check {final_stage_name} ---")
    print(
        f"A0 final = {A0_after_lbfgs:.4f} Pa | "
        f"target {A0_true} | "
        f"dev = {100*abs(A0_after_lbfgs-A0_true)/abs(A0_true):.3f}%"
    )

    print(
        f"A1 final = {A1_after_lbfgs:.4f} Pa | "
        f"target {A1_true} | "
        f"dev = {100*abs(A1_after_lbfgs-A1_true)/abs(A1_true):.3f}%"
    )

    print(
        f"T  final = {T_after_lbfgs:.6f} s | "
        f"target {T_true} | "
        f"dev = {100*abs(T_after_lbfgs-T_true)/abs(T_true):.3f}%"
    )

    total_end_time = sync_time()
    total_time = total_end_time - total_start_time

    optimizer_time = adam_time + lbfgs_time
    overhead_time = max(0.0, total_time - optimizer_time)

    print("\n" + "=" * 60)
    print("Training time summary")
    print("=" * 60)

    for i, (window, stage_epochs, stage_time) in enumerate(
        zip(adam_stage_window_history, adam_stage_epoch_history, adam_stage_time_history),
        start=1
    ):
        print(
            f"Adam stage {i} | "
            f"t <= {window} s | "
            f"epochs = {stage_epochs} | "
            f"time = {stage_time:.2f} s ({stage_time/60:.2f} min)"
        )

    print("-" * 60)
    print(f"Total Adam time  : {adam_time:.2f} s ({adam_time/60:.2f} min)")
    print(f"L-BFGS time      : {lbfgs_time:.2f} s ({lbfgs_time/60:.2f} min)")
    print(f"Optimizer time   : {optimizer_time:.2f} s ({optimizer_time/60:.2f} min)")
    print(f"Overhead time    : {overhead_time:.2f} s ({overhead_time/60:.2f} min)")
    print(f"Total train time : {total_time:.2f} s ({total_time/60:.2f} min)")
    print("=" * 60)

    if SAVE_CHECKPOINT:
        save_stage4_checkpoint(
            final_stage_name=final_stage_name,
            adam_time=adam_time,
            lbfgs_time=lbfgs_time,
            optimizer_time=optimizer_time,
            overhead_time=overhead_time,
            total_time=total_time
        )

else:
    checkpoint = load_stage4_checkpoint()

    def ckpt_get(key, default=None):
        return checkpoint[key] if key in checkpoint else default

    EXPERIMENT_FAMILY = ckpt_get(
        "EXPERIMENT_FAMILY",
        EXPERIMENT_FAMILY
    )

    inverse_case_name = ckpt_get(
        "inverse_case_name",
        inverse_case_name
    )

    infer_A1 = ckpt_get(
        "infer_A1",
        infer_A1
    )

    infer_T = ckpt_get(
        "infer_T",
        infer_T
    )

    TRAINING_PROFILE = ckpt_get(
        "TRAINING_PROFILE",
        TRAINING_PROFILE
    )

    A0_after_lbfgs = model.A0.item() * p_ref
    A1_after_lbfgs = model.A1_star().item() * p_ref
    T_after_lbfgs  = model.T_star().item() * t_ref

    loss_history        = checkpoint["loss_history"]
    epoch_history       = checkpoint["epoch_history"]
    loss_term_history   = checkpoint["loss_term_history"]
    weights             = checkpoint["weights"]
    lr_history          = checkpoint["lr_history"]
    lr_change_history   = checkpoint["lr_change_history"]

    A0_history          = checkpoint["A0_history"]
    A1_history          = checkpoint["A1_history"]
    T_history           = checkpoint["T_history"]
    A0_history_lbfgs    = checkpoint["A0_history_lbfgs"]
    A1_history_lbfgs    = checkpoint["A1_history_lbfgs"]
    T_history_lbfgs     = checkpoint["T_history_lbfgs"]

    loss_history_lbfgs = ckpt_get("loss_history_lbfgs", [])
    lbfgs_iter_history = ckpt_get(
        "lbfgs_iter_history",
        list(range(len(loss_history_lbfgs)))
    )

    adam_stage_time_history = ckpt_get("adam_stage_time_history", [])
    adam_stage_window_history = ckpt_get("adam_stage_window_history", [])
    adam_stage_epoch_history = ckpt_get("adam_stage_epoch_history", [])

    adam_time = ckpt_get("adam_time", None)
    lbfgs_time = ckpt_get("lbfgs_time", None)
    optimizer_time = ckpt_get("optimizer_time", None)
    overhead_time = ckpt_get("overhead_time", None)
    total_time = ckpt_get("total_time", None)

    A0_after_adam = ckpt_get(
        "A0_after_adam",
        A0_history[-1] if len(A0_history) > 0 else None
    )

    A1_after_adam = ckpt_get(
        "A1_after_adam",
        A1_history[-1] if len(A1_history) > 0 else None
    )

    T_after_adam = ckpt_get(
        "T_after_adam",
        T_history[-1] if len(T_history) > 0 else None
    )

    final_stage_name = ckpt_get("final_stage_name", "loaded checkpoint")

    OBSERVATION_CASE    = checkpoint["OBSERVATION_CASE"]
    run_name            = checkpoint["run_name"]
    plot_dir = FIGURES_ROOT / run_name
    check_existing_outputs_before_run()
    USE_TRANSFER_LEARNING = ckpt_get("USE_TRANSFER_LEARNING", USE_TRANSFER_LEARNING)

    t_obs_phys          = checkpoint["t_obs_phys"]
    p_obs_phys          = checkpoint["p_obs_phys"]
    A0_true             = checkpoint["A0_true"]
    A1_true             = checkpoint["A1_true"]
    T_true              = checkpoint["T_true"]
    A0_initial          = checkpoint["A0_initial"]
    A1_initial          = checkpoint["A1_initial"]
    T_initial           = checkpoint["T_initial"]
    SELECTED_OBSERVATION_POINTS = ckpt_get(
        "SELECTED_OBSERVATION_POINTS",
        SELECTED_OBSERVATION_POINTS
    )

    OBSERVATION_POINT_CATALOG = ckpt_get(
        "OBSERVATION_POINT_CATALOG",
        OBSERVATION_POINT_CATALOG
    )

    observation_points_phys = ckpt_get(
        "observation_points_phys",
        observation_points_phys
    )

    n_obs_points = ckpt_get(
        "n_obs_points",
        len(SELECTED_OBSERVATION_POINTS)
    )

    n_obs_times = ckpt_get(
        "n_obs_times",
        len(t_obs_phys) if "t_obs_phys" in checkpoint else n_obs_times
    )

    t_obs_unique_phys = ckpt_get(
        "t_obs_unique_phys",
        np.unique(t_obs_phys).reshape(-1, 1) if "t_obs_phys" in checkpoint else t_obs_unique_phys
    )

    p_obs_phys_matrix = ckpt_get(
        "p_obs_phys_matrix",
        None
    )

    if total_time is not None:
        print("\nLoaded timing information:")

        if len(adam_stage_time_history) > 0:
            for i, (window, stage_epochs, stage_time) in enumerate(
                zip(adam_stage_window_history, adam_stage_epoch_history, adam_stage_time_history),
                start=1
            ):
                print(
                    f"Adam stage {i} | "
                    f"t <= {window} s | "
                    f"epochs = {stage_epochs} | "
                    f"time = {stage_time:.2f} s ({stage_time/60:.2f} min)"
                )

        if adam_time is not None:
            print(f"Total Adam time  : {adam_time:.2f} s ({adam_time/60:.2f} min)")

        if lbfgs_time is not None:
            print(f"L-BFGS time      : {lbfgs_time:.2f} s ({lbfgs_time/60:.2f} min)")

        if optimizer_time is not None:
            print(f"Optimizer time   : {optimizer_time:.2f} s ({optimizer_time/60:.2f} min)")

        if overhead_time is not None:
            print(f"Overhead time    : {overhead_time:.2f} s ({overhead_time/60:.2f} min)")

        print(f"Total train time : {total_time:.2f} s ({total_time/60:.2f} min)")
    else:
        print("\nNo timing information found in this checkpoint.")
    print(f"(originally trained with seed {checkpoint['seed']})")
    print("Weights used for this trained model:")
    for k, v in weights.items():
        print(f"  {k:12s} = {v}")

# =========================
# Evaluation
# =========================
model.eval()

with torch.no_grad():
    output_data = model(y_data, x_data, t_data)

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
# Run Summary
# =========================
print("\n" + "=" * 60)
print("RUN SUMMARY")
print("=" * 60)
print(f"Run name          : {run_name}")
print(f"Inverse case      : {inverse_case_name}")
print(f"Observation case   : {OBSERVATION_CASE}")
print(f"Observation points : {fmt_points(observation_points_phys)} m")
print(f"Observation keys   : {SELECTED_OBSERVATION_POINTS}")
print(f"Inlet sensitivities: {fmt_vec(1.0 - observation_points_phys[:, 0] / L)}")
print(f"Number of points   : {n_obs_points}")
print(f"Number of times    : {n_obs_times}")
print(f"Total p data       : {len(p_obs_phys)}")
print(f"A0 final          : {A0_after_lbfgs:.6f} Pa")
print(f"A1 final          : {A1_after_lbfgs:.6f} Pa")
print(f"T final           : {T_after_lbfgs:.6f} s")
print(f"rel_L2 u_x        : {100*rel_l2_u_x_dim.item():.4f}%")
print(f"rel_L2 p          : {100*rel_l2_p_dim.item():.4f}%")
print("=" * 60)

# =========================
# Per-time-slice error
# =========================

# Per-time-slice relative L2 errors
t_np   = t_data.cpu().numpy().flatten()
uniq_t = np.unique(t_np)
print("\n=== Per-time-slice error ===")
print(f"{'t (s)':>7} | {'u_x rel':>9} | {'p rel':>9} | {'p RMSE (Pa)':>11}")
print("-" * 48)

for tj in uniq_t:
    idx = torch.tensor(np.where(np.isclose(t_np, tj))[0], device=device)

    eu   = 100 * relative_l2(u_x_pred[idx], u_x_data[idx]).item()
    ep   = 100 * relative_l2(p_pred[idx],   p_data[idx]).item()
    p_rmse = rmse(p_pred[idx], p_data[idx]).item()

    print(f"{tj*t_ref:7.2f} | {eu:8.3f}% | {ep:8.3f}% | {p_rmse:9.4f}")

# =========================
# Boundary checks
# =========================
with torch.no_grad():
    # Upper Wall predictions
    output_wall_upper = model(y_wall_upper, x_wall_upper, t_wall_upper)

    u_y_wall_pred_upper = output_wall_upper[:, 0:1] * u_ref
    u_x_wall_pred_upper = output_wall_upper[:, 1:2] * u_ref

    # Lower Wall predictions
    output_wall_lower = model(y_wall_lower, x_wall_lower, t_wall_lower)

    u_y_wall_pred_lower = output_wall_lower[:, 0:1] * u_ref
    u_x_wall_pred_lower = output_wall_lower[:, 1:2] * u_ref

    # Pressure inlet/outlet
    output_in = model(y_in, x_in, t_in)
    output_out = model(y_out, x_out, t_out)

    p_in_pred = output_in[:, 2:3] * p_ref
    p_out_pred = output_out[:, 2:3] * p_ref

    p_in_target = p_in_star(t_in) * p_ref
    err_in = torch.abs(p_in_pred - p_in_target)

    # Velocity inlet/outlet
    u_y_in_pred = output_in[:, 0:1] * u_ref
    u_y_out_pred = output_out[:, 0:1] * u_ref

print("\n=== Boundary checks ===")
print(f"Max |u_y_wall_upper| (m/s):     {torch.max(torch.abs(u_y_wall_pred_upper)).item():.6e}")
print(f"Mean |u_y_wall_upper| (m/s):    {torch.mean(torch.abs(u_y_wall_pred_upper)).item():.6e}")
print(f"Max |u_x_wall_upper| (m/s):     {torch.max(torch.abs(u_x_wall_pred_upper)).item():.6e}")
print(f"Mean |u_x_wall_upper| (m/s):    {torch.mean(torch.abs(u_x_wall_pred_upper)).item():.6e}")
print("-" * 60)
print(f"Max |u_y_wall_lower| (m/s):     {torch.max(torch.abs(u_y_wall_pred_lower)).item():.6e}")
print(f"Mean |u_y_wall_lower| (m/s):    {torch.mean(torch.abs(u_y_wall_pred_lower)).item():.6e}")
print(f"Max |u_x_wall_lower| (m/s):     {torch.max(torch.abs(u_x_wall_pred_lower)).item():.6e}")
print(f"Mean |u_x_wall_lower| (m/s):    {torch.mean(torch.abs(u_x_wall_pred_lower)).item():.6e}")
print("-" * 60)
print(f"Max predicted p_outlet (Pa):    {torch.max(torch.abs(p_out_pred)).item():.6e}")
print(f"Mean predicted p_outlet (Pa):   {torch.mean(p_out_pred).item():.6e}")
print(f"Target p_outlet (Pa):           {p_out_dim:.6e}")
print("-" * 60)
print(f"Max |p_inlet - target| (Pa):  {err_in.max().item():.6e}")
print(f"Mean |p_inlet - target| (Pa): {err_in.mean().item():.6e}")
print(f"Inlet target range (Pa):      [{p_in_target.min().item():.3f}, {p_in_target.max().item():.3f}]")
print("-" * 60)
print(f"Max predicted u_y_inlet (m/s):  {torch.max(torch.abs(u_y_in_pred)).item():.6e}")
print(f"Mean predicted u_y_inlet (m/s):  {torch.mean(u_y_in_pred).item():.6e}")
print(f"Max predicted u_y_outlet (m/s):  {torch.max(torch.abs(u_y_out_pred)).item():.6e}")
print(f"Mean predicted u_y_outlet (m/s): {torch.mean(u_y_out_pred).item():.6e}")

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
# Plotting helpers
# =========================

t_slice_desired = 2.60
t_idx = int(np.argmin(np.abs(t_values - t_slice_desired)))
t_slice = t_values[t_idx]

sl = np.arange(n_nodes) * n_times + t_idx

print(f"Field comparison plots at t = {t_slice:.3f} s ({sl.size} nodes)")

triang_slice = tri.Triangulation(x_plot[sl], y_plot[sl])


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

    fig.suptitle(f"{field_name} comparison for t = {t_slice:.3f} s", fontsize=14)

    # -------------------------
    # COMSOL
    # -------------------------
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

    # -------------------------
    # PINN
    # -------------------------
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

    # -------------------------
    # Absolute error
    # -------------------------
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

    # -------------------------
    # Relative percentage error
    # -------------------------
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
    # Inferred parameter evolution plots
    # =========================

    def plot_param_evolution(adam_hist, lbfgs_hist, target, name, unit, decimals=4):
        fig, ax = plt.subplots(figsize=(10, 6))

        n_adam = len(adam_hist)

        ax.plot(
            range(n_adam),
            adam_hist,
            color="C0",
            linewidth=1.8,
            label=f"{name} (Adam)"
        )

        if len(lbfgs_hist) > 0:
            lbfgs_x = range(n_adam - 1, n_adam + len(lbfgs_hist))
            lbfgs_y = [adam_hist[-1]] + lbfgs_hist

            ax.plot(
                lbfgs_x,
                lbfgs_y,
                color="red",
                linewidth=1.8,
                label=f"{name} (L-BFGS) -> {lbfgs_hist[-1]:.{decimals}f} {unit}"
            )

        ax.axhline(
            target,
            color="k",
            linestyle="--",
            label=f"Target = {target} {unit}"
        )

        ax.set_xlabel("Training step")
        ax.set_ylabel(f"{name} ({unit})")
        ax.set_title(f"{name} inference")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()

        save_fig(fig, f"{name}_evolution")
        show_or_close(fig)

    plot_param_evolution(
        A0_history,
        A0_history_lbfgs,
        A0_true,
        "A0",
        "Pa",
        decimals=4
    )

    if infer_A1:
        plot_param_evolution(
            A1_history,
            A1_history_lbfgs,
            A1_true,
            "A1",
            "Pa",
            decimals=4
        )

    if infer_T:
        plot_param_evolution(
            T_history,
            T_history_lbfgs,
            T_true,
            "T",
            "s",
            decimals=6
        )

    # =========================
    # Pressure time series at selected observation points: PINN vs COMSOL
    # =========================

    t_series = np.linspace(0.0, t_final, 400).reshape(-1, 1)  # s

    for i, key in enumerate(SELECTED_OBSERVATION_POINTS):

        xp, yp = observation_points_phys[i]

        # PINN on a dense time vector at this fixed point
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

        # COMSOL reference interpolated at the same spatial point
        p_series_comsol = np.array([
            griddata(
                np.hstack([x_dim, y_dim]),
                p_wide[:, j],
                np.array([[xp, yp]]),
                method="linear"
            )[0]
            for j in range(n_times)
        ])

        if np.isnan(p_series_comsol).any():
            raise ValueError(
                f"NaN in COMSOL pressure time series at "
                f"(x={xp}, y={yp}). Check interpolation point."
            )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            t_series.flatten(),
            p_series_pinn,
            linewidth=1.8,
            label="PINN"
        )

        ax.plot(
            t_values,
            p_series_comsol,
            "o",
            color="k",
            ms=5,
            alpha=0.6,
            label="COMSOL"
        )

        # Mark the actual observation data used in the inverse loss
        ax.plot(
            t_obs_unique_phys.flatten(),
            p_obs_phys_matrix[i, :],
            "s",
            ms=3,
            label="Observation data"
        )

        ax.set_xlabel("t (s)")
        ax.set_ylabel("p (Pa)")
        ax.set_title(f"Pressure at point (x={xp:g} m, y={yp:g} m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()

        save_fig(fig, f"pressure_at_{key}")
        show_or_close(fig)

    # =========================
    # Phase comparison at (x=5, y=2): u_x and p on twin y-axes
    # =========================
    xp, yp = 5.0, 2.0

    # PINN on a dense time vector at the fixed point
    t_series = np.linspace(0.0, t_final, 400).reshape(-1, 1)          # s
    x_s = torch.tensor(np.full_like(t_series, xp) / x_ref, dtype=torch.float32, device=device)
    y_s = torch.tensor(np.full_like(t_series, yp) / y_ref, dtype=torch.float32, device=device)
    t_s = torch.tensor(t_series / t_ref,                    dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        out_s = model(y_s, x_s, t_s)
        u_x_series_pinn = (out_s[:, 1:2] * u_ref).cpu().numpy().flatten()
        p_series_pinn   = (out_s[:, 2:3] * p_ref).cpu().numpy().flatten()

    # COMSOL reference: nearest exported node to (5,2)
    node = int(np.argmin((x_dim.flatten() - xp)**2 + (y_dim.flatten() - yp)**2))
    print(f"Nearest COMSOL node to (5,2): x={x_dim.flatten()[node]:.3f}, y={y_dim.flatten()[node]:.3f}")
    u_x_series_comsol = u_x_wide[node, :]     # dimensional (n_times,)
    p_series_comsol   = p_wide[node, :]

    fig, ax_v = plt.subplots(figsize=(11, 6))
    ax_p = ax_v.twinx()                        # second y-axis sharing the same x

    # Velocity on the LEFT axis (blue)
    l1, = ax_v.plot(t_series.flatten(), u_x_series_pinn, color="C0", lw=1.8, label="u_x PINN")
    l2, = ax_v.plot(t_values, u_x_series_comsol, "o", color="C0", ms=3, alpha=0.5, label="u_x COMSOL")
    ax_v.set_xlabel("t (s)")
    ax_v.set_ylabel("u_x (m/s)", color="C0")
    ax_v.tick_params(axis="y", labelcolor="C0")

    # Pressure on the RIGHT axis (green)
    l3, = ax_p.plot(t_series.flatten(), p_series_pinn, color="C2", lw=1.8, label="p PINN")
    l4, = ax_p.plot(t_values, p_series_comsol, "s", color="C2", ms=3, alpha=0.5, label="p COMSOL")
    ax_p.set_ylabel("p (Pa)", color="C2")
    ax_p.tick_params(axis="y", labelcolor="C2")

    ax_v.set_title("Phase comparison at (x=5, y=2): velocity vs pressure")
    ax_v.grid(alpha=0.3)
    ax_v.legend(handles=[l1, l2, l3, l4], loc="best", fontsize=9)

    plt.tight_layout()
    save_fig(fig, "phase_comparison_at_5_2")
    show_or_close(fig)

    # =========================
    # Individual loss terms evolution plot
    # =========================

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
    save_fig(fig, "loss_terms_evolution")
    show_or_close(fig)

    # =========================
    # Total loss evolution plot: Adam + L-BFGS
    # =========================

    fig, ax = plt.subplots(figsize=(10, 6))

    # Adam loss
    ax.plot(
        epoch_history,
        loss_history,
        label="Adam loss",
        linewidth=2.0
    )

    # L-BFGS loss continued after the final Adam epoch
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
            label="Adam → L-BFGS"
        )

    else:
        print("\nCombined loss plot: no L-BFGS loss history found.")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Total loss")
    if len(loss_history_lbfgs) > 0:
        ax.set_title(
            "Combined total loss evolution during Adam and L-BFGS training"
        )
    else:
        ax.set_title("Total loss evolution during Adam training")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    save_fig(fig, "combined_loss_evolution")
    show_or_close(fig)

    # =========================
    # Adam learning rate evolution plot
    # =========================

    if len(lr_history) == len(epoch_history) and len(lr_history) > 0:

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            epoch_history,
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
        print("\nAdam learning rate plot skipped.")
        print(f"epoch_history length = {len(epoch_history)}")
        print(f"lr_history length    = {len(lr_history)}")
"""
Stage 3 pressure animation.

This optional script creates a GIF animation of the predicted PINN pressure
field for the pulsatile forward problem.

The script loads the COMSOL reference mesh and time snapshots, loads the
trained Stage 3 PINN checkpoint, and evaluates the pressure field over time.

Paths are resolved automatically relative to the repository root.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# =========================
# Configuration
# =========================

OVERWRITE_EXISTING_OUTPUTS = False

FPS = 12
LEVELS = 60
CMAP = "viridis"

REPO_ROOT = Path(__file__).resolve().parents[1]

COMSOL_DATA_PATH = REPO_ROOT / "data" / "comsol" / "stage3_stage4" / "NS_xy_pulsatile.txt"

CHECKPOINT_PATH = REPO_ROOT / "models" / "stage3_forward_model.pt"

OUTPUT_DIR = REPO_ROOT / "stage3_pulsatile_forward" / "animations"
OUTPUT_GIF_PATH = OUTPUT_DIR / "stage3_pinn_pressure_field.gif"


# =========================
# Physical parameters
# =========================

MU = 1.0
RHO = 1.0

P_IN_MEAN = 10.0
P_IN_AMPLITUDE = 5.0
P_OUT_DIM = 0.0
P_PERIOD = 0.5

DT_EXPORT = 0.05

T_INITIAL = 0.0
T_FINAL = 3.0

LENGTH = 12.0
HEIGHT = 4.0


# =========================
# Runtime parameter setup
# =========================

mu = MU
rho = RHO

A0 = P_IN_MEAN
A1 = P_IN_AMPLITUDE
T = P_PERIOD

p_in_dim = P_IN_MEAN
p_out_dim = P_OUT_DIM
deltaP = p_in_dim - p_out_dim

L = LENGTH
H = HEIGHT

x_ref = L
y_ref = H
t_ref = T
u_ref = deltaP * H**2 / (8.0 * mu * L)
p_ref = rho * u_ref**2


# =========================
# Network architecture
# =========================

INPUT_DIM = 3
OUTPUT_DIM = 3
HIDDEN_LAYERS = 5
NEURONS = 64

layers = [INPUT_DIM] + [NEURONS] * HIDDEN_LAYERS + [OUTPUT_DIM]


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
        # p(x=0,t) = p_in(t), p(x=1,t) = p_out
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
# Output utilities
# =========================

def stop_if_output_exists(path):
    path = Path(path)

    if path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        raise FileExistsError(
            f"Output file already exists and would be overwritten:\n"
            f"{path}\n\n"
            f"Delete the existing file or set OVERWRITE_EXISTING_OUTPUTS = True."
        )


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================
# Load COMSOL data
# =========================

def load_comsol_data():
    if not COMSOL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"COMSOL file not found:\n{COMSOL_DATA_PATH}"
        )

    data = np.loadtxt(COMSOL_DATA_PATH, comments="%")
    print(f"Loaded COMSOL data from: {COMSOL_DATA_PATH}")

    n_nodes = data.shape[0]
    n_cols = data.shape[1]

    if (n_cols - 2) % 3 != 0:
        raise ValueError("Unexpected COMSOL file format. Expected 2 + 3*n_times columns.")

    n_times = (n_cols - 2) // 3
    t_values = T_INITIAL + DT_EXPORT * np.arange(n_times)

    expected_n_times = int(round((T_FINAL - T_INITIAL) / DT_EXPORT)) + 1

    if n_times != expected_n_times or not np.isclose(t_values[-1], T_FINAL):
        raise ValueError(
            "COMSOL time grid does not match the configured animation range: "
            f"expected {expected_n_times} snapshots from {T_INITIAL:.2f} to "
            f"{T_FINAL:.2f} s with DT_EXPORT = {DT_EXPORT:.2f} s, "
            f"but found {n_times} snapshots ending at {t_values[-1]:.2f} s."
        )

    x_dim = data[:, 0:1]
    y_dim = data[:, 1:2]

    fields = data[:, 2:].reshape(n_nodes, n_times, 3)
    p_comsol_wide = fields[:, :, 2]

    print(f"COMSOL nodes: {n_nodes}")
    print(f"Time snapshots: {n_times}")
    print(f"t range: {t_values[0]:.2f} to {t_values[-1]:.2f} s")

    return x_dim, y_dim, p_comsol_wide, t_values

# =========================
# Load trained Stage 3 PINN
# =========================

def load_stage3_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Stage 3 checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Run Stage 3 first with SAVE_CHECKPOINT = True."
        )

    model = PINN(layers).to(device)
    checkpoint = load_torch_checkpoint(CHECKPOINT_PATH, device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError(
            "Stage 3 checkpoint does not contain 'model_state_dict'. "
            "Generate it with the current NS_xy_stage3.py script."
        )

    if checkpoint.get("layers") != layers:
        raise ValueError(
            "Stage 3 checkpoint architecture does not match the animation model. "
            f"Checkpoint layers: {checkpoint.get('layers')}; current layers: {layers}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded Stage 3 checkpoint from: {CHECKPOINT_PATH}")

    return model

# =========================
# Predict PINN pressure for all nodes and time snapshots
# =========================

def predict_pinn_pressure_all_times(
    model,
    device,
    x_dim,
    y_dim,
    t_values,
    batch_size=100000
):
    n_nodes = x_dim.shape[0]
    n_times = len(t_values)

    x_long = np.repeat(x_dim, n_times, axis=1).reshape(-1, 1)
    y_long = np.repeat(y_dim, n_times, axis=1).reshape(-1, 1)
    t_long = np.tile(t_values, (n_nodes, 1)).reshape(-1, 1)

    x_star = x_long / x_ref
    y_star = y_long / y_ref
    t_star = t_long / t_ref

    p_pred_list = []

    with torch.no_grad():
        for start in range(0, x_star.shape[0], batch_size):
            end = start + batch_size

            x_t = torch.tensor(
                x_star[start:end],
                dtype=torch.float32,
                device=device
            )

            y_t = torch.tensor(
                y_star[start:end],
                dtype=torch.float32,
                device=device
            )

            t_t = torch.tensor(
                t_star[start:end],
                dtype=torch.float32,
                device=device
            )

            output = model(y_t, x_t, t_t)
            p_pred = output[:, 2:3] * p_ref

            p_pred_list.append(p_pred.detach().cpu().numpy())

    p_pinn_long = np.vstack(p_pred_list).reshape(-1)
    p_pinn_wide = p_pinn_long.reshape(n_nodes, n_times)

    return p_pinn_wide

# =========================
# Animation setup
# =========================

def setup_axis(ax):
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, H)
    ax.set_aspect("equal", adjustable="box")

# =========================
# Create PINN pressure GIF
# =========================

def create_pinn_pressure_gif(
    x_dim,
    y_dim,
    p_comsol_wide,
    p_pinn_wide,
    t_values
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stop_if_output_exists(OUTPUT_GIF_PATH)

    x_plot = x_dim.flatten()
    y_plot = y_dim.flatten()

    triang = tri.Triangulation(x_plot, y_plot)

    p_vmin = min(np.nanmin(p_comsol_wide), np.nanmin(p_pinn_wide))
    p_vmax = max(np.nanmax(p_comsol_wide), np.nanmax(p_pinn_wide))

    norm = Normalize(vmin=p_vmin, vmax=p_vmax)
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    fig.colorbar(sm, ax=ax, label="Pressure p (Pa)")

    def update(i):
        ax.clear()

        ax.tricontourf(
            triang,
            p_pinn_wide[:, i],
            levels=LEVELS,
            cmap=CMAP,
            vmin=p_vmin,
            vmax=p_vmax
        )

        setup_axis(ax)
        ax.set_title(f"PINN pressure field, t = {t_values[i]:.2f} s")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t_values),
        interval=1000 / FPS,
        repeat=True
    )

    writer = animation.PillowWriter(fps=FPS)

    ani.save(OUTPUT_GIF_PATH, writer=writer)
    plt.close(fig)

    print(f"Saved GIF: {OUTPUT_GIF_PATH}")

# =========================
# Main
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    x_dim, y_dim, p_comsol_wide, t_values = load_comsol_data()

    model = load_stage3_model(device)

    print("Predicting PINN pressure field...")
    p_pinn_wide = predict_pinn_pressure_all_times(
        model=model,
        device=device,
        x_dim=x_dim,
        y_dim=y_dim,
        t_values=t_values
    )
    print("PINN pressure prediction completed.")

    create_pinn_pressure_gif(
        x_dim=x_dim,
        y_dim=y_dim,
        p_comsol_wide=p_comsol_wide,
        p_pinn_wide=p_pinn_wide,
        t_values=t_values
    )

    print("Stage 3 pressure animation completed.")

if __name__ == "__main__":
    main()
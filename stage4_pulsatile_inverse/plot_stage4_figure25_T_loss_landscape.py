"""
Figure 25: Data-loss landscape with respect to period T.

This external script evaluates the pressure-data-loss landscape using the
final trained state of Stage 4 Experiment 9 (F2, transfer learning, all
61 pressure observations at x=2 m, y=2 m, progressive time windows).

The complete Experiment 9 checkpoint is loaded and then frozen. No Stage 4
training or re-optimization is performed inside this script. During the
diagnostic, only the period T is varied while the trained neural-network
weights, A0, and the known amplitude A1 remain fixed.

The pressure-data loss is evaluated for two observation windows:
- [0, 0.5] s,
- [0, 3.0] s.

The red vertical line marks the original inverse-problem initialization
T_initial = 0.8 s; it is not the T value stored in the trained checkpoint.

Paths are resolved automatically relative to the repository root.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# =========================
# Configuration
# =========================

SHOW_PLOT = False
OVERWRITE_EXISTING_OUTPUTS = False

# Original inverse-problem initialization.
# A0_FIXED and T_INITIAL are used to validate the Experiment 9 checkpoint;
# T_INITIAL is also used for the red vertical marker.
A0_FIXED = 8.0
A1_FIXED = 5.0
T_INITIAL = 0.8
T_TRUE = 0.5

# Pressure observation point
OBSERVATION_POINT_KEY = "x2_y2"
OBSERVATION_POINT = (2.0, 2.0)

# Observation windows compared in the figure
WINDOWS = [
    {
        "label": "Window [0,0.5] s",
        "t_window_max": 0.5,
        "color": "tab:blue",
        "linewidth": 2.0,
    },
    {
        "label": "Window [0,3] s",
        "t_window_max": 3.0,
        "color": "tab:orange",
        "linewidth": 2.0,
    },
]

# T-scan range
T_SCAN_MIN = 0.30
T_SCAN_MAX = 1.20
T_SCAN_STEP = 0.001

# Optional diagnostic range used only to report a secondary valley
LOCAL_VALLEY_MIN = 0.65
LOCAL_VALLEY_MAX = 0.90

# Physical parameters, matching NS_xy_stage4.py
MU = 1.0
RHO = 1.0

P_IN_MEAN = 10.0
P_IN_AMPLITUDE = 5.0
P_OUT_DIM = 0.0
P_PERIOD = 0.5

DT_EXPORT = 0.05
T_FINAL = 3.0

LENGTH = 12.0
HEIGHT = 4.0

# Network architecture, matching NS_xy_stage4.py
INPUT_DIM = 3
OUTPUT_DIM = 3
HIDDEN_LAYERS = 5
NEURONS = 64

# Figure appearance
FIGSIZE = (10, 6)

TITLE = r"Data-loss landscape with respect to $T$"
X_LABEL = r"$T$ (s)"
Y_LABEL = "Data loss"

TRUE_T_LINESTYLE = "--"
TRUE_T_COLOR = "black"
TRUE_T_LINEWIDTH = 1.3

INITIAL_T_LINESTYLE = ":"
INITIAL_T_COLOR = "red"
INITIAL_T_LINEWIDTH = 1.5

PLOT_XMIN = T_SCAN_MIN
PLOT_XMAX = T_SCAN_MAX


# =========================
# Paths
# =========================

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = REPO_ROOT / "models"

STAGE4_CHECKPOINT_PATH = (
    MODEL_DIR
    / (
        "stage4_F2_TL_adam700_300_300_300_lbfgs500_"
        "sp1_x2_y2_tm61_A0init8_Tinit0p8_model.pt"
    )
)

COMSOL_DATA_PATH = (
    REPO_ROOT
    / "data"
    / "comsol"
    / "stage3_stage4"
    / "NS_xy_pulsatile.txt"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "stage4_pulsatile_inverse"
    / "external_figures"
)

PNG_PATH = (
    OUTPUT_DIR
    / "figure25_T_loss_landscape.png"
)


# =========================
# Device
# =========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================
# Physical scales
# =========================

mu = MU
rho = RHO

p_in_dim = P_IN_MEAN
p_out_dim = P_OUT_DIM

deltaP = (
    p_in_dim
    - p_out_dim
)

A1_true = P_IN_AMPLITUDE
T_true = P_PERIOD

L = LENGTH
H = HEIGHT

x_ref = L
y_ref = H
t_ref = T_true

u_ref = (
    deltaP * H**2
    / (8.0 * mu * L)
)

p_ref = (
    rho * u_ref**2
)

layers = (
    [INPUT_DIM]
    + [NEURONS] * HIDDEN_LAYERS
    + [OUTPUT_DIM]
)


# =========================
# Utilities
# =========================

def stop_if_output_exists(path):
    path = Path(path)

    if (
        path.exists()
        and not OVERWRITE_EXISTING_OUTPUTS
    ):
        raise FileExistsError(
            f"Output already exists and would be overwritten:\n"
            f"{path}\n\n"
            f"Delete the existing output or set "
            f"OVERWRITE_EXISTING_OUTPUTS = True."
        )


def load_torch_checkpoint(path):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False
        )

    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


# =========================
# Minimal Stage 4 F2 model
# =========================

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation = nn.Tanh()

        for i in range(
            len(layers) - 1
        ):
            layer = nn.Linear(
                layers[i],
                layers[i + 1]
            )

            nn.init.xavier_normal_(
                layer.weight
            )

            nn.init.zeros_(
                layer.bias
            )

            self.layers.append(
                layer
            )

        self.A0 = nn.Parameter(
            torch.tensor(
                [A0_FIXED / p_ref],
                dtype=torch.float32
            ),
            requires_grad=False
        )

        self.A1_log = nn.Parameter(
            torch.log(
                torch.tensor(
                    [A1_FIXED / p_ref],
                    dtype=torch.float32
                )
            ),
            requires_grad=False
        )

        self.T_log = nn.Parameter(
            torch.log(
                torch.tensor(
                    [T_INITIAL / t_ref],
                    dtype=torch.float32
                )
            ),
            requires_grad=False
        )

    def A1_star(self):
        return torch.exp(
            self.A1_log
        )

    def T_star(self):
        return torch.exp(
            self.T_log
        )

    def forward(
        self,
        y,
        x,
        t
    ):
        inp = torch.cat(
            [y, x, t],
            dim=1
        )

        out = inp

        for i in range(
            len(self.layers) - 1
        ):
            out = self.activation(
                self.layers[i](out)
            )

        raw = self.layers[-1](out)

        # Hard no-slip wall boundary conditions
        phi = (
            y * (1.0 - y)
        )

        u_y = (
            raw[:, 0:1]
            * phi
        )

        u_x = (
            raw[:, 1:2]
            * phi
        )

        # Hard pressure boundary conditions
        x_bar = (
            x
            / (L / x_ref)
        )

        p_in_t = (
            self.A0
            + self.A1_star()
            * torch.sin(
                2.0
                * np.pi
                * t
                / self.T_star()
            )
        )

        p_out_star = torch.full_like(
            p_in_t,
            p_out_dim / p_ref
        )

        p = (
            (1.0 - x_bar)
            * p_in_t
            + x_bar
            * p_out_star
            + x_bar
            * (1.0 - x_bar)
            * raw[:, 2:3]
        )

        return torch.cat(
            [u_y, u_x, p],
            dim=1
        )


# =========================
# Frozen Experiment 9 checkpoint
# =========================

def load_stage4_experiment9_checkpoint(
    model
):
    if not STAGE4_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Experiment 9 checkpoint not found:\n"
            f"{STAGE4_CHECKPOINT_PATH}"
        )

    checkpoint = load_torch_checkpoint(
        STAGE4_CHECKPOINT_PATH
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Experiment 9 checkpoint must be a dictionary."
        )

    required_keys = [
        "model_state_dict",
        "layers",
        "EXPERIMENT_FAMILY",
        "USE_TRANSFER_LEARNING",
        "OBSERVATION_CASE",
        "SELECTED_OBSERVATION_POINTS",
        "n_obs_times",
        "infer_A1",
        "infer_T",
        "training_schedule",
        "run_lbfgs",
        "max_lbfgs_iterations",
        "A0_initial",
        "T_initial",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise KeyError(
            "Experiment 9 checkpoint is missing required keys: "
            f"{missing_keys}"
        )

    if checkpoint["layers"] != layers:
        raise ValueError(
            "Checkpoint architecture does not match Figure 25 model.\n"
            f"Checkpoint layers: {checkpoint['layers']}\n"
            f"Expected layers:   {layers}"
        )

    expected_values = {
        "EXPERIMENT_FAMILY": "F2",
        "USE_TRANSFER_LEARNING": True,
        "OBSERVATION_CASE": "all",
        "n_obs_times": 61,
        "infer_A1": False,
        "infer_T": True,
        "run_lbfgs": True,
        "max_lbfgs_iterations": 500,
    }

    for key, expected in expected_values.items():
        actual = checkpoint[key]

        if actual != expected:
            raise ValueError(
                "Wrong checkpoint for Figure 25.\n"
                f"{key}: expected {expected!r}, found {actual!r}."
            )

    if checkpoint["SELECTED_OBSERVATION_POINTS"] != [
        OBSERVATION_POINT_KEY
    ]:
        raise ValueError(
            "Figure 25 requires observation point "
            f"{OBSERVATION_POINT_KEY}."
        )

    training_schedule = [
        (float(window), int(epochs))
        for window, epochs in checkpoint["training_schedule"]
    ]

    expected_schedule = [
        (0.5, 700),
        (1.0, 300),
        (2.0, 300),
        (3.0, 300),
    ]

    if training_schedule != expected_schedule:
        raise ValueError(
            "Wrong Experiment 9 training schedule.\n"
            f"Expected: {expected_schedule}\n"
            f"Found:    {training_schedule}"
        )

    if not np.isclose(
        float(checkpoint["A0_initial"]),
        A0_FIXED
    ):
        raise ValueError(
            f"Expected A0_initial = {A0_FIXED:g} Pa."
        )

    if not np.isclose(
        float(checkpoint["T_initial"]),
        T_INITIAL
    ):
        raise ValueError(
            f"Expected T_initial = {T_INITIAL:g} s."
        )

    # Load the COMPLETE trained Stage 4 state:
    # network weights + trained A0 + trained T.
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    # Freeze everything. During Figure 25 only T will be
    # temporarily changed by the diagnostic sweep.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model.eval()

    A0_loaded = float(
        model.A0.detach().cpu().item()
        * p_ref
    )

    A1_loaded = float(
        model.A1_star().detach().cpu().item()
        * p_ref
    )

    T_loaded = float(
        model.T_star().detach().cpu().item()
        * t_ref
    )

    print("=" * 70)
    print("Frozen Stage 4 Experiment 9 checkpoint loaded")
    print(
        f"Checkpoint         : "
        f"{STAGE4_CHECKPOINT_PATH.name}"
    )
    print(
        f"Observation point  : "
        f"{OBSERVATION_POINT_KEY}"
    )
    print(
        f"Trained A0 fixed   : "
        f"{A0_loaded:.6f} Pa"
    )
    print(
        f"Known A1 fixed     : "
        f"{A1_loaded:.6f} Pa"
    )
    print(
        f"Trained T stored   : "
        f"{T_loaded:.6f} s"
    )
    print(
        f"Initial T marker   : "
        f"{T_INITIAL:.6f} s"
    )
    print(
        f"True T             : "
        f"{T_TRUE:.6f} s"
    )
    print("=" * 70)


# =========================
# COMSOL loading
# =========================

def load_comsol_reference():
    if not COMSOL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"COMSOL data file not found:\n"
            f"{COMSOL_DATA_PATH}"
        )

    data = np.loadtxt(
        COMSOL_DATA_PATH,
        comments="%"
    )

    n_nodes = data.shape[0]
    n_cols = data.shape[1]

    if (
        (n_cols - 2) % 3
        != 0
    ):
        raise ValueError(
            "Unexpected COMSOL column count: "
            "expected 2 + 3*n_times."
        )

    n_times = (
        (n_cols - 2)
        // 3
    )

    t_values = (
        DT_EXPORT
        * np.arange(
            n_times
        )
    )

    expected_n_times = (
        int(
            round(
                T_FINAL
                / DT_EXPORT
            )
        )
        + 1
    )

    if (
        n_times
        != expected_n_times
    ):
        raise ValueError(
            f"Expected {expected_n_times} "
            f"COMSOL time snapshots, "
            f"found {n_times}."
        )

    if not np.isclose(
        t_values[-1],
        T_FINAL
    ):
        raise ValueError(
            f"Expected COMSOL data through "
            f"t = {T_FINAL:g} s, "
            f"found t = "
            f"{t_values[-1]:.6f} s."
        )

    x_dim = (
        data[:, 0:1]
    )

    y_dim = (
        data[:, 1:2]
    )

    fields = data[
        :,
        2:
    ].reshape(
        n_nodes,
        n_times,
        3
    )

    p_wide = (
        fields[:, :, 2]
    )

    print("=" * 70)
    print(
        "COMSOL reference data loaded"
    )
    print(
        f"File           : "
        f"{COMSOL_DATA_PATH.name}"
    )
    print(
        f"Nodes          : "
        f"{n_nodes}"
    )
    print(
        f"Time snapshots : "
        f"{n_times}"
    )
    print(
        f"Time range     : "
        f"[{t_values[0]:.3f}, "
        f"{t_values[-1]:.3f}] s"
    )
    print("=" * 70)

    return {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "p_wide": p_wide,
        "t_values": t_values,
    }


# =========================
# Observation windows
# =========================

def build_observation_window(
    reference_data,
    t_window_max
):
    x_dim = (
        reference_data[
            "x_dim"
        ]
    )

    y_dim = (
        reference_data[
            "y_dim"
        ]
    )

    p_wide = (
        reference_data[
            "p_wide"
        ]
    )

    t_values = (
        reference_data[
            "t_values"
        ]
    )

    xp, yp = (
        OBSERVATION_POINT
    )

    if not (
        0.0 <= xp <= L
        and 0.0 <= yp <= H
    ):
        raise ValueError(
            f"Observation point "
            f"({xp}, {yp}) "
            f"lies outside the domain."
        )

    t_obs_unique_phys = (
        t_values[
            t_values
            <= t_window_max
            + 1e-12
        ]
        .reshape(-1, 1)
    )

    if (
        len(
            t_obs_unique_phys
        )
        == 0
    ):
        raise ValueError(
            f"No observation times found "
            f"for t_window_max="
            f"{t_window_max}."
        )

    spatial_points = (
        np.hstack(
            [x_dim, y_dim]
        )
    )

    x_obs_phys_list = []
    y_obs_phys_list = []
    t_obs_phys_list = []
    p_obs_phys_list = []

    for t_phys in (
        t_obs_unique_phys
        .flatten()
    ):
        t_idx = int(
            np.argmin(
                np.abs(
                    t_values
                    - t_phys
                )
            )
        )

        p_val = griddata(
            spatial_points,
            p_wide[:, t_idx],
            np.array(
                [[xp, yp]]
            ),
            method="linear"
        )[0]

        if np.isnan(
            p_val
        ):
            raise ValueError(
                f"griddata returned NaN "
                f"at "
                f"(x={xp}, "
                f"y={yp}, "
                f"t={t_phys})."
            )

        x_obs_phys_list.append(
            xp
        )

        y_obs_phys_list.append(
            yp
        )

        t_obs_phys_list.append(
            t_phys
        )

        p_obs_phys_list.append(
            p_val
        )

    x_obs_phys = (
        np.asarray(
            x_obs_phys_list
        )
        .reshape(-1, 1)
    )

    y_obs_phys = (
        np.asarray(
            y_obs_phys_list
        )
        .reshape(-1, 1)
    )

    t_obs_phys = (
        np.asarray(
            t_obs_phys_list
        )
        .reshape(-1, 1)
    )

    p_obs_phys = (
        np.asarray(
            p_obs_phys_list
        )
        .reshape(-1, 1)
    )

    x_obs = torch.tensor(
        x_obs_phys / x_ref,
        dtype=torch.float32,
        device=device
    )

    y_obs = torch.tensor(
        y_obs_phys / y_ref,
        dtype=torch.float32,
        device=device
    )

    t_obs = torch.tensor(
        t_obs_phys / t_ref,
        dtype=torch.float32,
        device=device
    )

    p_obs = torch.tensor(
        p_obs_phys / p_ref,
        dtype=torch.float32,
        device=device
    )

    print(
        f"Observation window "
        f"[0,{t_window_max:g}] s: "
        f"{len(t_obs_unique_phys)} "
        f"active time points"
    )

    return {
        "x_obs": x_obs,
        "y_obs": y_obs,
        "t_obs": t_obs,
        "p_obs": p_obs,
        "n_obs_times": len(
            t_obs_unique_phys
        ),
    }


# =========================
# Data-loss scan
# =========================

def evaluate_data_loss(
    model,
    T_candidate_phys,
    obs_window
):
    if (
        T_candidate_phys
        <= 0.0
    ):
        raise ValueError(
            "T scan values must "
            "be positive."
        )

    with torch.no_grad():
        model.T_log.copy_(
            torch.log(
                torch.tensor(
                    [
                        T_candidate_phys
                        / t_ref
                    ],
                    dtype=torch.float32,
                    device=device
                )
            )
        )

        output_obs = model(
            obs_window[
                "y_obs"
            ],
            obs_window[
                "x_obs"
            ],
            obs_window[
                "t_obs"
            ]
        )

        p_data_pred = (
            output_obs[
                :,
                2:3
            ]
        )

        loss_data = torch.mean(
            (
                p_data_pred
                - obs_window[
                    "p_obs"
                ]
            )**2
        )

    return float(
        loss_data.item()
    )


def scan_data_loss_over_T(
    model,
    obs_window,
    T_scan_phys
):
    original_T_log = (
        model.T_log
        .detach()
        .clone()
    )

    losses = []

    try:
        for T_candidate in (
            T_scan_phys
        ):
            loss_value = (
                evaluate_data_loss(
                    model=model,
                    T_candidate_phys=float(
                        T_candidate
                    ),
                    obs_window=obs_window
                )
            )

            losses.append(
                loss_value
            )

    finally:
        with torch.no_grad():
            model.T_log.copy_(
                original_T_log
            )

    return np.asarray(
        losses,
        dtype=float
    )


def value_at_T(
    T_scan_phys,
    losses,
    T_value
):
    idx = int(
        np.argmin(
            np.abs(
                T_scan_phys
                - T_value
            )
        )
    )

    if not np.isclose(
        T_scan_phys[idx],
        T_value,
        atol=1e-12,
        rtol=0.0
    ):
        raise ValueError(
            f"T = {T_value:g} s "
            f"is not contained exactly "
            f"in the T-scan grid."
        )

    return float(
        losses[idx]
    )


def report_landscape(
    label,
    T_scan_phys,
    losses,
    n_obs_times
):
    min_idx = int(
        np.argmin(
            losses
        )
    )

    true_loss = value_at_T(
        T_scan_phys,
        losses,
        T_TRUE
    )

    initial_loss = value_at_T(
        T_scan_phys,
        losses,
        T_INITIAL
    )

    print("=" * 70)
    print(label)

    print(
        f"Observation times : "
        f"{n_obs_times}"
    )

    print(
        f"Best T from sweep : "
        f"{T_scan_phys[min_idx]:.6f} s"
    )

    print(
        f"Minimum data loss : "
        f"{losses[min_idx]:.6e}"
    )

    print(
        f"Loss at true T    : "
        f"{true_loss:.6e}"
    )

    print(
        f"Loss at initial T : "
        f"{initial_loss:.6e}"
    )

    valley_mask = (
        (
            T_scan_phys
            >= LOCAL_VALLEY_MIN
        )
        & (
            T_scan_phys
            <= LOCAL_VALLEY_MAX
        )
    )

    if np.any(
        valley_mask
    ):
        valley_T = (
            T_scan_phys[
                valley_mask
            ]
        )

        valley_loss = (
            losses[
                valley_mask
            ]
        )

        valley_idx = int(
            np.argmin(
                valley_loss
            )
        )

        print(
            f"Best T in "
            f"[{LOCAL_VALLEY_MIN:g},"
            f"{LOCAL_VALLEY_MAX:g}] s: "
            f"{valley_T[valley_idx]:.6f} s"
        )

        print(
            f"Local-range loss  : "
            f"{valley_loss[valley_idx]:.6e}"
        )

    print("=" * 70)


# =========================
# Plot
# =========================

def main():
    if not np.isclose(
        A1_FIXED,
        A1_true
    ):
        raise ValueError(
            "A1_FIXED must match "
            "the known F2 amplitude."
        )

    if not np.isclose(
        T_TRUE,
        T_true
    ):
        raise ValueError(
            "T_TRUE must match "
            "the physical reference period."
        )

    if (
        T_SCAN_STEP
        <= 0.0
    ):
        raise ValueError(
            "T_SCAN_STEP must "
            "be positive."
        )

    model = PINN(
        layers
    ).to(
        device
    )

    load_stage4_experiment9_checkpoint(
        model
    )

    reference_data = (
        load_comsol_reference()
    )

    T_scan_phys = np.arange(
        T_SCAN_MIN,
        T_SCAN_MAX
        + 0.5 * T_SCAN_STEP,
        T_SCAN_STEP,
        dtype=float
    )

    for required_T in [
        T_TRUE,
        T_INITIAL
    ]:
        if not np.any(
            np.isclose(
                T_scan_phys,
                required_T,
                atol=1e-12,
                rtol=0.0
            )
        ):
            raise ValueError(
                f"The T-scan grid must "
                f"contain T = "
                f"{required_T:g} s "
                f"exactly."
            )

    window_results = []

    for window in WINDOWS:
        obs_window = (
            build_observation_window(
                reference_data=reference_data,
                t_window_max=window[
                    "t_window_max"
                ]
            )
        )

        losses = (
            scan_data_loss_over_T(
                model=model,
                obs_window=obs_window,
                T_scan_phys=T_scan_phys
            )
        )

        report_landscape(
            label=window[
                "label"
            ],
            T_scan_phys=T_scan_phys,
            losses=losses,
            n_obs_times=obs_window[
                "n_obs_times"
            ]
        )

        window_results.append({
            "label": window[
                "label"
            ],
            "color": window[
                "color"
            ],
            "linewidth": window[
                "linewidth"
            ],
            "losses": losses,
        })

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    for result in (
        window_results
    ):
        ax.plot(
            T_scan_phys,
            result[
                "losses"
            ],
            color=result[
                "color"
            ],
            linewidth=result[
                "linewidth"
            ],
            label=result[
                "label"
            ]
        )

    ax.axvline(
        T_TRUE,
        color=TRUE_T_COLOR,
        linestyle=TRUE_T_LINESTYLE,
        linewidth=TRUE_T_LINEWIDTH,
        label="True T"
    )

    ax.axvline(
        T_INITIAL,
        color=INITIAL_T_COLOR,
        linestyle=INITIAL_T_LINESTYLE,
        linewidth=INITIAL_T_LINEWIDTH,
        label="Initial T"
    )

    ax.set_title(
        TITLE
    )

    ax.set_xlabel(
        X_LABEL
    )

    ax.set_ylabel(
        Y_LABEL
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        PLOT_XMIN,
        PLOT_XMAX
    )

    ax.grid(
        True,
        which="both",
        alpha=0.30
    )

    ax.legend(
        loc="lower left",
        framealpha=0.95
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stop_if_output_exists(
        PNG_PATH
    )

    fig.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved figure: "
        f"{PNG_PATH}"
    )

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(
            fig
        )


if __name__ == "__main__":
    main()
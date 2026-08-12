from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# Configuration
# =========================

SHOW_PLOT = False
OVERWRITE_EXISTING_OUTPUTS = False


# =========================
# Paths
# =========================

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = REPO_ROOT / "models"
OUTPUT_DIR = (
    REPO_ROOT
    / "stage4_pulsatile_inverse"
    / "external_figures"
)

PNG_PATH = OUTPUT_DIR / "figure27_A1_TL_observations.png"


# =========================
# Experiment checkpoints
# =========================

TRAINING_SCHEDULE = [
    (0.5, 1000),
    (1.0, 500),
    (2.0, 500),
    (3.0, 500),
]

WITHOUT_TL_STYLE = (0, (6, 2, 1.2, 2))

RUNS = [
    {
        "experiment": 20,
        "observation_case": "all",
        "n_obs_times": 61,
        "use_transfer_learning": True,
        "filename": (
            "stage4_F3_TL_adam1000_500_500_500_lbfgs1000_"
            "sp1_x2_y2_tm61_A0init8_A1init3_Tinit0p8_model.pt"
        ),
        "color": "tab:blue",
        "linestyle": "-",
    },
    {
        "experiment": 23,
        "observation_case": "all",
        "n_obs_times": 61,
        "use_transfer_learning": False,
        "filename": (
            "stage4_F3_noTL_adam1000_500_500_500_lbfgs1000_"
            "sp1_x2_y2_tm61_A0init8_A1init3_Tinit0p8_model.pt"
        ),
        "color": "tab:blue",
        "linestyle": WITHOUT_TL_STYLE,
    },
    {
        "experiment": 22,
        "observation_case": "five",
        "n_obs_times": 5,
        "use_transfer_learning": True,
        "filename": (
            "stage4_F3_TL_adam1000_500_500_500_lbfgs1000_"
            "sp1_x2_y2_tm5_A0init8_A1init3_Tinit0p8_model.pt"
        ),
        "color": "tab:orange",
        "linestyle": "-",
    },
    {
        "experiment": 24,
        "observation_case": "five",
        "n_obs_times": 5,
        "use_transfer_learning": False,
        "filename": (
            "stage4_F3_noTL_adam1000_500_500_500_lbfgs1000_"
            "sp1_x2_y2_tm5_A0init8_A1init3_Tinit0p8_model.pt"
        ),
        "color": "tab:orange",
        "linestyle": WITHOUT_TL_STYLE,
    },
]


# =========================
# Utilities
# =========================

def stop_if_output_exists(path):
    path = Path(path)

    if path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
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
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


def normalize_training_schedule(schedule):
    return [
        (float(window), int(epochs))
        for window, epochs in schedule
    ]


def validate_checkpoint(checkpoint, run, path):
    required_keys = [
        "A1_history",
        "A1_history_lbfgs",
        "A1_true",
        "A1_initial",
        "A0_initial",
        "T_initial",
        "training_schedule",
        "max_lbfgs_iterations",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"Required key '{key}' not found in checkpoint:\n"
                f"{path}"
            )

    expected_values = {
        "EXPERIMENT_FAMILY": "F3",
        "USE_TRANSFER_LEARNING": run["use_transfer_learning"],
        "TRAINING_PROFILE": "curriculum_hard",
        "OBSERVATION_CASE": run["observation_case"],
        "n_obs_times": run["n_obs_times"],
        "infer_A1": True,
        "infer_T": True,
        "run_lbfgs": True,
        "max_lbfgs_iterations": 1000,
    }

    for key, expected in expected_values.items():
        actual = checkpoint.get(key)

        if actual != expected:
            raise ValueError(
                f"Checkpoint validation failed for Experiment "
                f"{run['experiment']}.\n"
                f"Key: {key}\n"
                f"Expected: {expected}\n"
                f"Found: {actual}\n"
                f"Checkpoint: {path}"
            )

    selected_points = checkpoint.get(
        "SELECTED_OBSERVATION_POINTS"
    )

    if selected_points != ["x2_y2"]:
        raise ValueError(
            f"Experiment {run['experiment']} must use the "
            f"x2_y2 observation point.\n"
            f"Found: {selected_points}"
        )

    initial_values = {
        "A0_initial": (8.0, "Pa"),
        "A1_initial": (3.0, "Pa"),
        "T_initial": (0.8, "s"),
    }

    for key, (expected, unit) in initial_values.items():
        actual = float(checkpoint[key])

        if not np.isclose(actual, expected):
            raise ValueError(
                f"Experiment {run['experiment']} must use "
                f"{key} = {expected:g} {unit}.\n"
                f"Found: {actual:g} {unit}"
            )

    actual_schedule = normalize_training_schedule(
        checkpoint["training_schedule"]
    )

    expected_schedule = normalize_training_schedule(
        TRAINING_SCHEDULE
    )

    if actual_schedule != expected_schedule:
        raise ValueError(
            f"Checkpoint validation failed for Experiment "
            f"{run['experiment']}.\n"
            f"Expected training schedule: {expected_schedule}\n"
            f"Found: {actual_schedule}\n"
            f"Checkpoint: {path}"
        )


def load_A1_evolution(run):
    path = MODEL_DIR / run["filename"]

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    checkpoint = load_torch_checkpoint(path)

    validate_checkpoint(
        checkpoint=checkpoint,
        run=run,
        path=path
    )

    A1_adam = np.asarray(
        checkpoint["A1_history"],
        dtype=float
    ).flatten()

    A1_lbfgs = np.asarray(
        checkpoint["A1_history_lbfgs"],
        dtype=float
    ).flatten()

    if len(A1_adam) == 0:
        raise ValueError(
            f"A1_history is empty in checkpoint:\n{path}"
        )

    A1_total = np.concatenate([
        A1_adam,
        A1_lbfgs
    ])

    steps = np.arange(
        1,
        len(A1_total) + 1
    )

    A1_true = float(checkpoint["A1_true"])

    print("=" * 70)
    print(f"Experiment {run['experiment']}")
    print(f"Checkpoint       : {path.name}")
    print(
        f"Transfer learning: "
        f"{run['use_transfer_learning']}"
    )
    print(f"Observations     : {run['n_obs_times']}")
    print(f"Adam points      : {len(A1_adam)}")
    print(f"L-BFGS points    : {len(A1_lbfgs)}")
    print(
        f"A1 initial       : "
        f"{float(checkpoint['A1_initial']):.6f} Pa"
    )
    print(f"A1 after Adam    : {A1_adam[-1]:.6f} Pa")
    print(f"A1 final         : {A1_total[-1]:.6f} Pa")
    print("=" * 70)

    return {
        "steps": steps,
        "A1_total": A1_total,
        "n_adam": len(A1_adam),
        "A1_true": A1_true,
    }


# =========================
# Plot
# =========================

def main():
    loaded_runs = [
        (run, load_A1_evolution(run))
        for run in RUNS
    ]

    A1_true_values = np.array([
        result["A1_true"]
        for _, result in loaded_runs
    ])

    if not np.allclose(
        A1_true_values,
        A1_true_values[0]
    ):
        raise ValueError(
            "The checkpoints do not use the same true A1 value."
        )

    transition_steps = {
        result["n_adam"]
        for _, result in loaded_runs
    }

    if len(transition_steps) != 1:
        raise ValueError(
            "The checkpoints do not use the same number of Adam steps."
        )

    stage_epochs = [
        epochs for _, epochs in TRAINING_SCHEDULE
    ]

    curriculum_boundaries = list(
        np.cumsum(stage_epochs)[:-1]
    )

    adam_end_step = next(
        iter(transition_steps)
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for run, result in loaded_runs:
        ax.plot(
            result["steps"],
            result["A1_total"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=2.2,
            zorder=3
        )

    A1_true = A1_true_values[0]

    ax.axhline(
        A1_true,
        color="black",
        linestyle="--",
        linewidth=1.8,
        zorder=2
    )

    for step in curriculum_boundaries:
        ax.axvline(
            step,
            color="0.55",
            linestyle="--",
            linewidth=1.5,
            alpha=0.90,
            zorder=1
        )

    ax.axvline(
        adam_end_step,
        color="black",
        linestyle=":",
        linewidth=2.0,
        zorder=1
    )

    ax.set_title(
        r"Convergence of amplitude $A_1$",
        fontsize=18
    )

    ax.set_xlabel(
        "Training step",
        fontsize=14
    )

    ax.set_ylabel(
        r"$A_1$ (Pa)",
        fontsize=14
    )

    max_step = max(
        result["steps"][-1]
        for _, result in loaded_runs
    )

    ax.set_xlim(0, max_step)
    ax.set_ylim(2.5, 5.2)

    ax.grid(
        True,
        which="major",
        alpha=0.30
    )

    legend_handles_main = [
        Line2D(
            [0], [0],
            color="tab:blue",
            linewidth=2.5,
            linestyle="-",
            label=r"$N_d = 61$"
        ),
        Line2D(
            [0], [0],
            color="tab:orange",
            linewidth=2.5,
            linestyle="-",
            label=r"$N_d = 5$"
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=2.5,
            linestyle="-",
            label="With transfer learning"
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=2.5,
            linestyle=WITHOUT_TL_STYLE,
            label="Without transfer learning"
        ),
    ]

    legend_handles_aux = [
        Line2D(
            [0], [0],
            color="black",
            linewidth=1.8,
            linestyle="--",
            label=rf"$A_1 = {A1_true:g}$ Pa"
        ),
        Line2D(
            [0], [0],
            color="0.55",
            linewidth=1.5,
            linestyle="--",
            label="Curriculum window boundary"
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=2.0,
            linestyle=":",
            label="Adam → L-BFGS"
        ),
    ]

    leg1 = fig.legend(
        handles=legend_handles_main,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.06),
        handlelength=3.8
    )

    fig.legend(
        handles=legend_handles_aux,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=3.8
    )

    fig.add_artist(leg1)

    plt.tight_layout(
        rect=[0, 0.12, 1, 1]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stop_if_output_exists(PNG_PATH)

    fig.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved figure: {PNG_PATH}")


    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
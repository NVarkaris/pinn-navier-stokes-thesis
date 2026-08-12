from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


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

PNG_PATH = OUTPUT_DIR / "figure23_T_evolution.png"


# =========================
# Experiment checkpoints
# =========================

RUNS = [
    {
        "experiment": 8,
        "label": "Single time window",
        "training_schedule": [(3.0, 1600)],
        "filename": (
            "stage4_F2_TL_adam1600_lbfgs500_"
            "sp1_x2_y2_tm61_A0init8_Tinit0p8_model.pt"
        ),
        "color": "tab:blue",
        "linestyle": "-",
        "zorder": 3,
    },
    {
        "experiment": 9,
        "label": "Time-window curriculum",
        "training_schedule": [
            (0.5, 700),
            (1.0, 300),
            (2.0, 300),
            (3.0, 300),
        ],
        "filename": (
            "stage4_F2_TL_adam700_300_300_300_lbfgs500_"
            "sp1_x2_y2_tm61_A0init8_Tinit0p8_model.pt"
        ),
        "color": "tab:red",
        "linestyle": "--",
        "zorder": 4,
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
        "T_history",
        "T_history_lbfgs",
        "T_true",
        "T_initial",
        "A0_initial",
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
        "EXPERIMENT_FAMILY": "F2",
        "USE_TRANSFER_LEARNING": True,
        "TRAINING_PROFILE": "custom",
        "OBSERVATION_CASE": "all",
        "n_obs_times": 61,
        "infer_A1": False,
        "infer_T": True,
        "run_lbfgs": True,
        "max_lbfgs_iterations": 500,
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

    if not np.isclose(
        float(checkpoint["T_initial"]),
        0.8
    ):
        raise ValueError(
            f"Experiment {run['experiment']} must use "
            f"T_initial = 0.8 s."
        )

    if not np.isclose(
        float(checkpoint["A0_initial"]),
        8.0
    ):
        raise ValueError(
            f"Experiment {run['experiment']} must use "
            f"A0_initial = 8 Pa."
        )

    actual_schedule = normalize_training_schedule(
        checkpoint["training_schedule"]
    )

    expected_schedule = normalize_training_schedule(
        run["training_schedule"]
    )

    if actual_schedule != expected_schedule:
        raise ValueError(
            f"Checkpoint validation failed for Experiment "
            f"{run['experiment']}.\n"
            f"Expected training schedule: {expected_schedule}\n"
            f"Found: {actual_schedule}\n"
            f"Checkpoint: {path}"
        )


def load_T_evolution(run):
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

    T_adam = np.asarray(
        checkpoint["T_history"],
        dtype=float
    ).flatten()

    T_lbfgs = np.asarray(
        checkpoint["T_history_lbfgs"],
        dtype=float
    ).flatten()

    if len(T_adam) == 0:
        raise ValueError(
            f"T_history is empty in checkpoint:\n{path}"
        )

    T_total = np.concatenate([
        T_adam,
        T_lbfgs
    ])

    steps = np.arange(
        1,
        len(T_total) + 1
    )

    T_true = float(checkpoint["T_true"])
    T_initial = float(checkpoint["T_initial"])

    schedule = normalize_training_schedule(
        checkpoint["training_schedule"]
    )

    stage_epochs = [
        epochs for _, epochs in schedule
    ]

    curriculum_boundaries = list(
        np.cumsum(stage_epochs)[:-1]
    )

    print("=" * 70)
    print(f"Experiment {run['experiment']}")
    print(f"Checkpoint       : {path.name}")
    print(f"Training schedule: {schedule}")
    print(f"Adam points      : {len(T_adam)}")
    print(f"L-BFGS points    : {len(T_lbfgs)}")
    print(f"T initial        : {T_initial:.6f} s")
    print(f"T after Adam     : {T_adam[-1]:.6f} s")
    print(f"T final          : {T_total[-1]:.6f} s")
    print("=" * 70)

    return {
        "steps": steps,
        "T_total": T_total,
        "n_adam": len(T_adam),
        "T_true": T_true,
        "curriculum_boundaries": curriculum_boundaries,
    }


# =========================
# Plot
# =========================

def main():
    loaded_runs = [
        (run, load_T_evolution(run))
        for run in RUNS
    ]

    T_true_values = np.array([
        result["T_true"]
        for _, result in loaded_runs
    ])

    if not np.allclose(
        T_true_values,
        T_true_values[0]
    ):
        raise ValueError(
            "The checkpoints do not use the same true T value."
        )

    transition_steps = {
        result["n_adam"]
        for _, result in loaded_runs
    }

    curriculum_result = next(
        result
        for run, result in loaded_runs
        if run["experiment"] == 9
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    for run, result in loaded_runs:
        ax.plot(
            result["steps"],
            result["T_total"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=2.0,
            zorder=run["zorder"],
            label=run["label"]
        )

    T_true = T_true_values[0]

    ax.axhline(
        T_true,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"True $T = {T_true:g}$ s"
    )

    for i, step in enumerate(
        curriculum_result["curriculum_boundaries"]
    ):
        ax.axvline(
            step,
            color="0.45",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.75,
            label=(
                "Curriculum window boundary"
                if i == 0
                else None
            )
        )

    for i, step in enumerate(
        sorted(transition_steps)
    ):
        ax.axvline(
            step,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=(
                "Adam → L-BFGS"
                if i == 0
                else None
            )
        )

    ax.set_title(
        r"Convergence of inlet-pressure period $T$",
        fontsize=15
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$T$ (s)")

    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

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
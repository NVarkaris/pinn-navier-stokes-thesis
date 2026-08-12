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

PNG_PATH = OUTPUT_DIR / "figure22_A1_evolution.png"


# =========================
# Experiment checkpoints
# =========================

RUNS = [
    {
        "experiment": 1,
        "label": "61 observations",
        "observation_case": "all",
        "n_obs_times": 61,
        "filename": (
            "stage4_F1_TL_adam300_lbfgs500_"
            "sp1_x2_y2_tm61_A0init8_A1init3_model.pt"
        ),
        "color": "tab:blue",
        "linestyle": "-.",
        "zorder": 4,
    },
    {
        "experiment": 2,
        "label": "30 observations",
        "observation_case": "half",
        "n_obs_times": 30,
        "filename": (
            "stage4_F1_TL_adam300_lbfgs500_"
            "sp1_x2_y2_tm30_A0init8_A1init3_model.pt"
        ),
        "color": "tab:orange",
        "linestyle": "--",
        "zorder": 3,
    },
    {
        "experiment": 3,
        "label": "5 observations",
        "observation_case": "five",
        "n_obs_times": 5,
        "filename": (
            "stage4_F1_TL_adam300_lbfgs500_"
            "sp1_x2_y2_tm5_A0init8_A1init3_model.pt"
        ),
        "color": "tab:green",
        "linestyle": "-",
        "zorder": 2,
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
        "EXPERIMENT_FAMILY": "F1",
        "USE_TRANSFER_LEARNING": True,
        "TRAINING_PROFILE": "single_fast",
        "OBSERVATION_CASE": run["observation_case"],
        "n_obs_times": run["n_obs_times"],
        "infer_A1": True,
        "infer_T": False,
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
        float(checkpoint["A1_initial"]),
        3.0
    ):
        raise ValueError(
            f"Experiment {run['experiment']} must use "
            f"A1_initial = 3 Pa."
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

    expected_schedule = [(3.0, 300)]

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
    A1_initial = float(checkpoint["A1_initial"])

    print("=" * 70)
    print(f"Experiment {run['experiment']}")
    print(f"Checkpoint       : {path.name}")
    print(f"Adam points      : {len(A1_adam)}")
    print(f"L-BFGS points    : {len(A1_lbfgs)}")
    print(f"A1 initial       : {A1_initial:.6f} Pa")
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

    fig, ax = plt.subplots(figsize=(11, 6))

    for run, result in loaded_runs:
        ax.plot(
            result["steps"],
            result["A1_total"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=2.0,
            zorder=run["zorder"],
            label=run["label"]
        )

    A1_true = A1_true_values[0]

    ax.axhline(
        A1_true,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"True $A_1 = {A1_true:g}$ Pa"
    )

    for i, step in enumerate(
        sorted(transition_steps)
    ):
        ax.axvline(
            step,
            color="black",
            linestyle=":",
            linewidth=1.4,
            label=(
                "Adam → L-BFGS"
                if i == 0
                else None
            )
        )

    ax.set_title(
        r"Convergence of inlet-pressure amplitude $A_1$",
        fontsize=15
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$A_1$ (Pa)")

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
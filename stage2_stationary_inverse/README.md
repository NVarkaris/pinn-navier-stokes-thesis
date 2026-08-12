# Stage 2 - Stationary Inverse Problem

This folder contains the implementation of the second stage of the thesis work.

The problem is a stationary inverse Physics-Informed Neural Network (PINN) for incompressible Navier-Stokes flow in a two-dimensional rectangular channel.

The model is trained using physics-informed losses together with sparse pressure observations extracted from the COMSOL reference solution.

The inverse problem estimates the inlet pressure `p_in`.

COMSOL data are used for extracting the sparse pressure observations, evaluation, and plotting. They are not used as full-field supervised training data.

## Problem description

The neural network receives the non-dimensional spatial coordinates as input and predicts the non-dimensional velocity and pressure fields:

`(y*, x*) -> (u_y*, u_x*, p*)`

The loss function includes the continuity equation, the two Navier-Stokes momentum equations, the boundary conditions at the inlet, outlet, upper wall, and lower wall, and an additional sparse pressure-data loss.

Compared with Stage 1, the inlet pressure is represented by an additional trainable model parameter. Its initial value is controlled by `P_IN_INITIAL` and it is optimized together with the neural-network parameters.

The reference inlet pressure is `P_IN_DIM = 10 Pa`.

## Default setup

| Parameter                     | Value                 |
| ----------------------------- | --------------------- |
| Network inputs                | 2                     |
| Hidden layers                 | 5                     |
| Neurons per hidden layer      | 64                    |
| Network outputs               | 3                     |
| Activation function           | tanh                  |
| Weight initialization         | Xavier normal         |
| Formulation                   | Non-dimensional       |
| Domain collocation points     | 2000                  |
| Upper-wall points             | 500                   |
| Lower-wall points             | 500                   |
| Inlet points                  | 200                   |
| Outlet points                 | 200                   |
| Collocation sampling          | Random uniform        |
| Pressure observations         | 5                     |
| Initial inlet-pressure guess  | 8 Pa                  |
| Loss weights                  | Fixed, all equal to 1 |
| Adam epochs                   | 5000                  |
| Adam learning rate            | 1e-3                  |
| L-BFGS maximum iterations     | 300                   |
| L-BFGS learning rate          | 1e-1                  |

Training is performed sequentially using Adam followed by L-BFGS.

The trainable inlet pressure is optimized by the same optimizers and learning rate as the neural-network parameters.

The initial network state is evaluated before the first Adam update and is stored as epoch 0 in the training history.

## Pressure observation scenarios

All pressure observations are located on the channel centerline at `y = H/2`.

The available observation scenarios are:

| `DATA_POINTS` | x positions (m) |
| ------------- | --------------- |
| 5             | 2, 4, 6, 8, 10  |
| 4             | 2, 4, 8, 10     |
| 3             | 2, 4, 8         |
| 2             | 4, 8            |

The corresponding pressure values are extracted from the COMSOL reference solution.

## Required data

The COMSOL reference solution must be located at:

`data/comsol/stage1_stage2/NS_xy_stationary.txt`

Paths are resolved automatically relative to the repository root.

## Scripts

Stage 2 contains two Python scripts:

`stage2_stationary_inverse/NS_xy_stage2.py`

`stage2_stationary_inverse/plot_stage2_figures_13_14.py`

`NS_xy_stage2.py` trains and evaluates the stationary inverse PINN.

`plot_stage2_figures_13_14.py` reproduces Figures 13 and 14 of the thesis from saved training histories. It does not train a model.

## Running Stage 2

From the repository root, run:

`python stage2_stationary_inverse/NS_xy_stage2.py`

The script automatically uses CUDA when a CUDA-enabled PyTorch installation is available. Otherwise, it runs on CPU.

## Configuration

The main user-adjustable parameters are located in the `Configuration` block at the beginning of `NS_xy_stage2.py`.

`TRAIN_MODE` controls whether the model is trained from scratch or loaded from an existing checkpoint.

`SAVE_CHECKPOINT` controls whether the trained model and run metadata are saved.

`SAVE_HISTORY` controls whether the training history required by the thesis-figure plotting script is saved.

`SAVE_PLOT_DATA` controls whether predictions, reference data, errors, inverse-parameter information, and evaluation metrics are saved for later use.

`CREATE_PLOTS` controls figure generation.

`SHOW_PLOTS` controls whether generated figures are displayed interactively.

`OVERWRITE_EXISTING_OUTPUTS` controls whether existing output files may be overwritten.

These output-related options do not affect the training result.

`RANDOM_SEED` controls the seed mode. When set to `False`, the value specified by `FIXED_SEED` is used. When set to `True`, a new seed is generated for the run.

The default fixed seed is `189869491`.

The seed used in each run is printed to the console and stored in the generated outputs where applicable.

`P_IN_INITIAL` defines the initial guess of the trainable inlet pressure.

`DATA_POINTS` selects one of the available sparse pressure-observation scenarios.

## Thesis experiments

The four basic inverse experiments use the same initial inlet-pressure estimate and training settings while varying the number of pressure observations:

| Experiment | `P_IN_INITIAL` | `DATA_POINTS` | `ADAM_EPOCHS` |
| ---------- | --------------: | ------------ | ------------  |
| Basic 1    | 8               | 5            | 5000          |
| Basic 2    | 8               | 4            | 5000          |
| Basic 3    | 8               | 3            | 5000          |
| Basic 4    | 8               | 2            | 5000          |

An additional extreme initialization experiment uses:

| Experiment | `P_IN_INITIAL` | `DATA_POINTS` | `ADAM_EPOCHS` |
| ---------- | -------------- | ------------- | ------------- |
| Extreme    | 20             | 2             | 10000         |

For the thesis settings, use `RANDOM_SEED = False` and `FIXED_SEED = 189869491`.

Small numerical differences may still occur across different PyTorch versions, CUDA versions, hardware, or numerical environments.

## Reproducing Figures 13 and 14

The history files required by `plot_stage2_figures_13_14.py` are not provided as pre-generated data.

To reproduce Figures 13 and 14, first run the five experiments listed above with:

`SAVE_HISTORY = True`

This generates:

`stage2_stationary_inverse/histories/stage2_pinit8_points5.pt`

`stage2_stationary_inverse/histories/stage2_pinit8_points4.pt`

`stage2_stationary_inverse/histories/stage2_pinit8_points3.pt`

`stage2_stationary_inverse/histories/stage2_pinit8_points2.pt`

`stage2_stationary_inverse/histories/stage2_pinit20_points2.pt`

After all five histories have been generated, run:

`python stage2_stationary_inverse/plot_stage2_figures_13_14.py`

The plotting script verifies that each history was generated using the expected initial inlet pressure, number of observation points, and number of Adam epochs.

It then generates:

`stage2_stationary_inverse/external_figures/pin_comparison_pinit8_points2to5.png`

`stage2_stationary_inverse/external_figures/pin_extreme_pinit20_points2.png`

These correspond to Figures 13 and 14 of the thesis. Text displayed inside the generated figures is provided in English.

The plotting script requires only the saved history files and does not require the trained model checkpoints.

## Evaluation metrics

Predictions are evaluated against the COMSOL reference solution using three global error metrics:

`RMSE`

`L2 error`

`Relative L2 error`

For `u_y`, RMSE and L2 error are reported. The relative L2 error is not reported numerically because the L2 norm of the reference `u_y` field is close to zero.

RMSE, L2 error, and relative L2 error are reported for:

`u_x`

`|v|`

`p`

Pointwise absolute errors are additionally calculated for plotting and post-processing.

The final inferred inlet pressure is also reported together with its relative deviation from the reference value.

## Outputs

Outputs are identified by a run name constructed from `P_IN_INITIAL` and `DATA_POINTS`.

For example, the default configuration uses:

`RUN_NAME = pinit8_points5`

Depending on the selected output options, the main script can generate a trained model checkpoint at:

`models/stage2_inverse_{RUN_NAME}.pt`

A training history at:

`stage2_stationary_inverse/histories/stage2_{RUN_NAME}.pt`

Evaluation and plotting data at:

`stage2_stationary_inverse/stage2_{RUN_NAME}_plot_data.npz`

Figures from an individual run are stored in:

`stage2_stationary_inverse/figures/{RUN_NAME}/`

The standalone thesis Figures 13 and 14 generated by `plot_stage2_figures_13_14.py` are stored in:

`stage2_stationary_inverse/external_figures/`

The saved evaluation data include the PINN predictions, COMSOL reference fields, pointwise absolute errors, evaluation metrics, observation locations and values, inlet-pressure history, seed, and number of evaluation points.

The checkpoint additionally stores the model architecture, physical parameters, non-dimensionalization scales, inverse-problem settings, training settings, loss histories, and basic execution-environment information.

## Notes

`P_IN_INITIAL` is the initial guess of the trainable inverse parameter and is not the reference inlet pressure.

The history files used to reproduce Figures 13 and 14 are generated by `NS_xy_stage2.py` and are intentionally not provided in advance.
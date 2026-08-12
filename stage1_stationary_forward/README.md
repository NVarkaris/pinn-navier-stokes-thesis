# Stage 1 - Stationary Forward Problem

This folder contains the implementation of the first stage of the thesis work.

The problem is a stationary forward Physics-Informed Neural Network (PINN) for incompressible Navier-Stokes flow in a two-dimensional rectangular channel.

The model is trained using physics-informed losses only. COMSOL reference data are used exclusively for evaluation and plotting and are not included in the training loss.

## Problem description

The neural network receives the non-dimensional spatial coordinates as input and predicts the non-dimensional velocity and pressure fields:

`(y*, x*) -> (u_y*, u_x*, p*)`

The loss function includes the continuity equation, the two Navier-Stokes momentum equations, and the boundary conditions at the inlet, outlet, upper wall, and lower wall.

## Default setup

| Parameter                 | Value                 |
| ------------------------- | --------------------- |
| Network inputs            | 2                     |
| Hidden layers             | 5                     |
| Neurons per hidden layer  | 64                    |
| Network outputs           | 3                     |
| Activation function       | tanh                  |
| Weight initialization     | Xavier normal         |
| Formulation               | Non-dimensional       |
| Domain collocation points | 2000                  |
| Upper-wall points         | 500                   |
| Lower-wall points         | 500                   |
| Inlet points              | 200                   |
| Outlet points             | 200                   |
| Collocation sampling      | Random uniform        |
| Loss weights              | Fixed, all equal to 1 |
| Adam epochs               | 5000                  |
| Adam learning rate        | 1e-3                  |
| L-BFGS maximum iterations | 300                   |
| L-BFGS learning rate      | 1e-1                  |

Training is performed sequentially using Adam followed by L-BFGS.

The initial network state is evaluated before the first Adam update and is stored as epoch 0 in the training history.

The default numerical and training settings correspond to those used for the Stage 1 thesis results.

## Required data

The COMSOL reference solution must be located at:

`data/comsol/stage1_stage2/NS_xy_stationary.txt`

Paths are resolved automatically relative to the repository root.

## Running Stage 1

From the repository root, run:

`python stage1_stationary_forward/NS_xy_stage1.py`

The script automatically uses CUDA when a CUDA-enabled PyTorch installation is available. Otherwise, it runs on CPU.

## Configuration

The main user-adjustable parameters are located in the `Configuration` block at the beginning of the script.

`TRAIN_MODE` controls whether the model is trained from scratch or loaded from an existing checkpoint.

`SAVE_CHECKPOINT` controls whether the trained model and run metadata are saved.

`SAVE_PLOT_DATA` controls whether predictions, reference data, errors, and evaluation metrics are saved for later use.

`CREATE_PLOTS` controls figure generation.

`SHOW_PLOTS` controls whether generated figures are displayed interactively.

`OVERWRITE_EXISTING_OUTPUTS` controls whether existing output files may be overwritten.

These output-related options do not affect the training result.

`RANDOM_SEED` controls the seed mode. When set to `False`, the value specified by `FIXED_SEED` is used. When set to `True`, a new seed is generated for the run.

The default fixed seed is `189869491`.

The seed used in each run is printed to the console and stored in the generated checkpoint and evaluation data when these outputs are enabled.

## Reproducibility

The default Stage 1 configuration performs exactly `ADAM_EPOCHS = 5000` Adam optimizer updates.

The untrained network is evaluated separately before the first Adam update and is stored as epoch 0 in the training history. This initial evaluation does not count as an optimizer update.

For the fixed-seed thesis configuration, use:

`RANDOM_SEED = False`

`FIXED_SEED = 189869491`

Small numerical differences may still occur across different PyTorch versions, CUDA versions, hardware, or numerical environments.

## Evaluation metrics

Predictions are evaluated against the COMSOL reference solution using three global error metrics:

`RMSE`

`L2 error`

`Relative L2 error`

The metrics are calculated for `u_y`, `u_x`, `|v|`, and `p`.

The relative L2 error of `u_y` is also calculated and reported, but it is not considered a representative accuracy measure because the reference `u_y` field is close to zero.

Pointwise absolute errors are additionally calculated for plotting and post-processing.

## Outputs

Depending on the selected output options, Stage 1 can generate a trained model checkpoint at:

`models/stage1_forward_model.pt`

Evaluation and plotting data at:

`stage1_stationary_forward/stage1_plot_data.npz`

Generated figures at:

`stage1_stationary_forward/figures/`

The saved evaluation data include the PINN predictions, COMSOL reference fields, pointwise absolute errors, RMSE, L2 errors, relative L2 errors, the seed used for the run, and the number of evaluation points.

The checkpoint contains the trained network state together with the model architecture, physical parameters, non-dimensionalization scales, training settings, seed information, loss histories, and basic execution-environment information.

## Notes

COMSOL data are not used as supervised training data.

The Stage 1 implementation uses fixed loss weights and a fixed Adam learning rate.
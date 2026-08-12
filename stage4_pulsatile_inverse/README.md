# Stage 4 - Pulsatile Inverse Problem

This folder contains the implementation of the fourth stage of the thesis work.

The problem is a time-dependent inverse Physics-Informed Neural Network (PINN) for incompressible Navier-Stokes flow.

The inlet pressure is described by a sinusoidal function whose parameters are partially or fully inferred from sparse pressure observations.

COMSOL data are used for extracting the sparse pressure observations, evaluation, and plotting. They are not used as full-field supervised training data.


## Problem description

The neural network receives the non-dimensional spatial coordinates and non-dimensional time as input and predicts the non-dimensional velocity and pressure fields.

Input and output:

`(y*, x*, t*) -> (u_y*, u_x*, p*)`

The inlet pressure is defined as:

`p_in(t) = A0 + A1 sin(2 * pi * t / T)`

The reference inlet-pressure parameters are:

- `A0 = 10 Pa`
- `A1 = 5 Pa`
- `T = 0.5 s`

The mean inlet-pressure parameter `A0` is trainable in all inverse-problem families.

Depending on the selected experiment family, the amplitude `A1` and period `T` can either be known or inferred.

The loss function includes:

- continuity residual
- y-momentum Navier-Stokes residual
- x-momentum Navier-Stokes residual
- inlet normal-velocity condition
- outlet normal-velocity condition
- initial-condition loss
- sparse pressure-data loss

All loss terms use a weight equal to 1.

The wall no-slip boundary conditions and the inlet/outlet pressure boundary conditions are imposed through hard constraints in the network output.

The inlet and outlet pressure values therefore do not appear as soft pressure-boundary loss terms.


## Inverse-problem families

The available inverse-problem families are controlled by `EXPERIMENT_FAMILY`.

| Family | `A0`      | `A1`      | `T`       |
| ------ | --------- | --------- | --------- |
| `F1`   | Trainable | Trainable | Known     |
| `F2`   | Trainable | Known     | Trainable |
| `F3`   | Trainable | Trainable | Trainable |

For known parameters, the corresponding reference value is used.

The initial values used when the parameters are trainable are controlled by:

- `A0_INITIAL`
- `A1_INITIAL`
- `P_PERIOD_INITIAL`

The standard initial values used in the thesis experiments are:

- `A0_INITIAL = 8.0 Pa`
- `A1_INITIAL = 3.0 Pa`
- `P_PERIOD_INITIAL = 0.8 s`


## Method

The final Stage 4 implementation uses the following general setup:

- Network architecture: 3 input neurons, 5 hidden layers, 64 neurons per hidden layer, 3 output neurons
- Activation function: tanh
- Weight initialization: Xavier normal
- Formulation: non-dimensional
- Domain collocation points: 10000
- Inlet points: 200
- Outlet points: 200
- Initial-condition points: 200
- Upper-wall diagnostic points: 500
- Lower-wall diagnostic points: 500
- Collocation sampling: random uniform
- Boundary treatment: hard no-slip wall boundary conditions
- Pressure treatment: hard inlet/outlet pressure ansatz
- Network Adam learning rate: 1e-3
- Network Adam scheduler: MultiStepLR
- L-BFGS learning rate: 1e-1
- L-BFGS line search: strong Wolfe

The initial condition corresponds to the steady fully developed solution used as the starting state of the pulsatile problem.

The model is not reinitialized between consecutive time windows.


## Transfer learning

Stage 4 can optionally use transfer learning from the trained Stage 3 forward model.

Transfer learning is controlled by:

`USE_TRANSFER_LEARNING`

When `USE_TRANSFER_LEARNING = True`, the following checkpoint must exist:

`models/stage3_forward_model.pt`

Only the neural-network layer weights and biases are transferred from Stage 3.

The inverse parameters `A0`, `A1`, and `T` are not transferred and retain the Stage 4 initial values defined by the selected inverse problem.

The Stage 3 checkpoint architecture is validated before the network parameters are loaded.

When `USE_TRANSFER_LEARNING = False`, the Stage 4 network uses its own Xavier initialization.


## Pressure observation scenarios

Sparse pressure observations are extracted from the COMSOL reference solution.

The amount of temporal observation data is controlled by `OBSERVATION_CASE`.

| `OBSERVATION_CASE` | Number of observation times | Description                                      |
| ------------------ | --------------------------- | ------------------------------------------------ |
| `all`              | 61                          | All COMSOL snapshots from 0 to 3 s               |
| `half`             | 30                          | Every second COMSOL snapshot, starting at 0.05 s |
| `five`             | 5                           | Five selected observation times                  |

For the `five` case, the observation times are:

`[0.10, 0.25, 0.40, 1.25, 2.40] s`

The available spatial pressure-observation locations are:

| Key     | Position     |
| ------- | ------------ |
| `x2_y2` | `(2 m, 2 m)` |
| `x3_y2` | `(3 m, 2 m)` |
| `x4_y2` | `(4 m, 2 m)` |
| `x5_y2` | `(5 m, 2 m)` |

The selected locations are controlled by:

`SELECTED_OBSERVATION_POINTS`

For example:

`SELECTED_OBSERVATION_POINTS = ["x2_y2"]`

The implementation also supports selecting more than one available spatial observation point.


## Time-window training profiles

The training strategy is controlled by `TRAINING_PROFILE`.

Four profiles are available.

### `single_fast`

A single full time window is used:

`time_windows_phys = [3.0]`

with:

`epochs_per_window = [300]`

and:

`max_lbfgs_iterations = 500`

### `curriculum_medium`

Progressive time windows are used:

`time_windows_phys = [0.5, 1.0, 2.0, 3.0]`

with:

`epochs_per_window = [700, 300, 300, 500]`

and:

`max_lbfgs_iterations = 500`

### `curriculum_hard`

Progressive time windows are used:

`time_windows_phys = [0.5, 1.0, 2.0, 3.0]`

with:

`epochs_per_window = [1000, 500, 500, 500]`

and:

`max_lbfgs_iterations = 1000`

### `custom`

The training schedule is defined manually using:

- `CUSTOM_TIME_WINDOWS_PHYS`
- `CUSTOM_EPOCHS_PER_WINDOW`
- `CUSTOM_MAX_LBFGS_ITERATIONS`

This profile is used for thesis experiments that require a specific training schedule not represented by one of the predefined profiles.

During progressive training, each new time window extends the previous one while retaining the current model and inverse-parameter state.

After Adam training is complete, the training tensors are rebuilt for the full `[0,3] s` interval before the final L-BFGS optimization.


## Adam learning rates

The neural-network parameters and inverse parameters use separate Adam optimizers.

The neural-network Adam learning rate is:

`LR_ADAM = 1e-3`

Within each time window, a `MultiStepLR` scheduler reduces the network learning rate at approximately:

- 40% of the window epochs
- 70% of the window epochs
- 90% of the window epochs

using:

`gamma = 0.5`

When:

`RESET_LR_EACH_WINDOW = True`

the neural-network learning rate is reset to `1e-3` at the beginning of each new time window.

The inverse parameters use the stage-dependent learning-rate schedule:

`INVERSE_LR_BY_STAGE = [1e-2, 5e-3, 1e-3, 1e-3]`

If more stages are used than learning rates are provided, the final value is reused.


## Current repository configuration

The current configuration of `NS_xy_stage4.py` corresponds to the successful progressive-window F2 experiment used in the thesis comparison of period identification strategies.

The active settings are:

| Parameter                     | Value                    |
| ----------------------------- | ------------------------ |
| `EXPERIMENT_FAMILY`           | `F2`                     |
| `USE_TRANSFER_LEARNING`       | `True`                   |
| `OBSERVATION_CASE`            | `all`                    |
| `SELECTED_OBSERVATION_POINTS` | `["x2_y2"]`              |
| `TRAINING_PROFILE`            | `custom`                 |
| Time windows                  | `[0.5, 1.0, 2.0, 3.0] s` |
| Adam epochs                   | `[700, 300, 300, 300]`   |
| L-BFGS maximum iterations     | `500`                    |
| Initial `A0`                  | `8 Pa`                   |
| Known `A1`                    | `5 Pa`                   |
| Initial `T`                   | `0.8 s`                  |

This configuration corresponds to Experiment 9 used in the period-recovery analysis.


## Required data

The COMSOL reference solution must be located at:

`data/comsol/stage3_stage4/NS_xy_pulsatile.txt`

The reference file contains 61 time snapshots from:

`0.00 s`

to:

`3.00 s`

with:

`dt = 0.05 s`

Paths are resolved automatically relative to the repository root.


## Scripts

Stage 4 contains six Python scripts:

`stage4_pulsatile_inverse/NS_xy_stage4.py`

`stage4_pulsatile_inverse/plot_stage4_figure22_A1_evolution.py`

`stage4_pulsatile_inverse/plot_stage4_figure23_T_evolution.py`

`stage4_pulsatile_inverse/plot_stage4_figure25_T_loss_landscape.py`

`stage4_pulsatile_inverse/plot_stage4_figure26_T_initialization.py`

`stage4_pulsatile_inverse/plot_stage4_figure27_A1_TL_observations.py`

`NS_xy_stage4.py` trains and evaluates the pulsatile inverse PINN.

The remaining scripts reproduce standalone figures used in the thesis from previously generated Stage 4 checkpoints.


## Running Stage 4

From the repository root, run:

`python stage4_pulsatile_inverse/NS_xy_stage4.py`

The script automatically uses CUDA when a CUDA-enabled PyTorch installation is available. Otherwise, it runs on CPU.


## Configuration

The main user-adjustable options are located in the `Configuration` block at the beginning of `NS_xy_stage4.py`.

The most relevant options are:

- `TRAIN_MODE`
- `SAVE_CHECKPOINT`
- `CREATE_PLOTS`
- `SHOW_PLOTS`
- `OVERWRITE_EXISTING_OUTPUTS`
- `RANDOM_SEED`
- `FIXED_SEED`
- `LOAD_MODEL_PATH`
- `EXPERIMENT_FAMILY`
- `USE_TRANSFER_LEARNING`
- `OBSERVATION_CASE`
- `SELECTED_OBSERVATION_POINTS`
- `A0_INITIAL`
- `A1_INITIAL`
- `P_PERIOD_INITIAL`
- `TRAINING_PROFILE`
- `CUSTOM_TIME_WINDOWS_PHYS`
- `CUSTOM_EPOCHS_PER_WINDOW`
- `CUSTOM_MAX_LBFGS_ITERATIONS`
- `RESET_LR_EACH_WINDOW`
- `INVERSE_LR_BY_STAGE`

`TRAIN_MODE = True` trains the selected inverse configuration.

`TRAIN_MODE = False` loads and evaluates an existing Stage 4 checkpoint.

When `TRAIN_MODE = False` and `LOAD_MODEL_PATH = None`, the checkpoint path is generated automatically from the current configuration.

Alternatively, `LOAD_MODEL_PATH` can be set explicitly to an existing Stage 4 checkpoint.


## Reproducibility

`RANDOM_SEED` controls the seed mode.

When:

`RANDOM_SEED = False`

the value specified by:

`FIXED_SEED`

is used.

The fixed seed used for the thesis implementation is:

`189869491`

The seed is applied to Python, NumPy, and PyTorch.

The seed used for each run is printed to the console and stored in the generated checkpoint.

A fixed seed improves reproducibility, but small numerical differences can still occur across different PyTorch versions, CUDA versions, hardware, or numerical environments.


## Run naming and checkpoints

The run name is generated automatically from the selected configuration.

It contains information about:

- experiment family
- transfer-learning state
- Adam epoch schedule
- L-BFGS iteration limit
- spatial observation points
- number of temporal observations
- inverse-parameter initial values

The model checkpoint is saved at:

`models/stage4_{RUN_NAME}_model.pt`

For example, the current configuration generates:

`models/stage4_F2_TL_adam700_300_300_300_lbfgs500_sp1_x2_y2_tm61_A0init8_Tinit0p8_model.pt`

The checkpoint stores the trained model state together with the information required to identify and reproduce the run, including:

- network architecture
- physical parameters
- non-dimensionalization scales
- inverse-problem family
- transfer-learning setting
- observation configuration
- training schedule
- initial and reference inverse parameters
- Adam and L-BFGS histories
- inverse-parameter histories
- loss histories
- learning-rate history
- timing information
- random seed


## Outputs

Figures generated directly by `NS_xy_stage4.py` are stored in:

`stage4_pulsatile_inverse/figures/{RUN_NAME}/`

The main script can generate figures including:

- field comparisons at selected times
- inferred-parameter convergence histories
- pressure time series at observation locations
- phase comparison between velocity and pressure
- individual loss-term evolution
- combined Adam and L-BFGS loss evolution
- Adam learning-rate evolution

All figures are saved as PNG files.

Standalone thesis figures generated by the external plotting scripts are stored in:

`stage4_pulsatile_inverse/external_figures/`


## Reproducing the external thesis figures

The external plotting scripts do not train new models.

They load specific Stage 4 checkpoints corresponding to the thesis experiments, validate their configuration, and generate the required figure.

The necessary Stage 4 experiments must therefore be run and saved before the corresponding plotting script can be executed.


### Figure 22 - Amplitude convergence with different observation counts

Run:

`python stage4_pulsatile_inverse/plot_stage4_figure22_A1_evolution.py`

The script compares the F1 experiments using:

- transfer learning
- observation point `(2 m, 2 m)`
- 61, 30, and 5 temporal observations
- 300 Adam epochs
- 500 L-BFGS maximum iterations

The generated figure is:

`stage4_pulsatile_inverse/external_figures/figure22_A1_evolution.png`


### Figure 23 - Period convergence and time-window curriculum

Run:

`python stage4_pulsatile_inverse/plot_stage4_figure23_T_evolution.py`

The script compares:

- Experiment 8: direct training on the complete `[0,3] s` interval
- Experiment 9: progressive time-window training

Both experiments use:

- F2
- transfer learning
- all 61 temporal observations
- observation point `(2 m, 2 m)`
- the same total number of Adam epochs
- 500 L-BFGS maximum iterations

Experiment 8 uses:

`[(3.0, 1600)]`

Experiment 9 uses:

`[(0.5, 700), (1.0, 300), (2.0, 300), (3.0, 300)]`

The generated figure is:

`stage4_pulsatile_inverse/external_figures/figure23_T_evolution.png`


### Figure 25 - Data-loss landscape with respect to period T

Run:

`python stage4_pulsatile_inverse/plot_stage4_figure25_T_loss_landscape.py`

The script uses the final trained checkpoint of Experiment 9:

`models/stage4_F2_TL_adam700_300_300_300_lbfgs500_sp1_x2_y2_tm61_A0init8_Tinit0p8_model.pt`

The complete trained Stage 4 state is loaded and frozen.

The network is not retrained for each tested value of `T`.

Instead, only `T` is swept over the selected range while the trained network and the remaining inverse parameters are kept fixed.

The pressure-data loss is evaluated for:

- the initial time window `[0,0.5] s`
- the complete time interval `[0,3] s`

The generated figure is:

`stage4_pulsatile_inverse/external_figures/figure25_T_loss_landscape.png`


### Figure 26 - Sensitivity to the initial period estimate

Run:

`python stage4_pulsatile_inverse/plot_stage4_figure26_T_initialization.py`

The script compares F2 experiments with different inverse-parameter initializations, focusing on sensitivity to the initial period estimate.

The experiments use:

- transfer learning
- 5 temporal observations
- observation point `(2 m, 2 m)`
- `curriculum_medium`
- Adam schedule `[700, 300, 300, 500]`
- 500 L-BFGS maximum iterations

The compared initializations are:

- Experiment 13: A0_initial = 8 Pa, T_initial = 0.8 s
- Experiment 17: A0_initial = 5 Pa, T_initial = 2.0 s
- Experiment 18: A0_initial = 5 Pa, T_initial = 3.0 s

The generated figure is:

`stage4_pulsatile_inverse/external_figures/figure26_T_initialization.png`


### Figure 27 - Transfer learning and observation-count comparison

Run:

`python stage4_pulsatile_inverse/plot_stage4_figure27_A1_TL_observations.py`

The script compares F3 experiments with:

- 61 versus 5 temporal observations
- transfer learning versus training without transfer learning
- observation point `(2 m, 2 m)`
- `curriculum_hard`
- Adam schedule `[1000, 500, 500, 500]`
- 1000 L-BFGS maximum iterations

The generated figure is:

`stage4_pulsatile_inverse/external_figures/figure27_A1_TL_observations.png`


## Evaluation metrics

Predictions are evaluated against the COMSOL reference solution using:

- RMSE
- L2 error
- Relative L2 error

RMSE, L2 error, and relative L2 error are reported for:

- `u_x`
- velocity magnitude `|v|`
- pressure `p`

For `u_y`, RMSE and L2 error are reported.

The relative L2 error of `u_y` is not interpreted numerically because the L2 norm of the reference `u_y` field is close to zero.

The script also reports:

- final inferred inverse parameters
- relative deviation of the inferred parameters from their reference values
- per-time-slice field errors
- boundary-condition checks
- training time for each Adam stage
- total Adam time
- L-BFGS time
- total training time


## Notes

The Stage 4 implementation corresponds to the final pulsatile inverse methodology used in the thesis.

The same core script supports the F1, F2, and F3 inverse-problem families.

Progressive time-window training acts as a continuation strategy: the inverse problem is first optimized on a shorter time interval and the trained state is then continued into progressively larger intervals.

The model and inverse parameters are not reinitialized when the time window is expanded.

Transfer learning, when enabled, transfers only the Stage 3 neural-network weights and biases. It does not transfer inverse-parameter values.

COMSOL data are used only for sparse pressure observations, evaluation, and plotting. They are not used as full-field supervised training data.

All generated figures are saved as PNG files.
# Stage 3 - Pulsatile Forward Problem

This folder contains the implementation of the third stage of the thesis work.

The problem is a time-dependent forward Physics-Informed Neural Network (PINN) for incompressible Navier-Stokes flow in a two-dimensional rectangular channel.

The flow is driven by a known pulsatile inlet pressure.

The main PINN is trained using physics-informed losses only. COMSOL data are used for evaluation and plotting and are not used as full-field supervised training data.

## Problem description

The neural network receives the non-dimensional spatial coordinates and non-dimensional time as input and predicts the non-dimensional velocity and pressure fields:

`(y*, x*, t*) -> (u_y*, u_x*, p*)`

The prescribed inlet pressure is:

`p_in(t) = A0 + A1 sin(2*pi*t/T)`

with the reference parameters:

| Parameter                 | Value |
| ------------------------- | ----- |
| Mean inlet pressure `A0`  | 10 Pa |
| Pressure amplitude `A1`   | 5 Pa  |
| Period `T`                | 0.5 s |
| Outlet pressure           | 0 Pa  |
| Initial time              | 0 s   |
| Final time                | 3 s   |

The wall no-slip conditions and the inlet/outlet pressure conditions are imposed directly through hard constraints in the neural-network output.

The training loss contains the continuity residual, the two transient Navier-Stokes momentum residuals, the remaining inlet and outlet velocity constraints, and the initial-condition loss.

## Default setup

| Parameter                         | Value                 |
| --------------------------------- | --------------------- |
| Network inputs                    | 3                     |
| Hidden layers                     | 5                     |
| Neurons per hidden layer          | 64                    |
| Network outputs                   | 3                     |
| Activation function               | tanh                  |
| Weight initialization             | Xavier normal         |
| Formulation                       | Non-dimensional       |
| Domain collocation points         | 10000                 |
| Inlet points                      | 200                   |
| Outlet points                     | 200                   |
| Initial-condition points          | 200                   |
| Upper-wall boundary-check points  | 500                   |
| Lower-wall boundary-check points  | 500                   |
| Collocation sampling              | Random uniform        |
| Loss weights                      | Fixed, all equal to 1 |
| Adam learning rate                | 1e-3                  |
| L-BFGS maximum iterations         | 1000                  |
| L-BFGS learning rate              | 1e-1                  |

The upper and lower-wall point sets are retained for boundary-condition checks. The corresponding no-slip conditions are imposed through the hard output transformation rather than through wall penalty terms.

## Time-window training

Stage 3 uses progressive time-window training.

The default training sequence is:

| Stage | Training interval | Adam epochs |
| ----- | ----------------- | ----------- |
| 1     | `[0, 1] s`        | 3000        |
| 2     | `[0, 2] s`        | 3000        |
| 3     | `[0, 3] s`        | 4000        |

The Adam learning rate is reset at the beginning of each time window.

Within each window, `MultiStepLR` reduces the learning rate at 40%, 70%, and 90% of the corresponding Adam stage using a multiplicative factor of 0.5.

After the final Adam stage, the solution is refined using L-BFGS for up to 1000 iterations.

The training points are generated through dedicated point-building functions so that new collocation sets can be constructed as the active time window is expanded.

## Required data

The COMSOL pulsatile-flow reference solution must be located at:

`data/comsol/stage3_stage4/NS_xy_pulsatile.txt`

The exported reference solution contains snapshots from `t = 0` to `t = 3 s` with a time step of `0.05 s`, corresponding to 61 time snapshots.

Paths are resolved automatically relative to the repository root.

## Scripts

This folder contains two Python scripts:

`stage3_pulsatile_forward/NS_xy_stage3.py`

`stage3_pulsatile_forward/stage3_pressure_animation.py`

`NS_xy_stage3.py` trains and evaluates the pulsatile forward PINN and creates the numerical results and figures associated with Stage 3.

`stage3_pressure_animation.py` is an optional visualization script. It loads the trained Stage 3 checkpoint and creates an animated GIF of the predicted PINN pressure field over the full transient interval.

## Running the main Stage 3 model

From the repository root, run:

`python stage3_pulsatile_forward/NS_xy_stage3.py`

The script automatically uses CUDA when a CUDA-enabled PyTorch installation is available. Otherwise, it runs on CPU.

## Configuration

The main user-adjustable options are located in the `Configuration` block at the beginning of `NS_xy_stage3.py`.

`TRAIN_MODE` controls whether the model is trained or loaded from the saved Stage 3 checkpoint.

`SAVE_CHECKPOINT` controls whether the trained model and run metadata are saved.

`SAVE_PLOT_DATA` controls whether the evaluated fields, errors, and global metrics are saved for later post-processing.

`CREATE_PLOTS` controls figure generation.

`SHOW_PLOTS` controls whether generated figures are displayed interactively.

`OVERWRITE_EXISTING_OUTPUTS` controls whether existing output files may be overwritten.

`RANDOM_SEED` controls the seed mode. When set to `False`, the value specified by `FIXED_SEED` is used. When set to `True`, a new seed is generated for the run.

The default fixed seed is:

`FIXED_SEED = 189869491`

The default time-window configuration is:

`TIME_WINDOWS_PHYS = [1.0, 2.0, 3.0]`

`EPOCHS_PER_WINDOW = [3000, 3000, 4000]`

`RESET_LR_EACH_WINDOW = True`

The default physical parameters are:

`P_IN_MEAN = 10.0`

`P_IN_AMPLITUDE = 5.0`

`P_PERIOD = 0.5`

`P_OUT_DIM = 0.0`

`T_INITIAL = 0.0`

`T_FINAL = 3.0`

`DT_EXPORT = 0.05`

For reproducible runs corresponding to the thesis configuration, use:

`RANDOM_SEED = False`

`FIXED_SEED = 189869491`

Small numerical differences may still occur across different PyTorch versions, CUDA versions, hardware, or numerical environments.

## Stage 3 checkpoint

When checkpoint saving is enabled, the trained Stage 3 model is stored at:

`models/stage3_forward_model.pt`

The checkpoint contains the complete model state together with architecture, physical parameters, non-dimensionalization information, training settings, histories, seed information, and execution-environment metadata.

It also stores the neural-network layer state separately so that the trained Stage 3 network can be used for transfer learning in Stage 4.

The pressure-animation script requires this checkpoint.

If the checkpoint does not exist, first run `NS_xy_stage3.py` with:

`SAVE_CHECKPOINT = True`

If an output already exists and `OVERWRITE_EXISTING_OUTPUTS = False`, the corresponding script stops instead of overwriting it.

## Thesis figures

The main Stage 3 script generates the data and plots corresponding to Figures 15-21 of the thesis.

These include:

| Thesis figure | Content                                                                       |
| ------------- | ----------------------------------------------------------------------------- |
| 15            | Total loss evolution during Adam and L-BFGS training                          |
| 16            | Pressure time series at `(x, y) = (5, 2) m`                                   |
| 17            | Velocity-pressure temporal comparison at `(x, y) = (5, 2) m`                  |
| 18            | Velocity-magnitude field comparison at `t = 2.60 s`                           |
| 19            | Pressure-field comparison at `t = 2.60 s`                                     |
| 20            | Velocity and pressure profiles at selected times                              |
| 21            | Time evolution of the relative L2 errors of velocity magnitude and pressure   |

The generated figures use English titles, axis labels, legends, and annotations.

Additional diagnostic figures may also be produced by the main script.

## Evaluation metrics

Predictions are evaluated against the COMSOL reference solution using three global error metrics:

`RMSE`

`L2 error`

`Relative L2 error`

For `u_y`, RMSE and L2 error are reported. A numerical relative L2 error is not reported because the L2 norm of the reference `u_y` field is close to zero and the resulting ratio would not be representative.

RMSE, L2 error, and relative L2 error are reported for:

`u_x`

`|v|`

`p`

The script also evaluates pointwise absolute errors, boundary-condition satisfaction, errors at individual time snapshots, and the phase relation between pressure and longitudinal velocity at the selected monitoring point.

## Main-script outputs

Depending on the selected configuration, `NS_xy_stage3.py` can generate the following outputs.

Model checkpoint:

`models/stage3_forward_model.pt`

Optional evaluation and plotting data:

`stage3_pulsatile_forward/stage3_plot_data.npz`

Figures:

`stage3_pulsatile_forward/figures/`

The saved evaluation data include the PINN predictions, COMSOL reference fields, pointwise errors, global evaluation metrics, seed, and number of evaluation points.

## Pressure-field animation

The optional animation script creates a GIF showing the predicted PINN pressure field during the pulsatile simulation.

Run:

`python stage3_pulsatile_forward/stage3_pressure_animation.py`

The script:

- loads the COMSOL mesh and time grid,
- verifies that the data contain 61 snapshots from `0` to `3 s` with `DT_EXPORT = 0.05 s`,
- loads `models/stage3_forward_model.pt`,
- verifies that the checkpoint is compatible with the Stage 3 network architecture,
- evaluates the PINN pressure field at every COMSOL node and time snapshot,
- and creates the animated pressure-field visualization.

The generated GIF is saved at:

`stage3_pulsatile_forward/animations/stage3_pinn_pressure_field.gif`

The animation therefore contains one frame for each physical time snapshot:

`0.00, 0.05, 0.10, ..., 3.00 s`

The default animation playback rate is:

`FPS = 12`

`FPS` controls only the playback speed of the GIF. It does not change the physical simulation time step, which remains `0.05 s`.

The pressure color scale is kept fixed throughout the animation. The COMSOL reference pressure and the PINN pressure are used to determine common pressure limits, while the field displayed in each animation frame is the PINN prediction.

The animation was created for presentation use and is not required for training or evaluating the Stage 3 model.

## Animation configuration

The main animation options are located at the beginning of `stage3_pressure_animation.py`:

`OVERWRITE_EXISTING_OUTPUTS`

`FPS`

`LEVELS`

`CMAP`

The animation script automatically uses CUDA when available and otherwise runs on CPU.

## Notes

Stage 3 is a forward problem: the pulsatile inlet-pressure parameters are prescribed and are not inferred during training.

COMSOL data are not used as full-field supervised training data.

The progressive time-window strategy expands the temporal training domain while retaining the previously learned network state.

The Stage 3 checkpoint is also the pretrained network source used by Stage 4 when transfer learning is enabled.

The pressure animation is an optional presentation visualization and does not modify the trained model.
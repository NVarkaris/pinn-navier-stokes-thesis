# Reference Data

This directory contains the COMSOL Multiphysics reference solutions used by the four PINN stages.

The reference datasets are exported from COMSOL and are included in the repository so that the simulations can be evaluated and the inverse problems can extract the required sparse pressure observations without requiring access to the original COMSOL models.


## Directory structure

The expected structure is:

`data/comsol/stage1_stage2/NS_xy_stationary.txt`

`data/comsol/stage3_stage4/NS_xy_pulsatile.txt`

The filenames and directory structure should not be changed unless the corresponding paths in the Python scripts are also updated.


## Stationary reference solution

Stages 1 and 2 use:

`data/comsol/stage1_stage2/NS_xy_stationary.txt`

This file contains the COMSOL reference solution for the stationary incompressible Navier-Stokes problem.

### Stage 1

In Stage 1, the COMSOL solution is used for:

- evaluation of the trained PINN
- calculation of global error metrics
- plotting and comparison of the predicted and reference fields

The COMSOL data are not used during PINN training.

### Stage 2

In Stage 2, the same stationary COMSOL solution is used for:

- extraction of sparse pressure observations for the inverse problem
- evaluation of the trained PINN
- calculation of global error metrics
- plotting and comparison

Only the selected sparse pressure observations are included in the inverse training loss.

The full COMSOL field is not used as supervised training data.


## Pulsatile reference solution

Stages 3 and 4 use:

`data/comsol/stage3_stage4/NS_xy_pulsatile.txt`

This file contains the time-dependent COMSOL reference solution for the pulsatile incompressible Navier-Stokes problem.

The dataset contains 61 time snapshots covering:

`0.00 s <= t <= 3.00 s`

with:

`dt = 0.05 s`

### Stage 3

In Stage 3, the COMSOL solution is used for:

- evaluation of the trained time-dependent PINN
- calculation of global and per-time-slice error metrics
- field and time-series comparisons
- plotting

The COMSOL data are not used during PINN training.

### Stage 4

In Stage 4, the pulsatile COMSOL solution is used for:

- extraction of sparse pressure observations for the inverse problem
- evaluation of the trained PINN
- calculation of global and per-time-slice error metrics
- field and time-series comparisons
- plotting

Only the selected sparse pressure observations are used in the data-loss term.

The complete COMSOL field is not used as supervised training data.


## Data usage

The COMSOL datasets provide numerical reference solutions for evaluating the PINN predictions and, in the inverse stages, for constructing the specified sparse pressure observations.

The distinction between reference data and training data is important:

- Stages 1 and 3 are trained using physics-informed constraints without COMSOL field data.
- Stages 2 and 4 additionally use only selected sparse pressure observations extracted from the corresponding COMSOL solution.
- No stage is trained using the complete COMSOL solution as full-field supervised data.


## Notes

The two `.txt` files are included directly in the repository.

No COMSOL installation is required to run the provided Python scripts because the required reference solutions have already been exported.

The original COMSOL model files are not required by the Python implementation.
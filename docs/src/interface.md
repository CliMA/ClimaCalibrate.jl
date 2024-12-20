# ClimaCalibrate

ClimaCalibrate is just a placeholder name, we can bikeshed on the name later

ClimaCalibrate provides a framework for running component model calibrations using EnsembleKalmanProcesses. This way, artifacts for calibration experiments can be stored

To use this framework, individual models (or the coupler) will define their own versions of the functions provided in the interface (`get_config`, `get_forward_model`, and `run_forward_model`).

Calibrations can either be run using pure Julia or if required, an Sbatch pipeline for the Caltech Resnick cluster.

The framework has three main interface components:

**EKP**: Contains the functions for initializing calibration, updating ensembles, and additional emulate/sample steps.

**Model**: Contains hooks/function stubs for component models to integrate. This consists of three functions
- `forward_model = get_forward_model(Val{experiment_id})`: Dispatches on the experiment ID to find the right component model.

- `config = get_config(physical_model, experiment_id, iteration, member)`: Gets the configuration for the model to run

- `run_forward_model(physical_model, config)`: Runs the forward model, which is meant to save output to a file

**Slurm**: Provides utilities for running experiments on the central cluster, file and error handling.
It would be good to separate this from the Julia package, since this is largely specific to the central cluster.

# Component Model

## Model interface
The model must provide implementations of the following function stubs. This will be common across experiments for a single component model.

`physical_model = get_physical_model(Val{experiment_id})`: Dispatches on the experiment ID to find the right component model.

`config = get_config(physical_model, experiment_id, iteration, member)`: Gets the configuration for the model to run

`run_forward_model(physical_model, config)`: Runs the forward model, which is meant to save output to a file

## Experiments
This will define an individual experiment to work with the model interface and ClimaCalibrate.
An experiment should be self-documented and contained. 

Experiments need to have the following:
- **EKP configuration YAML**: This holds the configuration for the experiment, including paths to other experiment metada such as the prior distribution file.This contains:
    - Path to the prior distribution
    - Path to truth data and noise JLD2 files
    - Ensemble size
    - Number of iterations
    - Output directory
    - In the future, other EKP configuration arguments (Unscented etc)
- **Observational data and noise**: Serialized to JLD2
- **Prior distribution file**: TOML format. Distributions should generally use the [constrained_gaussian](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/ParameterDistributions/#EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian) function.
- **Project.toml and compat**: Setting compat requirements is important for reproducibility, and allows for additional imports needed in the observation map.
- **Observation map**: A function `observation_map` which computes the observation map for a full ensemble.
    - The `observation_map` function can live in a file in the experiment folder which is `included` in the calibration script. Alternatively, all observation maps for a given model can be kept in the model interface to avoid using `include`. However, it has the drawbacks of adding unnecessary dependencies for a given experiment.
- **Model configuration file** (optional): May be required by the model interface. This should inherit values like the output directory for the EKP config if applicable.
- **Script for generating the observational data** (optional): Ideal for reproducibility, especially for perfect model scenarios.
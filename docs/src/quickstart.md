# Getting Started

First, make sure your system meets the dependencies of [CalibrateEmulateSample.jl](https://clima.github.io/CalibrateEmulateSample.jl/dev/installation_instructions/).
You can run calibrations a cluster that supports Slurm or on your local machine.

A good way to get started is to run the example experiment, `surface_fluxes_perfect_model`, which uses the [SurfaceFluxes.jl](https://github.com/CliMA/SurfaceFluxes.jl) package to generate a physical model that calculates the Monin Obukhov turbulent surface fluxes based on idealized atmospheric and surface conditions. Since this is a "perfect model" example, the same model is used to generate synthetic observations using its default parameters and a small amount of noise. These synthetic observations are considered to be the ground truth, which is used to assess the model ensembles' performance when parameters are drawn from the prior parameter distributions. 
It is a perfect-model calibration, serving as a test case for the initial pipeline.
By default, it runs 10 ensemble members for 10 iterations. Further details can be found in the experiment folder, `experiments/surace_fluxes_perfect_model`.

## Local Machine

To run the example experiment on your local machine, first open your REPL with the proper project:
`julia --project=experiments/surface_fluxes_perfect_model`

Next, run the following code:
```julia
import CalibrateAtmos

experiment_dir = dirname(Base.active_project())

# Generate observational data and include observational map + model interface
include(joinpath(experiment_dir, "generate_data.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))
include(joinpath(experiment_dir, "model_interface.jl"))

eki = CalibrateAtmos.calibrate(experiment_dir)
include(joinpath(experiment_dir, "postprocessing.jl"))
```

## HPC Cluster
This method will queue Julia processes to run on your slurm cluster.

To run this experiment:
1. Log onto the Caltech HPC
2. Clone CalibrateAtmos.jl and `cd` into the repository.
3. Start julia: `julia --project=experiments/surace_fluxes_perfect_model`
4. Run the following:
```julia
import CalibrateAtmos: CaltechHPC, calibrate

experiment_dir = dirname(Base.active_project())

include(joinpath(experiment_dir, "generate_data.jl"))
model_interface = joinpath(experiment_dir, "model_interface.jl")
include(joinpath(experiment_dir, "observation_map.jl"))
eki = calibrate(
        CaltechHPC(),
        experiment_dir;
        time_limit = 3,
        model_interface)

include(joinpath(experiment_dir, "postprocessing.jl"))
```

New experiments should be defined within the component model repos (in this case, `SurfaceFluxes.jl`), so that the internals of `CalibrateAtmos.jl` do not explicitly depend on component models.

# Getting Started

Every calibration requires the following information:
- A forward model
- Observational data
- Prior parameter distribution

To run a calibration, you will need to implement two functions:
- [`forward_model(iteration, member)`](@ref): Run the forward model, saving model output to disk.
- [`observation_map(iteration)`](@ref): Return the full ensemble's observation map, transforming forward model output into the same space as the observational data.

Since these functions only have access to the iteration and member numbers but need to set parameters and save data, there are some helpful hooks:
- [`path_to_ensemble_member`](@ref) returns the ensemble member's output directory, which can be used to set the forward model's output directory.
- [`parameter_path`](@ref) returns the ensemble member's parameter file, which can be loaded in via TOML or passed to ClimaParams.

# Example Calibration

A good way to get started is to run the example experiment, `surface_fluxes_perfect_model`, which uses the [SurfaceFluxes.jl](https://github.com/CliMA/SurfaceFluxes.jl) package to generate a physical model that calculates the Monin Obukhov turbulent surface fluxes based on idealized atmospheric and surface conditions. Since this is a "perfect model" example, the same model is used to generate synthetic observations using its default parameters and a small amount of noise. These synthetic observations are considered to be the ground truth, which is used to assess the model ensembles' performance when parameters are drawn from the prior parameter distributions. 

It is a perfect-model calibration, using its own output as observational data. 
By default, it runs 20 ensemble members for 6 iterations. 
This example can be run on the most common backend, the WorkerBackend, with the following script:

```julia
using ClimaCalibrate, Distributed

addprocs(SlurmManager(5))
include(joinpath(pkgdir(ClimaCalibrate), "test", "sf_calibration_utils.jl"))

eki = calibrate(
    WorkerBackend,
    ensemble_size,
    n_iterations,
    observation,
    variance,
    prior,
    output_dir,
)

test_sf_calibration_output(eki, prior, observation)

theta_star_vec =
    (; coefficient_a_m_businger = 4.7, coefficient_a_h_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger", "coefficient_a_h_businger"],
)

g_vs_iter_plot(eki)
```

# Getting Started

Every calibration requires
- a forward model, which uses input parameters to return diagnostic output
- observational data, which can be a Vector or an [`EnsembleKalmanProcess.Observation`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/Observations/#EnsembleKalmanProcesses.Observation)
- a prior parameter distribution. The easiest way to construct a distribution is with the [`EnsembleKalmanProcess.constrained_gaussian`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/ParameterDistributions/#EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian) function.

## Implementing your experiment

### Forward Model

Your forward model must implement the [`forward_model(iteration, member)`](@ref) 
function stub.

Since this function only takes in the iteration and member numbers, there are some 
hooks to obtain parameters and the output directory:

- [`path_to_ensemble_member`](@ref) returns the ensemble member's output directory, 

which can be used to set the forward model's output directory.
- [`parameter_path`](@ref) returns the ensemble member's parameter file, which can 
be loaded in via TOML or passed to ClimaParams.

### Observational data

Observational data generally consists of a vector of observations with length `d`
 and the covariance matrix of the observational noise with size `d × d`.

If you need to stack or sample from observations, EnsembleKalmanProcesses.jl's 
[Observation](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/Observations/#Observation) 
or [ObservationSeries](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/Observations/#ObservationSeries) are fully-featured.

An **observation map** to process model output and return the full ensemble's observations is also required.

This is provided by implementing the function stub [`observation_map(iteration)`](@ref). This function needs to return an `Array arr` where `arr[:, i]` will return the i-th ensemble member's observational output.

Here is a readable template for the `observation_map`

```julia
function observation_map(iteration)
    single_observation_dims = 1
    G_ensemble = Array{Float64}(undef, single_observation_dims..., ensemble_size)
    for member in 1:ensemble_size
        G_ensemble[:, member] = process_member_data(iteration, member)
    end
    return G_ensemble
end
```

### Optional postprocessing

It may be the case that `observation_map` is insufficient as you need to more information,
such as information from the `ekp` object to compute `G_ensemble`. Further postprocessing of the
`G_ensemble` object can be done by implementing the `postprocess_g_ensemble` as shown
below.

```julia
function postprocess_g_ensemble(ekp, g_ensemble, prior, output_dir, iteration)
    return g_ensemble
end
```

After each evaluation of the observation map and before updating the ensemble, it may be
helpful to print the errors from the `ekp` object or plot `G_ensemble`. This can be done
by implementing the `analyze_iteration` as shown below.

```julia
function ClimaCalibrate.analyze_iteration(
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration,
)
    @info "Analyzing iteration"
    @info "Iteration $iteration"
    @info "Current mean parameter: $(EnsembleKalmanProcesses.get_ϕ_mean_final(prior, ekp))"
    @info "g_ensemble: $g_ensemble"
    @info "output_dir: $output_dir"
    return nothing
end
```

### Parameters

Every parameter that is being calibrated requires a prior distribution to sample from.

EnsembleKalmanProcesses.jl's [constrained_gaussian](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/ParameterDistributions/#EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian) 
provides a user-friendly way to constructor Gaussian distributions.

Multiple distributions can be combined using `combine_distributions(vec_of_distributions)`.

For more information, see the EKP documentation for [prior distributions](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/).

### Experiment Configuration
A calibration consisting of `m` ensemble members will run for `n` iterations.

A good rule of thumb is an ensemble size 10 times the number of parameters.

### Calibrate

Now all of the pieces should be in place:
- forward map
- observation map
- observations
- covariance matrix of the observations (noise)
- prior distribution
- ensemble size
- number of iterations

And we can put it all together:

`calibrate(ensemble_size, n_iterations, observations, noise, prior, output_dir)`

Lastly, you need to set the output directory, ensemble size and the number of iterations to run for. A good rule of thumb for your ensemble size is 10x the number of free parameters.

```julia
n_iterations = 7
ensemble_size = 10
output_dir = "output/my_experiment"
```
Once all of this has been set up, you can call put it all together using the [`calibrate`](@ref) function:

```julia
calibrate(ensemble_size, n_iterations, observations, noise, prior, output_dir)
```

For more information on parallelizing your calibration, see the [Backends](https://clima.github.io/ClimaCalibrate.jl/dev/backends/) page.

# Checkpointing

ClimaCalibrate checkpoints each forward model and iteration so that an interrupted
calibration can seamlessly pick up where it left off without wasting resources.

If a calibration (run via `calibrate`) exits after completing an iteration, 
when it is restarted it will automatically run the next iteration. 
This is done by checking if the ensemble forward map results file (`G_ensemble.jld2`) 
and the EKI file (`eki_file.jld2`) have been saved.

If a calibration is interrupted during forward model execution, 
causing a partial iteration, incomplete forward models will be rerun when the 
calibration is restarted. Completed forward models will not be rerun.
This is done by checking each model's checkpoint file and the flag it contains.

# Example Calibrations

The [example tutorial](https://clima.github.io/ClimaCalibrate.jl/dev/literate_example/)
provides a clear calibration example that can be run locally.

Another example experiment can be found in the package repo under `experiments/surface_fluxes_perfect_model`.
This experiment uses the [SurfaceFluxes.jl](https://github.com/CliMA/SurfaceFluxes.jl) package 
to generate a physical model that calculates the Monin Obukhov turbulent surface 
fluxes based on idealized atmospheric and surface conditions. Since this is a "perfect 
model" example, the same model is used to generate synthetic observations using its 
default parameters and a small amount of noise. These synthetic observations are 
considered to be the ground truth, which is used to assess the model ensembles' 
performance when parameters are drawn from the prior parameter distributions. 

It is a perfect-model calibration, using its own output as observational data. 
By default, it runs 20 ensemble members for 6 iterations. 
This example can be run on the most common backend, the WorkerBackend, with the following script:

```julia
using ClimaCalibrate

include(joinpath(pkgdir(ClimaCalibrate), "experiments", "surface_fluxes_perfect_model", "utils.jl"))
@show ensemble_size n_iterations observation variance prior

eki = calibrate(
    ensemble_size,
    n_iterations,
    observation,
    variance,
    prior,
    output_dir,
)

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

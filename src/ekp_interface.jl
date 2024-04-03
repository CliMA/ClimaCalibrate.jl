import TOML, YAML
import JLD2
import Random
using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaComms

export ExperimentConfig

struct ExperimentConfig
    id::AbstractString
    n_iterations::Integer
    ensemble_size::Integer
    observations::Any
    noise::Any
    prior::Any
    output_dir::Any
end

"""
    ExperimentConfig(experiment_id)

    ExperimentConfig(
        experiment_id,
        n_iterations,
        ensemble_size,
        observations,
        noise,
        prior,
        output_dir,
    )

ExperimentConfig stores the configuration for a specific experiment.
If only the `experiment_id` is passed, the config will be loaded from
`experiments/experiment_id/ekp_config.yml`
"""
function ExperimentConfig(experiment_id)
    config_yaml = joinpath("experiments", experiment_id, "ekp_config.yml")
    config_dict = isfile(config_yaml) ? Dict() : YAML.load_file(config_yaml)

    required_fields = ["n_iterations",
    "ensemble_size",
    "prior_path",
    "observations",
    "noise"]
    @assert issubset(required_fields, keys(config_dict))

    default_output =
        haskey(ENV, "CI") ? experiment_id : joinpath("output", experiment_id)
    output_dir = get(config_dict, "output_dir", default_output)

    observations = JLD2.load_object(config_dict["observations"])
    noise = JLD2.load_object(config_dict["noise"])

    prior = get_prior(config_dict["prior_path"])

    return ExperimentConfig(
        experiment_id,
        config_dict["n_iterations"],
        config_dict["ensemble_size"],
        observations,
        noise,
        prior,
        output_dir,
    )
end

"""
    path_to_iteration(output_dir, iteration)
Returns the path to the iteration folder within `output_dir` for the given iteration number.
"""
path_to_iteration(output_dir, iteration) =
    joinpath(output_dir, join(["iteration", lpad(iteration, 3, "0")], "_"))

"""
    get_prior(prior_pathAbstractString; names = nothing)
    get_prior(param_dict::AbstractDict; names = nothing)

Constructs the combined prior distribution from the TOML file at the `prior_path`.
If no parameter names are passed in, all parameters in the TOML are used in the distribution.
"""
function get_prior(prior_path::AbstractString; names = nothing)
    param_dict = TOML.parsefile(prior_path)
    return get_prior(param_dict; names)
end

function get_prior(param_dict::AbstractDict; names = nothing)
    names = isnothing(names) ? keys(param_dict) : names
    prior_vec = [get_parameter_distribution(param_dict, n) for n in names]
    prior = combine_distributions(prior_vec)
    return prior
end

"""
    save_G_ensemble(experiment_id, iteration, G_ensemble)

Save an ensemble's observation map output to the correct folder.
"""
function save_G_ensemble(experiment_id, iteration, G_ensemble)
    config = ExperimentConfig(experiment_id)
    iter_path = path_to_iteration(config.output_dir, iteration)
    JLD2.save_object(joinpath(iter_path, "G_ensemble.jld2"), G_ensemble)
end

"""
    initialize(config::ExperimentConfig)
    initialize(
        experiment_id::AbstractString;
        kwargs...
    )
Initializes the EKP object and the model ensemble. See ExperimentConfig for a full list of keyword arguments.
"""
initialize(experiment_id::AbstractString) =
    initialize(ExperimentConfig(experiment_id))

function initialize(config::ExperimentConfig)
    Random.seed!(rng_seed)
    rng_ekp = Random.MersenneTwister(rng_seed)

    (; observations, noise, prior, output_dir) = config

    initial_ensemble =
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size)
    eki = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        observations,
        noise,
        EKP.Inversion();
        rng = rng_ekp,
        failure_handler_method = EKP.SampleSuccGauss(),
    )

    save_parameter_ensemble(
        EKP.get_u_final(eki), # constraints applied when saving
        prior,
        param_dict,
        output_dir,
        "parameters.toml",
        0,  # Initial iteration = 0
    )

    # Save in EKI object in iteration_000 folder
    eki_path = joinpath(output_dir, "iteration_000", "eki_file.jld2")
    JLD2.save_object(eki_path, eki)
    return eki
end

"""
    update_ensemble(
        experiment_id,
        iteration;
        config = YAML.load_file("experiments/\$experiment_id/ekp_config.yml"),
    )
Updates the EKI object and saves parameters for the next iteration.
Assumes that the observation map has been run and saved in the current iteration folder.
"""
update_ensemble(experiment_id, iteration) =
    update_ensemble(ExperimentConfig(experiment_id), iteration)

function update_ensemble(configuration::ExperimentConfig, iteration;)
    (; prior, output_dir) = configuration
    # Load EKI object from iteration folder
    iter_path = path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))

    # Load data from the ensemble
    G_ens = JLD2.load_object(joinpath(iter_path, "G_ensemble.jld2"))

    # Update
    EKP.update_ensemble!(eki, G_ens)
    iteration += 1

    save_parameter_ensemble(
        EKP.get_u_final(eki),  # constraints applied when saving
        prior,
        param_dict,
        output_dir,
        "parameters.toml",
        iteration,
    )

    # Save EKI object for next iteration
    iter_path = path_to_iteration(output_dir, iteration)
    eki_path = joinpath(iter_path, "eki_file.jld2")
    JLD2.save_object(eki_path, eki)
    return eki
end

"""
    calibrate(experiment_id)

Convenience function for running a full calibration experiment for the given
`experiment_id`. 
This function requires the relevant experiment project and model interface to be loaded.

```julia
import CalibrateAtmos

experiment_id = "surface_fluxes_perfect_model"
experiment_path = joinpath(pkgdir(CalibrateAtmos), "experiments", experiment_id)
include(joinpath(experiment_path, "model_interface.jl"))
include(joinpath(experiment_path, "generate_data.jl"))

eki = CalibrateAtmos.calibrate(experiment_id)
```
"""
function calibrate(experiment_id; device = ClimaComms.device(), kwargs...)
    configuration = ExperimentConfig(experiment_id; kwargs...)
    # initialize the CalibrateAtmos
    initialize(experiment_id)

    # run experiment with CalibrateAtmos for N_iter iterations
    (; n_iterations, ensemble_size) = configuration

    eki = nothing
    physical_model = get_forward_model(Val(Symbol(experiment_id)))
    lk = ReentrantLock()
    for i in 0:(n_iterations - 1)
        ClimaComms.@threaded device for m in 1:ensemble_size
            # model run for each ensemble member
            run_forward_model(
                physical_model,
                get_config(physical_model, m, i, experiment_id);
                lk,
            )
            @info "Finished model run for member $m at iteration $i"
        end

        # update EKP with the ensemble output and update calibrated parameters
        G_ensemble = observation_map(Val(Symbol(experiment_id)), i)
        save_G_ensemble(experiment_id, i, G_ensemble)
        eki = update_ensemble(experiment_id, i)
    end
    return eki
end

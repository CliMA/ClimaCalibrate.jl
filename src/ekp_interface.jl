import TOML, YAML
import JLD2
import Random
using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaComms

export ExperimentConfig

"""
    ExperimentConfig(experiment_dir)
    ExperimentConfig(experiment_config_file)

    ExperimentConfig(
        experiment_id,
        n_iterations,
        ensemble_size,
        observations,
        noise,
        prior,
        output_dir,
        emulate_sample,
    )

ExperimentConfig stores the configuration for a calibration experiment.
If a .yml file is given, it will be used to construct the ExperimentConfig.
If a folder is given, the ExperimentConfig will be constructed from a nested file `ekp_config.yml`. 
For customizable interactive experiments, arguments can be passed in directly.
"""
struct ExperimentConfig
    id::AbstractString
    n_iterations::Integer
    ensemble_size::Integer
    observations::Any
    noise::Any
    prior::ParameterDistribution
    output_dir::Any
    emulate_sample::Bool
end

function ExperimentConfig(filepath::AbstractString; kwargs...)
    filepath_extension = joinpath(filepath, "ekp_config.yml")
    if endswith(filepath, ".yml") && isfile(filepath)
        config_dict = YAML.load_file(filepath)
        experiment_dir = dirname(filepath)
    elseif isdir(filepath) &&
           isfile(filepath_extension) &&
           endswith(filepath_extension, ".yml")
        config_dict = YAML.load_file(filepath_extension)
        experiment_dir = filepath
    else
        error("Invalid experiment configuration filepath: `$filepath`")
    end
    # If no ID is given, use name of folder containing config file as the ID
    experiment_id =
        get(config_dict, "experiment_id", last(splitdir(experiment_dir)))
    default_output =
        haskey(ENV, "CI") ? experiment_id : joinpath("output", experiment_id)
    output_dir = get(config_dict, "output_dir", default_output)

    n_iterations = config_dict["n_iterations"]
    ensemble_size = config_dict["ensemble_size"]
    observations =
        JLD2.load_object(joinpath(experiment_dir, config_dict["observations"]))
    noise = JLD2.load_object(joinpath(experiment_dir, config_dict["noise"]))
    prior = get_prior(joinpath(experiment_dir, config_dict["prior_path"]))

    return ExperimentConfig(
        experiment_id,
        n_iterations,
        ensemble_size,
        observations,
        noise,
        prior,
        output_dir,
        get(config_dict, "emulate_sample", false);
        kwargs...,
    )
end

"""
    path_to_ensemble_member(output_dir, iteration, member)

Returns the path to an ensemble member's folder within the `output_dir`
for the given iteration and member number.
Internally runs `EnsembleKalmanProcess.TOMLInterface.path_to_ensemble_member`
"""
path_to_ensemble_member(output_dir, iteration, member) =
    EKP.TOMLInterface.path_to_ensemble_member(output_dir, iteration, member)

"""
    path_to_iteration(output_dir, iteration)
Returns the path to the iteration folder within the `output_dir` for the given iteration number.
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
    get_param_dict(distribution; names)

Generate a parameter dictionary for use in `EKP.TOMLInterface.save_parameter_ensemble`.
Assumes that all variables in the distribution are floating-point types.
"""
function get_param_dict(
    distribution::PD;
    names = distribution.name,
) where {PD <: ParameterDistributions.ParameterDistribution}
    return Dict(
        name => Dict{Any, Any}("type" => "float") for name in distribution.name
    )
end

"""
    save_G_ensemble(experiment_id, iteration, G_ensemble)

Save an ensemble's observation map output to the correct folder.
"""
function save_G_ensemble(filepath, iteration, G_ensemble)
    config = ExperimentConfig(filepath)
    return save_G_ensemble(config, iteration, G_ensemble)
end

function save_G_ensemble(config::ExperimentConfig, iteration, G_ensemble)
    iter_path = path_to_iteration(config.output_dir, iteration)
    JLD2.save_object(joinpath(iter_path, "G_ensemble.jld2"), G_ensemble)
    return G_ensemble
end

function env_experiment_dir(env = ENV)
    key = "EXPERIMENT_DIR"
    haskey(env, key) || error("Experiment dir not found in environment")
    return string(env[key])
end

function env_model_interface(env = ENV)
    key = "MODEL_INTERFACE"
    haskey(env, key) || error("Model interface file not found in environment")
    return string(env[key])
end

function env_iter_number(env = ENV)
    key = "ITER_NUMBER"
    haskey(env, key) || error("Iteration number not found in environment")
    return parse(Int, env[key])
end

function env_member_number(env = ENV)
    key = "MEMBER_NUMBER"
    haskey(env, key) || error("Member number not found in environment")
    return parse(Int, env[key])
end

"""
    initialize(config::ExperimentConfig; rng_seed = 1234)
    initialize(filepath::AbstractString; rng_seed = 1234)

Initializes the EKP object and the model ensemble.
"""
initialize(filepath::AbstractString; kwargs...) =
    initialize(ExperimentConfig(filepath); kwargs...)

function initialize(config::ExperimentConfig; rng_seed = 1234)
    Random.seed!(rng_seed)
    rng_ekp = Random.MersenneTwister(rng_seed)

    (; observations, ensemble_size, noise, prior, output_dir) = config
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

    param_dict = get_param_dict(prior)

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
    update_ensemble(config_file, iteration)
    update_ensemble(ExperimentConfig, iteration)

Updates the EKI object and saves parameters for the next iteration.
Assumes that the observation map has been run and saved in the current iteration folder.
"""
update_ensemble(config_file, iteration) =
    update_ensemble(ExperimentConfig(config_file), iteration)

function update_ensemble(configuration::ExperimentConfig, iteration)
    (; prior, output_dir) = configuration
    # Load EKI object from iteration folder
    iter_path = path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))

    # Load data from the ensemble
    G_ens = JLD2.load_object(joinpath(iter_path, "G_ensemble.jld2"))
    # Update
    EKP.update_ensemble!(eki, G_ens)
    iteration += 1

    param_dict = get_param_dict(prior)

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
This function requires the relevant experiment project and model interface to be loaded:

```julia
import CalibrateAtmos

experiment_id = "surface_fluxes_perfect_model"
experiment_path = joinpath(pkgdir(CalibrateAtmos), "experiments", experiment_id)
include(joinpath(pkgdir(CalibrateAtmos), "model_interface.jl"))
include(joinpath(experiment_path, "generate_data.jl"))

eki = CalibrateAtmos.calibrate(experiment_path)
```
"""
calibrate(experiment_path) = calibrate(ExperimentConfig(experiment_path))

function calibrate(configuration::ExperimentConfig)
    # initialize the CalibrateAtmos
    initialize(configuration)

    # run experiment with CalibrateAtmos for N_iter iterations
    (; n_iterations, id, ensemble_size) = configuration

    eki = nothing
    physical_model = get_forward_model(Val(Symbol(id)))
    for i in 0:(n_iterations - 1)
        @info "Running iteration $i"
        for m in 1:ensemble_size
            # model run for each ensemble member
            run_forward_model(
                physical_model,
                get_config(physical_model, m, i, configuration),
            )
            @info "Completed member $m"
        end

        # update EKP with the ensemble output and update calibrated parameters
        G_ensemble = observation_map(Val(Symbol(id)), i)
        save_G_ensemble(configuration, i, G_ensemble)
        eki = update_ensemble(configuration, i)
    end
    return eki
end

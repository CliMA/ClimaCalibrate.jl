import TOML, YAML
import JLD2
import Random
using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface

export ExperimentConfig, get_prior, initialize, update_ensemble, save_G_ensemble
export path_to_ensemble_member, path_to_model_log, path_to_iteration

"""
    ExperimentConfig(
        n_iterations::Integer,
        ensemble_size::Integer,
        observations,
        noise,
        prior::ParameterDistribution,
        output_dir,
    )
    ExperimentConfig(filepath::AbstractString; kwargs...)

Construct an ExperimentConfig from a given YAML file or directory containing 'experiment_config.yml'.

ExperimentConfig holds the configuration for a calibration experiment.
This can be constructed from a YAML configuration file or directly using individual parameters.
"""
Base.@kwdef struct ExperimentConfig
    n_iterations::Integer
    ensemble_size::Integer
    observations::Any
    noise::Any
    prior::ParameterDistribution
    output_dir::Any
end

function ExperimentConfig(filepath::AbstractString; kwargs...)
    is_yaml_file(f) = isfile(f) && endswith(f, ".yml")
    filepath_extension = joinpath(filepath, "experiment_config.yml")
    if is_yaml_file(filepath)
        config_dict = YAML.load_file(filepath)
        experiment_dir = dirname(filepath)
    elseif isdir(filepath) && is_yaml_file(filepath_extension)
        config_dict = YAML.load_file(filepath_extension)
        experiment_dir = filepath
    else
        error("Invalid experiment configuration filepath: `$filepath`")
    end

    default_output = joinpath(experiment_dir, "output")
    output_dir = get(config_dict, "output_dir", default_output)

    n_iterations = config_dict["n_iterations"]
    ensemble_size = config_dict["ensemble_size"]

    observation_path =
        isabspath(config_dict["observations"]) ? config_dict["observations"] :
        joinpath(experiment_dir, config_dict["observations"])
    observations = JLD2.load_object(observation_path)

    noise_path =
        isabspath(config_dict["noise"]) ? config_dict["noise"] :
        joinpath(experiment_dir, config_dict["noise"])
    noise = JLD2.load_object(noise_path)

    prior_path =
        isabspath(config_dict["prior"]) ? config_dict["prior"] :
        joinpath(experiment_dir, config_dict["prior"])
    prior = get_prior(prior_path)

    return ExperimentConfig(;
        n_iterations,
        ensemble_size,
        observations,
        noise,
        prior,
        output_dir,
        kwargs...,
    )
end

"""
    path_to_ensemble_member(output_dir, iteration, member)

Constructs the path to an ensemble member's directory for a given iteration and member number.
"""
path_to_ensemble_member(output_dir, iteration, member) =
    EKP.TOMLInterface.path_to_ensemble_member(output_dir, iteration, member)

"""
    path_to_model_log(output_dir, iteration, member)

Constructs the path to an ensemble member's forward model log for a given iteration and member number.
"""
path_to_model_log(output_dir, iteration, member) = joinpath(
    path_to_ensemble_member(output_dir, iteration, member),
    "model_log.txt",
)

"""
    path_to_iteration(output_dir, iteration)

Creates the path to the directory for a specific iteration within the specified output directory.
"""
path_to_iteration(output_dir, iteration) =
    joinpath(output_dir, join(["iteration", lpad(iteration, 3, "0")], "_"))

"""
    get_prior(param_dict::AbstractDict; names = nothing)
    get_prior(prior_path::AbstractString; names = nothing)

Constructs the combined prior distribution from a `param_dict` or a TOML configuration file specified by `prior_path`.
If `names` is provided, only those parameters are used.
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

Generates a dictionary for parameters based on the specified distribution, assumed to be of floating-point type.
If `names` is not provided, the distribution's names will be used.
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
    save_G_ensemble(config::ExperimentConfig, iteration, G_ensemble)
    save_G_ensemble(output_dir::AbstractString, iteration, G_ensemble)

Saves the ensemble's observation map output to the correct directory based on the provided configuration.
Takes an output directory, either extracted from an ExperimentConfig or passed directly.
"""
save_G_ensemble(config::ExperimentConfig, iteration, G_ensemble) =
    save_G_ensemble(config.output_dir, iteration, G_ensemble)

function save_G_ensemble(output_dir::AbstractString, iteration, G_ensemble)
    iter_path = path_to_iteration(output_dir, iteration)
    JLD2.save_object(joinpath(iter_path, "G_ensemble.jld2"), G_ensemble)
    return G_ensemble
end

function env_experiment_dir(env = ENV)
    key = "CALIBRATION_EXPERIMENT_DIR"
    haskey(env, key) || error(
        "Experiment dir not found in environment. Ensure that env variable \"CALIBRATION_EXPERIMENT_DIR\" is set.",
    )
    return string(env[key])
end

function env_model_interface(env = ENV)
    key = "CALIBRATION_MODEL_INTERFACE"
    haskey(env, key) || error(
        "Model interface file not found in environment. Ensure that env variable \"CALIBRATION_MODEL_INTERFACE\" is set.",
    )
    return string(env[key])
end

function env_iteration(env = ENV)
    key = "CALIBRATION_ITERATION"
    haskey(env, key) || error(
        "Iteration number not found in environment. Ensure that env variable \"CALIBRATION_ITERATION\" is set.",
    )
    return parse(Int, env[key])
end

function env_member_number(env = ENV)
    key = "CALIBRATION_MEMBER_NUMBER"
    haskey(env, key) || error(
        "Member number not found in environment. Ensure that env variable \"CALIBRATION_MEMBER_NUMBER\" is set.",
    )
    return parse(Int, env[key])
end

"""
    initialize(
        ensemble_size,
        observations,
        noise,
        prior,
        output_dir;
        rng_seed = 1234,
    )
    initialize(config::ExperimentConfig; rng_seed = 1234)
    initialize(filepath::AbstractString; rng_seed = 1234)

Initializes the calibration process by setting up the EnsembleKalmanProcess object
and parameter files with a given seed for random number generation.
"""
initialize(filepath::AbstractString; kwargs...) =
    initialize(ExperimentConfig(filepath); kwargs...)


initialize(config::ExperimentConfig; kwargs...) = initialize(
    config.ensemble_size,
    config.observations,
    config.noise,
    config.prior,
    config.output_dir;
    kwargs...,
)

function initialize(
    ensemble_size,
    observations,
    noise,
    prior,
    output_dir;
    rng_seed = 1234,
    ekp_kwargs...,
)
    Random.seed!(rng_seed)
    rng_ekp = Random.MersenneTwister(rng_seed)

    initial_ensemble =
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size)
    eki = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        observations,
        noise,
        EKP.Inversion();
        rng = rng_ekp,
        failure_handler_method = EKP.SampleSuccGauss(),
        ekp_kwargs...,
    )

    param_dict = get_param_dict(prior)

    save_parameter_ensemble(
        EKP.get_u_final(eki),  # constraints applied when saving
        prior,
        param_dict,
        output_dir,
        "parameters.toml",
        0,  # Initial iteration = 0
    )

    # Save the EKI object in the 'iteration_000' folder
    eki_path = joinpath(output_dir, "iteration_000", "eki_file.jld2")
    JLD2.save_object(eki_path, eki)
    return eki
end

"""
    update_ensemble(output_dir::AbstractString, iteration, prior)
    update_ensemble(config::ExperimentConfig, iteration)
    update_ensemble(config_file::AbstractString, iteration)

Updates the EnsembleKalmanProcess object and saves the parameters for the next iteration.
"""
update_ensemble(config_file::AbstractString, iteration) =
    update_ensemble(ExperimentConfig(config_file), iteration)

update_ensemble(configuration::ExperimentConfig, iteration) =
    update_ensemble(configuration.output_dir, iteration, configuration.prior)

function update_ensemble(output_dir::AbstractString, iteration, prior)
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

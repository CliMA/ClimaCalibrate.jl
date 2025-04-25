import TOML, YAML
import JLD2
import Random
using Distributions
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import EnsembleKalmanProcesses.TOMLInterface as TI

export ExperimentConfig, get_prior, initialize, update_ensemble, save_G_ensemble
export path_to_ensemble_member,
    path_to_model_log, path_to_iteration, parameter_path

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
    prior::PD.ParameterDistribution
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
    load_ekp_struct(output_dir, iteration)

Return the EnsembleKalmanProcess struct for a completed iteration.
"""
load_ekp_struct(output_dir, iteration) =
    JLD2.load_object(ekp_path(output_dir, iteration))

"""
    path_to_ensemble_member(output_dir, iteration, member)

Return the path to an ensemble member's directory for a given iteration and member number.
"""
path_to_ensemble_member(output_dir, iteration, member) =
    TI.path_to_ensemble_member(output_dir, iteration, member)

const DEFAULT_PARAMETER_FILE = "parameters.toml"
const DEFAULT_EKP_FILE = "eki_file.jld2"
const DEFAULT_G_ENSEMBLE = "G_ensemble.jld2"
const DEFAULT_CHECKPOINT_FILE = "checkpoint.txt"

"""
    checkpoint_path(output_dir, iteration, member)

Return the path to an ensemble member's checkpoint file.
"""
checkpoint_path(output_dir, iteration, member) = joinpath(
    path_to_ensemble_member(output_dir, iteration, member),
    DEFAULT_CHECKPOINT_FILE,
)

"""
    parameter_path(output_dir, iteration, member)

Return the path to an ensemble member's parameter file.
"""
parameter_path(output_dir, iteration, member) = joinpath(
    path_to_ensemble_member(output_dir, iteration, member),
    DEFAULT_PARAMETER_FILE,
)

"""
    ekp_path(output_dir, iteration)

Return the path to the serialized EnsembleKalmanProcess struct file for a given iteration.
"""
ekp_path(output_dir, iteration) =
    joinpath(path_to_iteration(output_dir, iteration), DEFAULT_EKP_FILE)

"""
    path_to_model_log(output_dir, iteration, member)

Return the path to an ensemble member's forward model log for a given iteration and member number.
"""
path_to_model_log(output_dir, iteration, member) = joinpath(
    path_to_ensemble_member(output_dir, iteration, member),
    "model_log.txt",
)

"""
    path_to_iteration(output_dir, iteration)

Return the path to the directory for a given iteration within the specified output directory.
"""
path_to_iteration(output_dir, iteration) =
    joinpath(output_dir, join(["iteration", lpad(iteration, 3, "0")], "_"))

path_to_G_ensemble(output_dir, iteration) =
    joinpath(path_to_iteration(output_dir, iteration), DEFAULT_G_ENSEMBLE)

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
    prior_vec = [TI.get_parameter_distribution(param_dict, n) for n in names]
    prior = PD.combine_distributions(prior_vec)
    return prior
end

"""
    get_param_dict(distribution; names)

Generates a dictionary for parameters based on the specified distribution, assumed to be of floating-point type.
If `names` is not provided, the distribution's names will be used.
"""
function get_param_dict(
    distribution::PDD;
    names = distribution.name,
) where {PDD <: PD.ParameterDistribution}
    return Dict(name => Dict{Any, Any}("type" => "float") for name in names)
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
    JLD2.save_object(path_to_G_ensemble(output_dir, iteration), G_ensemble)
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

write_model_completed(output_dir, iteration, member) =
    open(checkpoint_path(output_dir, iteration, member), "w") do io
        write(io, "completed")
    end

write_model_started(output_dir, iteration, member) =
    open(checkpoint_path(output_dir, iteration, member), "w") do io
        write(io, "started")
    end

function model_completed(output_dir, iteration, member)
    file = checkpoint_path(output_dir, iteration, member)
    !isfile(file) && return false
    status = readline(file)
    return status == "completed" ? true : false
end

function model_started(output_dir, iteration, member)
    file = checkpoint_path(output_dir, iteration, member)
    !isfile(file) && return false
    status = readline(file)
    return status == "started" ? true : false
end

"""
    ekp_constructor(ensemble_size, prior, observations, noise = nothing)

Initialize the EnsembleKalmanProcess object and parameter files.

Can take in an existing EnsembleKalmanProcess which will be used to generate the
 initial parameter ensemble.

Noise is optional when the observation is an EKP.ObservationSeries.

Additional kwargs may be passed through to the EnsembleKalmanProcess constructor.
"""

function ekp_constructor(
    ensemble_size,
    prior,
    observations,
    noise = nothing;
    rng_seed = 1234,
    ekp_kwargs...,
)

    Random.seed!(rng_seed)
    rng_ekp = Random.MersenneTwister(rng_seed)
    initial_ensemble =
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size)

    # EKP 2.0 and later require the `default_options_dict`
    eki_constr = if hasproperty(EKP, :default_options_dict)
        ekp_kwargs = Dict([string(k) => v for (k, v) in ekp_kwargs])
        (args...) -> EKP.EnsembleKalmanProcess(
            args...,
            merge(EKP.default_options_dict(EKP.Inversion()), ekp_kwargs);
            rng = rng_ekp,
        )
    else
        (args...) -> EKP.EnsembleKalmanProcess(
            args...;
            rng = rng_ekp,
            failure_handler_method = EKP.SampleSuccGauss(),
            ekp_kwargs...,
        )
    end

    eki = if isnothing(noise)
        eki_constr(initial_ensemble, observations, EKP.Inversion())
    else
        eki_constr(initial_ensemble, observations, noise, EKP.Inversion())
    end
    return eki
end

"""
    initialize(ensemble_size, observations, noise, prior, output_dir)
    initialize(eki::EKP.EnsembleKalmanProcess, prior, output_dir)
    initialize(config)

Initialize a calibration, saving the initial parameter ensemble to a folder within `output_dir`.

If no EKP struct is given, construct an EKP struct and return it.
"""
function initialize(config::ExperimentConfig; rng_seed = 1234, ekp_kwargs...)
    (; ensemble_size, observations, noise, prior, output_dir) = config
    return initialize(
        ensemble_size,
        observations,
        noise,
        prior,
        output_dir;
        ekp_kwargs...,
        rng_seed,
    )
end

function initialize(
    ensemble_size,
    observations,
    noise,
    prior,
    output_dir;
    rng_seed = 1234,
    ekp_kwargs...,
)
    eki = ekp_constructor(
        ensemble_size,
        prior,
        observations,
        noise;
        rng_seed,
        ekp_kwargs...,
    )
    save_eki_and_parameters(eki, output_dir, 0, prior)
    return eki
end

function initialize(eki::EKP.EnsembleKalmanProcess, prior, output_dir)
    save_eki_and_parameters(eki, output_dir, 0, prior)
    return eki
end

"""
    save_eki_and_parameters(eki, output_dir, iteration, prior)

Save EKI state and parameters. Helper function for [`initialize`](@ref) and [`update_ensemble`](@ref)
"""
function save_eki_and_parameters(eki, output_dir, iteration, prior)
    param_dict = get_param_dict(prior)
    TI.save_parameter_ensemble(
        EKP.get_u_final(eki),
        prior,
        param_dict,
        output_dir,
        DEFAULT_PARAMETER_FILE,
        iteration,
    )
    JLD2.save_object(ekp_path(output_dir, iteration), eki)
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

# TODO: test that this is working as intended
function update_ensemble(output_dir::AbstractString, iteration, prior)
    iter_path = path_to_iteration(output_dir, iteration)
    G_ens = JLD2.load_object(path_to_G_ensemble(output_dir, iteration))

    ekp = load_ekp_struct(output_dir, iteration)
    update_ensemble!(ekp, G_ens, output_dir, iteration, prior)
    return ekp
end

"""
    update_ensemble!(ekp, G_ens, output_dir, iteration, prior)

Updates an EKP object with data G_ens, saving the object and final parameters to disk.
"""
function update_ensemble!(ekp, G_ens, output_dir, iteration, prior)
    terminate = EKP.update_ensemble!(ekp, G_ens)
    save_eki_and_parameters(ekp, output_dir, iteration + 1, prior)
    return terminate
end

"""
    observation_map_and_update!(ekp, output_dir, iteration, prior)

Compute the observation map and update the given EKP object.
"""
function observation_map_and_update!(ekp, output_dir, iteration, prior)
    g_ensemble = observation_map(iteration)
    g_ensemble =
        postprocess_g_ensemble(ekp, g_ensemble, prior, output_dir, iteration)
    save_G_ensemble(output_dir, iteration, g_ensemble)
    terminate = update_ensemble!(ekp, g_ensemble, output_dir, iteration, prior)
    analyze_iteration(ekp, g_ensemble, prior, output_dir, iteration)
    return terminate
end

"""
    last_completed_iteration(output_dir)

Determines the last completed iteration given an `output_dir` containing a calibration run.

If no iteration has been completed yet, return -1.
"""
function last_completed_iteration(output_dir)
    last_completed_iter = -1
    while isfile(path_to_G_ensemble(output_dir, last_completed_iter + 1)) &&
        isfile(ekp_path(output_dir, last_completed_iter + 2))
        last_completed_iter += 1
    end
    return last_completed_iter
end

import TOML, YAML
import JLD2
import Random
using Distributions
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import EnsembleKalmanProcesses.TOMLInterface as TI

export get_prior, initialize, update_ensemble, save_G_ensemble
export path_to_ensemble_member,
    path_to_model_log, path_to_iteration, parameter_path, load_latest_ekp

"""
    load_ekp_struct(output_dir, iteration)

Return the EnsembleKalmanProcess struct for a completed iteration.
"""
load_ekp_struct(output_dir, iteration) =
    JLD2.load_object(ekp_path(output_dir, iteration))

"""
    load_latest_ekp(output_dir)

Return the most recent EnsembleKalmanProcess struct from the given output directory.

Returns nothing if no EKP structs are found.
"""
function load_latest_ekp(output_dir)
    iter = -1
    while isfile(ekp_path(output_dir, iter + 1))
        iter += 1
    end
    iter == -1 && return nothing
    return load_ekp_struct(output_dir, iter)
end

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
    save_G_ensemble(output_dir::AbstractString, iteration, G_ensemble)

Saves the ensemble's observation map output to the correct directory based on the provided configuration.
Takes an output directory, iteration number, and the ensemble output to save.
"""
function save_G_ensemble(output_dir::AbstractString, iteration, G_ensemble)
    iter_path = path_to_iteration(output_dir, iteration)
    JLD2.save_object(path_to_G_ensemble(output_dir, iteration), G_ensemble)
    return G_ensemble
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
    initialize(eki::EKP.EnsembleKalmanProcess, prior, output_dir)

Initialize a calibration, saving the initial parameter ensemble to a folder
within `output_dir`.
"""
function initialize(eki::EKP.EnsembleKalmanProcess, prior, output_dir)
    save_eki_and_parameters(eki, output_dir, 0, prior)
    JLD2.save_object(
        joinpath(path_to_iteration(output_dir, 0), "prior.jld2"),
        prior,
    )
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

Updates the EnsembleKalmanProcess object and saves the parameters for the next iteration.
"""
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
    try
        analyze_iteration(ekp, g_ensemble, prior, output_dir, iteration)
    catch ret_code
        @error "`analyze_iteration` crashed. See stacktrace" exception =
            (ret_code, catch_backtrace())
    end
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

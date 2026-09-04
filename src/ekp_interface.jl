import TOML
import JLD2
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import EnsembleKalmanProcesses.TOMLInterface as TI

export initialize,
    last_completed_iteration,
    terminated_iteration,
    save_G_ensemble,
    update_ensemble,
    update_ensemble!,
    observation_map_and_update!,
    get_prior,
    get_param_dict,
    path_to_iteration,
    path_to_ensemble_member,
    path_to_model_log,
    parameter_path,
    checkpoint_path,
    load_latest_ekp,
    load_ekp_struct,
    ekp_path,
    save_eki_and_parameters,
    model_started,
    model_completed,
    write_model_started,
    write_model_completed

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
    # Iterations are 1-indexed: `initialize` writes the first EKP struct to
    # iteration_001
    iter = 0
    while isfile(ekp_path(output_dir, iter + 1))
        iter += 1
    end
    iter == 0 && return nothing
    return load_ekp_struct(output_dir, iter)
end

"""
    path_to_ensemble_member(output_dir, iteration, member)

Return the path to an ensemble member's directory for a given iteration and
member number.

This is where a forward model should write its output.

# Examples
```julia
ClimaCalibrate.path_to_ensemble_member("output", 3, 7)
# "output/iteration_003/member_007"
```

See also [`parameter_path`](@ref), [`checkpoint_path`](@ref),
[`path_to_model_log`](@ref), [`path_to_iteration`](@ref).
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

ClimaCalibrate writes this file before the forward model runs. It is TOML in the
format ClimaParams.jl reads, so it can be parsed with `TOML.parsefile` or passed
straight to `ClimaParams.create_toml_dict`.

# Examples
```julia
ClimaCalibrate.parameter_path("output", 3, 7)
# "output/iteration_003/member_007/parameters.toml"
```
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

"""
    path_to_G_ensemble(output_dir, iteration)

Return the path to the saved G ensemble matrix for a given iteration.
"""
path_to_G_ensemble(output_dir, iteration) =
    joinpath(path_to_iteration(output_dir, iteration), DEFAULT_G_ENSEMBLE)

"""
    get_prior(param_dict::AbstractDict; names = nothing)
    get_prior(prior_path::AbstractString; names = nothing)

Construct the combined prior distribution from a `param_dict` or a TOML
configuration file specified by `prior_path`.

If `names` is provided, only those parameters are used.

# Examples
```julia
prior = ClimaCalibrate.get_prior("prior.toml")
subset = ClimaCalibrate.get_prior("prior.toml"; names = ["coefficient_a"])
```
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

Generate a dictionary for parameters based on the specified distribution, assumed to be of floating-point type.
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

Save the ensemble's observation map output to the correct directory.
Takes an output directory, iteration number, and the ensemble output to save.
"""
function save_G_ensemble(output_dir::AbstractString, iteration, G_ensemble)
    JLD2.save_object(path_to_G_ensemble(output_dir, iteration), G_ensemble)
    return G_ensemble
end

"""
    write_model_completed(output_dir, iteration, member)

Record that an ensemble member's forward model finished successfully, so that a
restart skips it.

The forward model itself calls this, which is why it is exported: an
`HPCBackend` job script runs it as its last statement.
"""
write_model_completed(output_dir, iteration, member) =
    _write_checkpoint(output_dir, iteration, member, "completed")

"""
    write_model_started(output_dir, iteration, member)

Record that an ensemble member's forward model is about to run.

The checkpoint is overwritten by [`write_model_completed`](@ref) once the model
finishes, so a member left in the "started" state is one that was interrupted.
"""
write_model_started(output_dir, iteration, member) =
    _write_checkpoint(output_dir, iteration, member, "started")

"""
    _write_checkpoint(output_dir, iteration, member, status)

Write `status` to an ensemble member's checkpoint file.

The member directory is created if it does not exist. `initialize` normally
creates it while writing the member's parameters, but a forward model that runs
outside that flow should not fail on a missing directory.
"""
function _write_checkpoint(output_dir, iteration, member, status)
    file = checkpoint_path(output_dir, iteration, member)
    mkpath(dirname(file))
    open(file, "w") do io
        write(io, status)
    end
end

"""
    model_completed(output_dir, iteration, member)

Return `true` if the ensemble member's forward model finished successfully.

Returns `false` when no checkpoint exists, which is also the case for a member
that has not been run yet.
"""
function model_completed(output_dir, iteration, member)
    return _checkpoint_status(output_dir, iteration, member) == "completed"
end

"""
    model_started(output_dir, iteration, member)

Return `true` if the ensemble member's forward model started but did not finish.

This is how an interrupted run is distinguished from one that never began.
"""
function model_started(output_dir, iteration, member)
    return _checkpoint_status(output_dir, iteration, member) == "started"
end

"""
    _checkpoint_status(output_dir, iteration, member)

Return the contents of an ensemble member's checkpoint file, or `nothing` if
there is no readable checkpoint.

A checkpoint that is being written while another process reads it can come back
truncated, which should not take down the submission loop.
"""
function _checkpoint_status(output_dir, iteration, member)
    file = checkpoint_path(output_dir, iteration, member)
    isfile(file) || return nothing
    return try
        readline(file)
    catch e
        @warn "Could not read the checkpoint at $file" exception = e
        nothing
    end
end

"""
    initialize(ekp::EKP.EnsembleKalmanProcess, prior, output_dir)

Initialize a calibration, saving the initial parameter ensemble to a folder
within `output_dir`.

If `output_dir` already holds a calibration, the stored first-iteration
`EnsembleKalmanProcess` is returned instead and nothing is written. Overwriting
it would replace the parameters that the completed forward models were run
with: `ekp` is typically rebuilt on restart, and unless the caller seeded
the RNG, its initial ensemble is a fresh random draw. The ensemble update
would then pair `G(u_old)` from the checkpointed members with `u_new`.
"""
function initialize(ekp::EKP.EnsembleKalmanProcess, prior, output_dir)
    if isfile(ekp_path(output_dir, 1))
        stored_ekp = load_ekp_struct(output_dir, 1)
        _check_restart_matches(stored_ekp, ekp, output_dir)
        @info "Resuming the calibration in $(abspath(output_dir)) with the \
               stored EnsembleKalmanProcess. Its scheduler, accelerator, and \
               localizer are the ones the first iteration was created with"
        return stored_ekp
    end
    save_eki_and_parameters(ekp, output_dir, 1, prior)
    JLD2.save_object(
        joinpath(path_to_iteration(output_dir, 1), "prior.jld2"),
        prior,
    )
    return ekp
end

"""
    _check_restart_matches(stored_ekp, ekp, output_dir)

Check that `ekp` describes the same calibration as the `stored_ekp` that a
restart will use in its place.

The stored process is the one the completed forward models were run with, so a
restart keeps it. That is only sound while the two agree on the ensemble size,
the observations, and the process: an ensemble update against observations the
user has since edited would give an answer for neither set.
"""
function _check_restart_matches(stored_ekp, ekp, output_dir)
    mismatch(what) = error(
        """Restarting from $(abspath(output_dir)), whose stored \
        EnsembleKalmanProcess has a different $what than the one given to \
        `calibrate`. The stored process is the one the completed iterations \
        were run with, so it is the one a restart uses. Use a fresh output \
        directory to calibrate with the new one.""",
    )

    EKP.get_N_ens(stored_ekp) == EKP.get_N_ens(ekp) ||
        mismatch("ensemble size ($(EKP.get_N_ens(stored_ekp)) vs \
                 $(EKP.get_N_ens(ekp)))")

    nameof(typeof(EKP.get_process(stored_ekp))) ==
    nameof(typeof(EKP.get_process(ekp))) || mismatch("process")

    # The whole series is compared, not the current minibatch: a minibatcher
    # that shuffles draws a different first minibatch each time the observation
    # is built
    _observation_values(stored_ekp) == _observation_values(ekp) ||
        mismatch("observation")

    # The covariances are compared by shape. A rebuilt estimator gives a
    # `SVDplusD` that holds the same numbers in a new object, and `==` on it
    # falls back to identity
    _covariance_shapes(stored_ekp) == _covariance_shapes(ekp) ||
        mismatch("observational noise covariance shape")
    return nothing
end

"""
    _observation_values(ekp)
    _covariance_shapes(ekp)

Return the samples of every observation in `ekp`'s series, and the shapes of
their covariance blocks.
"""
function _observation_values(ekp)
    observations = EKP.get_observations(EKP.get_observation_series(ekp))
    return [EKP.get_obs(obs) for obs in observations]
end

function _covariance_shapes(ekp)
    observations = EKP.get_observations(EKP.get_observation_series(ekp))
    return [
        size.(EKP.get_obs_noise_cov(obs; build = false)) for obs in observations
    ]
end

"""
    save_eki_and_parameters(ekp, output_dir, iteration, prior)

Save the `EnsembleKalmanProcess` state and each ensemble member's parameters for
`iteration`.

Helper for [`initialize`](@ref) and [`update_ensemble`](@ref).
"""
function save_eki_and_parameters(ekp, output_dir, iteration, prior)
    param_dict = get_param_dict(prior)
    TI.save_parameter_ensemble(
        EKP.get_u_final(ekp),
        prior,
        param_dict,
        output_dir,
        DEFAULT_PARAMETER_FILE,
        iteration,
    )
    JLD2.save_object(ekp_path(output_dir, iteration), ekp)
    return nothing
end

"""
    update_ensemble(output_dir::AbstractString, iteration, prior)

Update the EnsembleKalmanProcess object and save the parameters for the next iteration.
"""
function update_ensemble(output_dir::AbstractString, iteration, prior)
    G_ens = JLD2.load_object(path_to_G_ensemble(output_dir, iteration))

    ekp = load_ekp_struct(output_dir, iteration)
    update_ensemble!(ekp, G_ens, output_dir, iteration, prior)
    return ekp
end

"""
    update_ensemble!(ekp, G_ens, output_dir, iteration, prior)

Update an EKP object with data G_ens, saving the object and parameters for the next iteration to disk.
"""
function update_ensemble!(ekp, G_ens, output_dir, iteration, prior)
    terminate = EKP.update_ensemble!(ekp, G_ens)
    save_eki_and_parameters(ekp, output_dir, iteration + 1, prior)
    return terminate
end

"""
    observation_map_and_update!(
        ekp,
        output_dir,
        iteration,
        prior,
        interface,
    )

Compute the observation map and update the given EKP object.
"""
function observation_map_and_update!(
    ekp,
    output_dir,
    iteration,
    prior,
    interface,
)
    g_ensemble = ClimaCalibrate.observation_map(interface, iteration)
    g_ensemble = ClimaCalibrate.postprocess_g_ensemble(
        interface,
        ekp,
        g_ensemble,
        prior,
        output_dir,
        iteration,
    )
    save_G_ensemble(output_dir, iteration, g_ensemble)
    terminate = update_ensemble!(ekp, g_ensemble, output_dir, iteration, prior)
    try
        ClimaCalibrate.analyze_iteration(
            interface,
            ekp,
            g_ensemble,
            prior,
            output_dir,
            iteration,
        )
    catch ret_code
        @error "`analyze_iteration` crashed. See stacktrace" exception =
            (ret_code, catch_backtrace())
    end
    return terminate
end

"""
    terminated_path(output_dir)

Return the path of the file that records the iteration at which the
`EnsembleKalmanProcess` scheduler terminated the calibration.
"""
terminated_path(output_dir) = joinpath(output_dir, "terminated.txt")

"""
    write_terminated(output_dir, iteration)

Record that the `EnsembleKalmanProcess` scheduler terminated the calibration at
`iteration`.
"""
function write_terminated(output_dir, iteration)
    write(terminated_path(output_dir), string(iteration))
    return nothing
end

"""
    terminated_iteration(output_dir)

Return the iteration at which the `EnsembleKalmanProcess` scheduler terminated
the calibration in `output_dir`, or `nothing` if it ran to the end.

`update_ensemble!` saves the next iteration's `eki_file.jld2` whether or not the
scheduler terminated, and the ensemble it saves is the one that was already
there. [`last_completed_iteration`](@ref) alone would send a restart off to run
those same parameters again.

# Examples
```julia
julia> ClimaCalibrate.terminated_iteration(output_dir)
4
```
"""
function terminated_iteration(output_dir)
    path = terminated_path(output_dir)
    isfile(path) || return nothing
    return tryparse(Int, strip(read(path, String)))
end

"""
    last_completed_iteration(output_dir)

Return the last completed iteration of the calibration in `output_dir`, or 0 if
none has completed.

An iteration counts as complete once its `G_ensemble.jld2` exists *and* the next
iteration's `eki_file.jld2` has been written, i.e. once the ensemble update that
consumed it finished. [`calibrate`](@ref) resumes from the iteration after this
one.

# Examples
```julia
ClimaCalibrate.last_completed_iteration(output_dir)
```

See also [`load_latest_ekp`](@ref), [`model_completed`](@ref).
"""
function last_completed_iteration(output_dir)
    last_completed_iter = 0
    while isfile(path_to_G_ensemble(output_dir, last_completed_iter + 1)) &&
          isfile(ekp_path(output_dir, last_completed_iter + 2))
        last_completed_iter += 1
    end
    return last_completed_iter
end

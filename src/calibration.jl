"""
    ClimaCalibrate.Calibration

The calibration loop itself.

[`calibrate`](@ref) writes the ensemble members' parameters, runs the forward
model for each member on the chosen backend, evaluates the observation map, and
updates the ensemble, repeating until it runs out of iterations or EKP
terminates. What it writes goes under a single output directory, which is also
what it reads to resume an interrupted run.
"""
module Calibration

import ClimaCalibrate
import ..ClimaCalibrate: Backend, AbstractModelInterface
import ClimaCalibrate.Backend: HPCBackend, WorkerBackend, JuliaBackend

# Needed for interfacing with WorkerBackend
import Distributed

import EnsembleKalmanProcesses as EKP

export calibrate

include("ekp_interface.jl")

"""
    calibrate(
        backend,
        ekp::EKP.EnsembleKalmanProcess,
        interface::AbstractModelInterface,
        n_iterations,
        prior,
        output_dir,
    )

Run a full calibration with `ekp` and `prior` for `n_iterations` on the given
`backend`, storing the results of the calibration in `output_dir`.

The loop is the same for every backend: write the ensemble members' parameters,
run the forward model for each member, evaluate the observation map, and update
the ensemble. Only the middle step differs, and that is what the backend
selects. See [`JuliaBackend`](@ref), [`WorkerBackend`](@ref), and
[`HPCBackend`](@ref).

If `output_dir` already contains a calibration, it is resumed: completed
iterations are skipped, as are ensemble members that recorded a completed
forward model. Pass a fresh `output_dir` to start over.

# Returns
The `EnsembleKalmanProcess` at the end of the run.

# Examples
```julia
ekp = ClimaCalibrate.calibrate(
    ClimaCalibrate.JuliaBackend(),
    ekp,
    MyModelInterface(output_dir, ensemble_size),
    10,
    prior,
    output_dir,
)
```

See also [`initialize`](@ref), [`last_completed_iteration`](@ref).
"""
function calibrate(
    backend::Backend.AbstractBackend,
    ekp::EKP.EnsembleKalmanProcess,
    interface::AbstractModelInterface,
    n_iterations,
    prior,
    output_dir,
)
    output_dir = abspath(output_dir)

    ensemble_size = EKP.get_N_ens(ekp)
    @info "Initializing calibration" n_iterations ensemble_size output_dir
    ekp = initialize(ekp, prior, output_dir)
    initialize_backend!(backend, interface, output_dir)

    terminated_at = terminated_iteration(output_dir)
    if !isnothing(terminated_at)
        @info "The scheduler terminated this calibration at iteration \
               $terminated_at, so there is nothing left to run"
        return load_latest_ekp(output_dir)
    end

    first_iter = last_completed_iteration(output_dir) + 1
    if first_iter > n_iterations
        @info "$(first_iter - 1) iterations are already complete in \
               $output_dir, so there is nothing left to run"
        return load_latest_ekp(output_dir)
    end

    for iter in first_iter:n_iterations
        @info "Iteration $iter"
        run_iteration(backend, interface, iter, ensemble_size, output_dir)
        @info "Completed iteration $iter, updating ensemble"
        ekp = load_ekp_struct(output_dir, iter)
        terminate =
            observation_map_and_update!(ekp, output_dir, iter, prior, interface)
        if !isnothing(terminate)
            write_terminated(output_dir, iter)
            break
        end
    end
    return ekp
end

"""
    initialize_backend!(backend, interface::AbstractModelInterface, output_dir)

Prepare `backend` to run a calibration in `output_dir`.

Most backends need nothing beyond a usable output directory. The `HPCBackend`s
also have to serialize the model interface for their job scripts to load, and to
register the exit hook that cancels submitted jobs.
"""
function initialize_backend!(
    ::Backend.AbstractBackend,
    ::AbstractModelInterface,
    output_dir,
)
    return nothing
end

function initialize_backend!(
    backend::HPCBackend,
    interface::AbstractModelInterface,
    output_dir,
)
    experiment_dir = abspath(ClimaCalibrate.experiment_dir(interface))
    isdir(experiment_dir) || throw(
        ArgumentError("Experiment directory does not exist: $experiment_dir"),
    )

    model_interface_fp =
        abspath(ClimaCalibrate.model_interface_filepath(interface))
    isfile(model_interface_fp) || throw(
        ArgumentError(
            "Model interface file does not exist: $model_interface_fp",
        ),
    )

    # Each ensemble member's job runs in a fresh process, so it loads the
    # interface from here
    JLD2.save_object(joinpath(output_dir, "interface.jld2"), interface)

    # Killing the driver process would otherwise leave the submitted ensemble
    # members running and billing
    Backend.cancel_jobs_at_exit(backend)
    return nothing
end

"""
    check_failure_rate(n_failed, ensemble_size, backend, iter)

Halt the calibration if more than `failure_rate(backend)` of the iteration's
ensemble members failed.
"""
function check_failure_rate(n_failed, ensemble_size, backend, iter)
    allowed = Backend.failure_rate(backend)
    iter_failure_rate = n_failed / ensemble_size
    if iter_failure_rate > allowed
        error("Execution halted: iteration $iter had a \
              $(round(iter_failure_rate * 100; digits = 2))% failure rate \
              ($n_failed of $ensemble_size members), exceeding the maximum \
              allowed threshold of $(allowed * 100)%.")
    elseif n_failed > 0
        @warn "Iteration $iter had $n_failed failed ensemble member(s) out of \
               $ensemble_size"
    end
    return nothing
end

"""
    run_iteration(
        backend::HPCBackend,
        interface::AbstractModelInterface,
        iter,
        ensemble_size,
        output_dir,
    )

Run the `iter`th iteration by submitting one scheduler job per ensemble member
and waiting for each to finish, successfully or not.
"""
function run_iteration(
    backend::HPCBackend,
    interface::AbstractModelInterface,
    iter,
    ensemble_size,
    output_dir,
)
    experiment_dir = abspath(ClimaCalibrate.experiment_dir(interface))
    model_interface_filepath =
        abspath(ClimaCalibrate.model_interface_filepath(interface))
    exeflags = ClimaCalibrate.exeflags(interface)

    # For each ensemble member, generate the job script that will be run by the backends
    job_scripts = map(1:ensemble_size) do member
        generate_job_script_for_ensemble_member(
            backend,
            iter,
            member,
            output_dir,
            model_interface_filepath,
            experiment_dir,
            exeflags,
        )
    end

    # Submit the jobs (if they are not completed) and get all of the JobInfos
    jobs = map(1:ensemble_size) do member
        submit_calibration_job(
            backend,
            job_scripts[member],
            iter,
            member,
            output_dir,
        )
    end

    if all(isnothing.(jobs))
        # This should not be possible but manually deleting files in the output
        # directory could lead to this
        @info "All jobs for this iteration are already completed"
        return nothing
    end

    statuses = wait_for_jobs(
        jobs,
        output_dir,
        iter;
        job_timeout = Backend.job_timeout(backend),
    )
    n_failed = report_status(statuses, jobs, iter, output_dir)
    check_failure_rate(n_failed, ensemble_size, backend, iter)
    return nothing
end

"""
    generate_job_script_for_ensemble_member(
        backend::HPCBackend,
        iter,
        member,
        output_dir,
        model_interface_filepath,
        experiment_dir,
        exeflags,
    )

Generate a job script for the `member`th ensemble member for iteration `iter`
that will run on `backend`.
"""
function generate_job_script_for_ensemble_member(
    backend::HPCBackend,
    iter,
    member,
    output_dir,
    model_interface_filepath,
    experiment_dir,
    exeflags,
)
    # This script is executed by each ensemble member
    job_body = """
    import ClimaCalibrate
    iteration = $iter; member = $member
    model_interface_filepath = "$model_interface_filepath"
    include(model_interface_filepath)
    interface = ClimaCalibrate._load(joinpath("$output_dir", "interface.jld2"))
    ClimaCalibrate.forward_model(interface, iteration, member)
    ClimaCalibrate.write_model_completed("$output_dir", iteration, member)
    """

    julia_command = """
    julia --project=$experiment_dir $exeflags -e '$job_body'
    """

    member_log = path_to_model_log(output_dir, iter, member)
    scheduler_script = Backend.make_job_script(
        backend,
        julia_command;
        job_name = "run_$(iter)_$(member)",
        output = member_log,
    )
    return scheduler_script
end

"""
    submit_calibration_job(
        backend::HPCBackend,
        job_script,
        iter,
        member,
        output_dir,
    )

Submit a job that runs `job_script` to the `backend` and return the job info.

If the forward model is already completed (e.g. from a previous calibration
attempt), then `nothing` is returned instead.
"""
function submit_calibration_job(
    backend::HPCBackend,
    job_script,
    iter,
    member,
    output_dir,
)
    if model_completed(output_dir, iter, member)
        @info "Skipping completed member $member (found checkpoint)"
        return nothing
    elseif model_started(output_dir, iter, member)
        @info "Resuming member $member (incomplete run detected)"
    else
        @info "Running member $member"
    end

    write_model_started(output_dir, iter, member)

    job_info = Backend.submit_job(backend, job_script)
    return job_info
end

"""
    wait_for_jobs(
        jobs::Vector{T},
        output_dir,
        iter;
        job_timeout = Backend.JOB_TIMEOUT,
    ) where {T <: Union{Backend.JobInfo, Nothing}}

Wait for the `jobs` to run to completion and return their final statuses, with
`nothing` for the members that were already complete before this iteration
started.

Each job is queried once per poll, since a query shells out to the scheduler.
On PBS that means running `qstat`.

A job that has been running for longer than `job_timeout` seconds ends the
iteration: the remaining jobs are cancelled and an error is raised. PBS reports
an unreachable job as running, so a scheduler outage would otherwise block
forever. The clock starts when a job leaves the queue, so a long wait for an
allocation does not count against it.
"""
function wait_for_jobs(
    jobs::Vector{T},
    output_dir,
    iter;
    job_timeout = Backend.JOB_TIMEOUT,
) where {T <: Union{Backend.JobInfo, Nothing}}
    statuses = Vector{Union{Backend.JobStatus, Nothing}}(nothing, length(jobs))
    completed_jobs = Set{Int}()
    t_running = Dict{Int, Float64}()
    try
        while length(completed_jobs) < length(jobs)
            for (m, job) in enumerate(jobs)
                m in completed_jobs && continue

                if isnothing(job)
                    push!(completed_jobs, m)
                    continue
                end

                status = Backend.job_status(job)
                statuses[m] = status
                status == Backend.PENDING || get!(t_running, m, time())

                Backend.iscompleted(status) || continue

                if Backend.isfailed(status)
                    log_member_error(output_dir, iter, m)
                else
                    @info "Ensemble member $m complete"
                end
                # The exit hook cancels what this backend submitted, and a
                # finished job does not need a `scancel`
                Backend.mark_job_finished!(job)
                push!(completed_jobs, m)
            end

            if length(completed_jobs) < length(jobs)
                overdue = filter(
                    m -> time() - t_running[m] > job_timeout,
                    collect(keys(t_running)),
                )
                setdiff!(overdue, completed_jobs)
                if !isempty(overdue)
                    error("Members $(join(sort(overdue), ", ")) of iteration \
                          $iter have been running for more than \
                          $(job_timeout)s. Check that the scheduler is \
                          reachable, or raise `job_timeout` on the backend.")
                end
                sleep(5)
            end
        end
    catch e
        Backend.cancel_job.(filter(!isnothing, jobs))
        if !(e isa InterruptException)
            @error "Pipeline crashed outside of a model run. Stacktrace:" exception =
                (e, catch_backtrace())
        end
        # Rethrow so the caller aborts instead of running the observation map on
        # an ensemble that never finished
        rethrow(e)
    end

    return statuses
end

"""
    report_status(statuses, jobs::Vector, iter, output_dir)

Report the status of the iteration for the `jobs` that ran.

Return the number of failed members.

A member counts as failed if the scheduler said so *or* if it never wrote a
"completed" checkpoint. The checkpoint is the more reliable of the two: a
scheduler can lose the record of a job, and a batch script can exit successfully
even though the forward model inside it did not.
"""
function report_status(statuses, jobs::Vector, iter, output_dir)
    ran = findall(!isnothing, jobs)
    isempty(ran) && return 0

    failed_members = filter(ran) do m
        scheduler_failed =
            !isnothing(statuses[m]) && Backend.isfailed(statuses[m])
        return scheduler_failed || !model_completed(output_dir, iter, m)
    end

    if !isempty(failed_members)
        @warn "Failed ensemble members for iteration $iter: \
               $failed_members. See model logs in \
               $(abspath(path_to_iteration(output_dir, iter)))"
    end
    return length(failed_members)
end

"""
    log_member_error(output_dir, iteration, member)

Log a warning message when an error occurs, including the ensemble member's output.
"""
function log_member_error(output_dir, iteration, member)
    member_log = path_to_model_log(output_dir, iteration, member)
    warn_str = """Ensemble member $member raised an error. See model log at \
    $(abspath(member_log)) for stacktrace"""
    # A job the scheduler rejected outright never produces a log. Reading it
    # unconditionally would throw from inside the polling loop and be reported
    # as a crash of the calibration itself
    if isfile(member_log)
        stacktrace = replace(readchomp(member_log), "\\n" => "\n")
        warn_str = warn_str * ": \n$stacktrace"
    else
        warn_str = warn_str * ", but no log was written. The job may have been \
                   rejected by the scheduler before it started."
    end
    @warn warn_str
end

"""
    run_iteration(
        backend::WorkerBackend,
        interface::AbstractModelInterface,
        iter,
        ensemble_size,
        output_dir,
    )

Run the `iter`th iteration.

This function submits the work for a single ensemble member to each worker
and waits for each worker to complete (succeed or fail).
"""
function run_iteration(
    backend::WorkerBackend,
    interface::AbstractModelInterface,
    iter,
    ensemble_size,
    output_dir,
)
    isempty(backend.worker_pool.workers) &&
        @info "No workers currently available"

    # For each ensemble member, generate the work that the workers will do
    work_to_do = map(1:ensemble_size) do member
        prepare_work_for_ensemble_member(iter, member, output_dir, interface)
    end

    (; worker_pool) = backend
    nfailures = Base.Threads.Atomic{Int}(0)
    # Number of forward models currently running on checked-out workers. Those
    # workers are absent from the pool but will return when they finish, so an
    # empty pool with `inflight > 0` is not a stall
    inflight = Base.Threads.Atomic{Int}(0)
    # Track how long the pool has been empty *with nothing running* so an
    # asynchronous calibration does not hang forever if no workers ever start
    t_last_available = time()
    @sync while !isempty(work_to_do)
        if !isempty(worker_pool.workers)
            t_last_available = time()
            worker = take!(worker_pool)
            run_fwd_model = pop!(work_to_do)
            Base.Threads.atomic_add!(inflight, 1)
            @async try
                run_fwd_model(worker)
            catch e
                @warn "Error running on worker $worker" exception = e
                # Use atomic add because nfailures is accessed by multiple
                # workers
                Base.Threads.atomic_add!(nfailures, 1)
            finally
                Base.Threads.atomic_sub!(inflight, 1)
                push!(worker_pool, worker)
            end
        else
            # No workers in the pool. With asynchronous submission this is
            # expected early on. Only error if the pool stays empty with no
            # workers initializing, none running, and no progress for longer
            # than the backend's `empty_pool_timeout`. Reset the timer while
            # models are running or workers are initializing, since those will
            # replenish the pool.
            if inflight[] > 0 || Backend.n_initializing_workers() > 0
                t_last_available = time()
            end
            t_empty = time() - t_last_available
            if inflight[] == 0 &&
               Backend.n_initializing_workers() == 0 &&
               t_empty > backend.empty_pool_timeout
                error(
                    "No workers available for $(round(Int, t_empty))s \
                    (timeout $(backend.empty_pool_timeout)s) with no workers \
                    initializing or running. Ensure workers were submitted \
                    (e.g. with `add_workers`) and are able to start.",
                )
            end
            @debug "No workers available"
            sleep(10) # Wait for workers to become available
        end
    end

    check_failure_rate(nfailures[], ensemble_size, backend, iter)
    return nothing
end

"""
    prepare_work_for_ensemble_member(iter, member, output_dir, interface)

Return a function that takes in a worker and runs the forward model if needed.
"""
function prepare_work_for_ensemble_member(iter, member, output_dir, interface)
    return (worker) -> begin
        if model_completed(output_dir, iter, member)
            @info "Skipping completed member $member (found checkpoint)"
            return
        elseif model_started(output_dir, iter, member)
            @info "Resuming member $member on worker $worker (incomplete run detected)"
        else
            @info "Running member $member on worker $worker"
        end
        write_model_started(output_dir, iter, member)
        Distributed.remotecall_wait(
            ClimaCalibrate.forward_model,
            worker,
            interface,
            iter,
            member,
        )
        write_model_completed(output_dir, iter, member)
    end
end

"""
    run_iteration(
        backend::JuliaBackend,
        interface::AbstractModelInterface,
        iter,
        ensemble_size,
        output_dir,
    )

Run the `iter`th iteration by completing the work of all the ensemble members
sequentially.
"""
function run_iteration(
    backend::JuliaBackend,
    interface::AbstractModelInterface,
    iter,
    ensemble_size,
    output_dir,
)
    on_error(e::InterruptException) = rethrow(e)
    on_error(e) =
        @error "Single ensemble member has errored. See stacktrace" exception =
            (e, catch_backtrace())

    failures = 0
    foreach(1:ensemble_size) do m
        # Checkpoint like the other backends do, so an interrupted iteration
        # resumes where it stopped, and so an output directory produced here
        # can be resumed by any backend
        if model_completed(output_dir, iter, m)
            @info "Skipping completed member $m (found checkpoint)"
            return
        elseif model_started(output_dir, iter, m)
            @info "Resuming member $m (incomplete run detected)"
        end
        write_model_started(output_dir, iter, m)
        try
            ClimaCalibrate.forward_model(interface, iter, m)
            write_model_completed(output_dir, iter, m)
            @info "Completed member $m"
        catch e
            failures += 1
            on_error(e)
        end
    end
    check_failure_rate(failures, ensemble_size, backend, iter)
    return nothing
end

end

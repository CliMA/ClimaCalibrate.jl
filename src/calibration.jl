module Calibration

import ClimaCalibrate
import ..ClimaCalibrate: Backend
import ClimaCalibrate.Backend: HPCBackend, WorkerBackend, JuliaBackend

# Needed for interfacing with WorkerBackend
import Distributed

import EnsembleKalmanProcesses as EKP

include("ekp_interface.jl")

"""
    calibrate(
        backend::HPCBackend,
        ekp::EKP.EnsembleKalmanProcess,
        n_iterations,
        prior,
        output_dir,
        model_interface;
        experiment_dir = project_dir(),
        exeflags = "",
    )

Run a full calibration with `ekp` and `prior` for `n_iterations` on the given
`backend`, storing the results of the calibration in `output_dir`.

The work of each ensemble member which is running the forward model is done by
submitting a job to the `backend`. The `model_interface` file and project
directory `experiment_dir` should contain all the dependencies to run the
forward model. The job begins by running
`julia --project=\$experiment_dir -e 'include(\$model_interface)'` and running
the forward model.

For more information about the `HPCBackend`, see [`HPCBackend`](@ref).
"""
function calibrate(
    backend::HPCBackend,
    ekp::EKP.EnsembleKalmanProcess,
    n_iterations,
    prior,
    output_dir,
    model_interface;
    experiment_dir = project_dir(),
    exeflags = "",
)
    output_dir = abspath(output_dir)
    experiment_dir = abspath(experiment_dir)
    model_interface = abspath(model_interface)

    ensemble_size = EKP.get_N_ens(ekp)
    @info "Initializing calibration" n_iterations ensemble_size output_dir
    ekp = initialize(ekp, prior, output_dir)

    # Validation checks for output_dir, experiment_dir, and model_interface
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"
    @assert isdir(experiment_dir) "Experiment directory does not exist: $experiment_dir"
    @assert isfile(model_interface) "Model interface file does not exist: $model_interface"

    first_iter = last_completed_iteration(output_dir) + 1
    for iter in first_iter:(n_iterations - 1)
        @info "Iteration $iter"
        run_iteration(
            backend,
            iter,
            ensemble_size,
            output_dir,
            model_interface,
            experiment_dir,
            exeflags,
        )
        @info "Completed iteration $iter, updating ensemble"
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update!(ekp, output_dir, iter, prior)
        !isnothing(terminate) && break
    end
    return ekp
end

"""
    run_iteration(
        backend::HPCBackend,
        iter,
        ensemble_size,
        output_dir,
        model_interface,
        experiment_dir,
        exeflags,
    )

Run the `iter`th iteration.

This function makes a job script for the `HPCBackend`, submits each job, and
waits for each job to complete (succeed or fail).
"""
function run_iteration(
    backend::HPCBackend,
    iter,
    ensemble_size,
    output_dir,
    model_interface,
    experiment_dir,
    exeflags,
)
    # For each ensemble member, generate the job script that will be run by the backends
    job_scripts = map(1:ensemble_size) do member
        generate_job_script_for_ensemble_member(
            backend,
            iter,
            member,
            output_dir,
            model_interface,
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
    else
        wait_for_jobs(jobs, output_dir, iter)
        report_status(jobs, iter, output_dir)
    end
    return nothing
end

"""
    generate_job_script_for_ensemble_member(
        backend::HPCBackend,
        iter,
        member,
        output_dir,
        model_interface,
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
    model_interface,
    experiment_dir,
    exeflags,
)
    # This script is executed by each ensemble member
    job_body = """
    import ClimaCalibrate
    iteration = $iter; member = $member
    model_interface = "$model_interface"; include(model_interface)
    experiment_dir = "$experiment_dir"
    ClimaCalibrate.forward_model(iteration, member)
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

Submit a job that run `job_script` to the `backend` and return the job info.

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
        iter,
    ) where {T <: Union{Backend.JobInfo, Nothing}}

Wait for the `jobs` to run to completion.
"""
function wait_for_jobs(
    jobs::Vector{T},
    output_dir,
    iter,
) where {T <: Union{Backend.JobInfo, Nothing}}
    completed_jobs = Set{Int}()
    try
        while length(completed_jobs) < length(jobs)
            for (m, job) in enumerate(jobs)
                m in completed_jobs && continue

                if isnothing(job)
                    push!(completed_jobs, m)
                    continue
                end

                if Backend.isfailed(job)
                    log_member_error(output_dir, iter, m)
                    push!(completed_jobs, m)
                elseif Backend.issuccess(job)
                    @info "Ensemble member $m complete"
                    push!(completed_jobs, m)
                end
            end
            sleep(5)
        end
    catch e
        jobs = filter(!isnothing, jobs)
        Backend.cancel_job.(jobs)
        if !(e isa InterruptException)
            @error "Pipeline crashed outside of a model run. Stacktrace:" exception =
                (e, catch_backtrace())
        end
    end

    return nothing
end

"""
    report_status(jobs::Vector, iter, output_dir)

Report the status of the iteration for the `jobs` that ran.
"""
function report_status(jobs::Vector, iter, output_dir)
    # On Derecho, this is very slow because of repeated calls to job_status
    jobs = filter(!isnothing, jobs)
    if !all(Backend.iscompleted.(jobs))
        error(
            "Some jobs are not complete: $(filter(!Backend.iscompleted, jobs))",
        )
    end

    jobs_are_failing = Backend.isfailed.(jobs)
    if all(jobs_are_failing)
        error(
            """Full ensemble for iteration $iter has failed. See model logs in
$(abspath(path_to_iteration(output_dir, iter)))""",
        )
    elseif any(jobs_are_failing)
        @warn "Failed ensemble members: $(findall(Backend.isfailed, jobs))"
    end
    return nothing
end

"""
    log_member_error(output_dir, iteration, member)

Log a warning message when an error occurs.

If `verbose`, includes the ensemble member's output.
"""
function log_member_error(output_dir, iteration, member)
    member_log = path_to_model_log(output_dir, iteration, member)
    warn_str = """Ensemble member $member raised an error. See model log at \
    $(abspath(member_log)) for stacktrace"""
    stacktrace = replace(readchomp(member_log), "\\n" => "\n")
    warn_str = warn_str * ": \n$stacktrace"
    @warn warn_str
end

"""
    calibrate(
        backend::WorkerBackend,
        ekp::EKP.EnsembleKalmanProcess,
        n_iterations,
        prior,
        output_dir,
    )

Run a full calibration with `ekp` and `prior` for `n_iterations` on the given
`backend`, storing the results of the calibration in `output_dir`.

For more information about the `WorkerBackend`, see [`WorkerBackend`](@ref).
"""
function calibrate(
    backend::WorkerBackend,
    ekp::EKP.EnsembleKalmanProcess,
    n_iterations,
    prior,
    output_dir,
)
    output_dir = abspath(output_dir)

    ensemble_size = EKP.get_N_ens(ekp)
    @info "Initializing calibration" n_iterations ensemble_size output_dir
    ekp = initialize(ekp, prior, output_dir)

    # Validation checks for output_dir
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"

    first_iter = last_completed_iteration(output_dir) + 1
    for iter in first_iter:(n_iterations - 1)
        @info "Iteration $iter"
        run_iteration(backend, iter, ensemble_size, output_dir)
        @info "Completed iteration $iter, updating ensemble"
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update!(ekp, output_dir, iter, prior)
        !isnothing(terminate) && break
    end
    return ekp
end

"""
    run_iteration(backend::WorkerBackend, iter, ensemble_size, output_dir)

Run the `iter`th iteration.

This function submits the work for a single ensemble member to each worker
and waits for each worker to complete (succeed or fail).
"""
function run_iteration(backend::WorkerBackend, iter, ensemble_size, output_dir)
    isempty(backend.worker_pool.workers) &&
        @info "No workers currently available"

    # For each ensemble member, generate the work that the workers will do
    work_to_do = map(1:ensemble_size) do member
        prepare_work_for_ensemble_member(iter, member, output_dir)
    end

    (; worker_pool) = backend
    nfailures = 0
    @sync while !isempty(work_to_do)
        if !isempty(worker_pool.workers)
            worker = take!(worker_pool)
            run_fwd_model = pop!(work_to_do)
            @async try
                run_fwd_model(worker)
            catch e
                @warn "Error running on worker $worker" exception = e
                # TODO: Is this safe to do for multiple workers to try to
                # modify nfailures?
                nfailures += 1
            finally
                push!(worker_pool, worker)
            end
        else
            @debug "No workers available"
            sleep(10) # Wait for workers to become available
        end
    end

    iter_failure_rate = nfailures / ensemble_size
    (; failure_rate) = backend
    if iter_failure_rate > failure_rate
        throw(
            ErrorException(
                "Execution halted: Iteration $iter had a $(round(iter_failure_rate * 100; digits=2))% failure rate, exceeding the maximum allowed threshold of $(failure_rate * 100)%.",
            ),
        )
    end

    return nothing
end

"""
    prepare_work_for_ensemble_member(iter, member, output_dir)

Return a function that takes in a worker and run the forward model if needed.
"""
function prepare_work_for_ensemble_member(iter, member, output_dir)
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
            iter,
            member,
        )
        write_model_completed(output_dir, iter, member)
    end
end

"""
    calibrate(
        backend::JuliaBackend,
        ekp::EKP.EnsembleKalmanProcess,
        n_iterations,
        prior,
        output_dir,
    )

Run a full calibration with `ekp` and `prior` for `n_iterations` on the given
`backend`, storing the results of the calibration in `output_dir`.

Calibration with the `JuliaBackend` does not support restarts.

For more information about the `JuliaBackend`, see [`JuliaBackend`](@ref).
"""
function calibrate(
    backend::JuliaBackend,
    ekp::EKP.EnsembleKalmanProcess,
    n_iterations,
    prior,
    output_dir,
)
    output_dir = abspath(output_dir)

    ensemble_size = EKP.get_N_ens(ekp)
    @info "Initializing calibration" n_iterations ensemble_size output_dir
    ekp = initialize(ekp, prior, output_dir)

    # Validation checks for output_dir
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"

    first_iter = last_completed_iteration(output_dir) + 1
    for iter in first_iter:(n_iterations - 1)
        @info "Iteration $iter"
        run_iteration(backend, iter, ensemble_size, output_dir)
        @info "Completed iteration $iter, updating ensemble"
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update!(ekp, output_dir, iter, prior)
        !isnothing(terminate) && break
    end
    return ekp
end

"""
    run_iteration(backend::JuliaBackend, iter, ensemble_size, output_dir)

Run the `iter`th iteration by completing the work of all the ensemble members
sequentially.
"""
function run_iteration(::JuliaBackend, iter, ensemble_size, output_dir)
    on_error(e::InterruptException) = rethrow(e)
    on_error(e) =
        @error "Single ensemble member has errored. See stacktrace" exception =
            (e, catch_backtrace())

    failures = 0
    foreach(1:ensemble_size) do m
        try
            ClimaCalibrate.forward_model(iter, m)
            @info "Completed member $m"
        catch e
            failures += 1
            on_error(e)
        end
    end
    if failures == ensemble_size
        error("Full ensemble has failed, aborting calibration.")
    end
    return nothing
end

end

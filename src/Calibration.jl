module Calibration # TODO: Find better name

import ClimaCalibrate
import ..ClimaCalibrate: BetterBackend
import ClimaCalibrate.BetterBackend: HPCBackend, WorkerBackend, JuliaBackend

import Distributed

import EnsembleKalmanProcesses as EKP

include("ekp_interface.jl")

# TODO: For the input to calibrate, I think there is a way to
# unify the inputs to the different calibrate functions with
# something like CalibrationSetup

# TODO: I think this belongs in the developer documentation?
# The calibration functions dispatch on the `HPCBackend`, `WorkerBackend`, and
# `JuliaBackend`. The entry point to starting a calibration is `calibrate`.
# TODO: For experiment_dir, this could automatically be determined from
# Base.active_project
"""
    calibrate(
        backend::HPCBackend,
        ekp::EKP.EnsembleKalmanProcess,
        n_iterations,
        prior,
        output_dir,
        experiment_dir,
        model_interface,
    )

Run a full calibration with `ekp` and `prior` for `n_iterations` on the given
`backend`, storing the results of the calibration in `output_dir`.

The work of each ensemble member which is running the forward model is done by
submitting a job to the `backend`. The `model_interface` file and project
directory `experiment_dir` should contain all the dependencies to run the
forward model.

For more information about the `HPCBackend`, see TODO.
"""
function calibrate(
    backend::HPCBackend,
    ekp::EKP.EnsembleKalmanProcess,
    n_iterations,
    prior,
    # TODO: These arguments differ from the different backends
    output_dir,
    experiment_dir, # experiment_dir is julia --project=experiment_dir
    model_interface, # together, it is julia --project=experiment_dir model_interface
)
    output_dir = abspath(output_dir)
    experiment_dir = abspath(experiment_dir)
    model_interface = abspath(model_interface)

    ensemble_size = EKP.get_N_ens(ekp)
    @info "Initializing calibration" n_iterations ensemble_size output_dir
    # TODO: This should be 1-index instead of 0-index
    ekp = initialize(ekp, prior, output_dir)

    # Validation checks for output_dir, experiment_dir, and model_interface
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"
    @assert isdir(experiment_dir) "Experiment directory does not exist: $experiment_dir"
    @assert isfile(model_interface) "Model interface file does not exist: $model_interface"

    # TODO: Is this broken for the first iteration?
    # TODO: To replicate, I think you can add Main.@infiltrate and remove all
    # the iteration beside the first iteration
    first_iter = last_completed_iteration(output_dir) + 1
    for iter in first_iter:(n_iterations - 1)
        # TODO: Add functionality to keep track of how long each iteration take
        @info "Iteration $iter"
        run_iteration(
            backend,
            iter,
            ensemble_size,
            output_dir,
            experiment_dir,
            model_interface,
        )
        @info "Completed iteration $iter, updating ensemble"
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update2!(ekp, output_dir, iter, prior)
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
        experiment_dir,
        model_interface,
    )

Run the `iter`th iteration.

This is done by
1. making a job script for the `backend`,
2. submitting each job,
3. waiting for each job to complete (succeed or fail).
"""
function run_iteration(
    backend::HPCBackend,
    iter,
    ensemble_size,
    output_dir,
    experiment_dir,
    model_interface,
)
    # For each ensemble member, generate the job script that will be ran by the backends
    job_scripts = map(1:ensemble_size) do member
        generate_job_script_for_ensemble_member(
            backend,
            iter,
            member,
            output_dir,
            experiment_dir,
            model_interface,
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
    # TODO: This abstraction can be better since I can't filter the jobs array
    # but I don't have a way of denoting a completed job :(
    # nothing can be here since if a job is completed, then there is no reason
    # to submit a job so nothing is used as a placeholder
    # One appraoch is to define a calibration job type that include jobinfo
    # and the ensemble member
    # or pass them separately if I don't want separate types
    jobs = filter!(!isnothing, jobs)

    # Wait for the jobs to finish
    # TODO: This is needed because jobs can be empty. Not sure if this is the
    # approach though
    if isempty(jobs)
        @info "All jobs for this iteration are already completed"
    else
        wait_for_jobs!(jobs, output_dir, iter; reruns = 0)
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
        experiment_dir,
        model_interface, # TODO: Maybe add exeflags here?
    )

Generate a job script for the `member`th ensemble member for iteration `iter`
that will run on `backend`.
"""
function generate_job_script_for_ensemble_member(
    backend::HPCBackend,
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface, # TODO: Maybe add exeflags here?
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
    julia --project=$experiment_dir -e '$job_body'
    """

    member_log = path_to_model_log(output_dir, iter, member)
    scheduler_script = BetterBackend.make_job_script(
        backend,
        julia_command;
        job_name = "run_$(iter)_$(member)",
        output = member_log,
        log = member_log,
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
        # TODO: Not really running since you are submitting and waiting for the
        # job to run...
        @info "Running member $member"
    end

    write_model_started(output_dir, iter, member)

    job_info = BetterBackend.submit_job(backend, job_script)
    return job_info
end

"""
    wait_for_jobs!(
        jobs::Vector,
        output_dir,
        iter;
        reruns = 1,
        verbose = true,
    )

Wait for the `jobs` to run to completion.
"""
function wait_for_jobs!( # TODO: Cause problem if all members are completed...
    jobs::Vector,
    output_dir,
    iter;
    reruns = 1,
    verbose = true,
)
    rerun_job_count = zeros(length(jobs))
    # A completed job either finish to completion or ran out of reruns
    completed_jobs = Set{Int}()

    try
        while length(completed_jobs) < length(jobs)
            for (m, job) in enumerate(jobs)
                m in completed_jobs && continue

                if BetterBackend.isfailed(job)
                    log_member_error(output_dir, iter, m, verbose)
                    if rerun_job_count[m] < reruns

                        @info "Rerunning ensemble member $m"
                        jobs[m] = BetterBackend.requeue_job(job)
                        rerun_job_count[m] += 1
                    else
                        push!(completed_jobs, m)
                    end
                elseif BetterBackend.issuccess(job)
                    # TODO: This is wrong since jobs was filtered before this function was
                    # called :(
                    @info "Ensemble member $m complete"
                    push!(completed_jobs, m)
                end
            end
            sleep(5)
        end
    catch e
        BetterBackend.kill_job.(jobs)
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
    # TODO: This can be optimized by calling job_status once?
    # For Derecho, this is very slow :(
    if !all(BetterBackend.iscompleted.(jobs))
        error(
            "Some jobs are not complete: $(filter(BetterBackend.iscompleted, jobs))",
        )
    end

    jobs_are_failing = BetterBackend.isfailed.(jobs)
    # TODO: Maybe should make a struct called CalibrateJob for keeping track
    # of the ensemble member id since that is not stored anywhere...
    if all(jobs_are_failing)
        error(
            """Full ensemble for iteration $iter has failed. See model logs in
$(abspath(path_to_iteration(output_dir, iter)))""",
        )
    elseif any(jobs_are_failing)
        @warn "Failed ensemble members: $(findall(BetterBackend.isfailed, jobs))"
    end
    return nothing
end

function observation_map_and_update2!(ekp, output_dir, iteration, prior)
    g_ensemble = ClimaCalibrate.observation_map(iteration)
    g_ensemble = ClimaCalibrate.postprocess_g_ensemble(
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
    log_member_error(output_dir, iteration, member, verbose=false)

Log a warning message when an error occurs.

If `verbose`, includes the ensemble member's output.
"""
function log_member_error(output_dir, iteration, member, verbose = false)
    member_log = path_to_model_log(output_dir, iteration, member)
    warn_str = """Ensemble member $member raised an error. See model log at \
    $(abspath(member_log)) for stacktrace"""
    if verbose
        stacktrace = replace(readchomp(member_log), "\\n" => "\n")
        warn_str = warn_str * ": \n$stacktrace"
    end
    @warn warn_str
end

# TODO: experiment_dir is only needed for starting the calibration script
# model_interface might be included if we want to include a
# @everywhere include($model_interface) but not sure if this is ideal or if it
# is the responsibility of the user to do that

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

For more information about the `WorkerBackend`, see TODO.
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
    # TODO: This should be 1-index instead of 0-index
    ekp = initialize(ekp, prior, output_dir)

    # Validation checks for output_dir
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"

    first_iter = last_completed_iteration(output_dir) + 1
    for iter in first_iter:(n_iterations - 1)
        @info "Iteration $iter"
        run_iteration(backend, iter, ensemble_size, output_dir)
        @info "Completed iteration $iter, updating ensemble"
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update2!(ekp, output_dir, iter, prior)
        !isnothing(terminate) && break
    end
    return ekp
end

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

    # TODO: This is not identical to what we are doing for the HPCBackends
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
    # TODO: This should be 1-index instead of 0-index
    ekp = initialize(ekp, prior, output_dir)

    # Validation checks for output_dir
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"

    first_iter = last_completed_iteration(output_dir) + 1
    for iter in first_iter:(n_iterations - 1)
        @info "Iteration $iter"
        run_iteration(backend, iter, ensemble_size, output_dir)
        @info "Completed iteration $iter, updating ensemble"
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update2!(ekp, output_dir, iter, prior)
        !isnothing(terminate) && break
    end
    return ekp
end

"""
    run_iteration(backend::JuliaBackend, iter, ensemble_size, output_dir)
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

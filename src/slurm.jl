
export kwargs, slurm_model_run, wait_for_jobs

# Initial code is common to PBS and Slurm schedulers

"""
    kwargs(; kwargs...)

Create a dictionary from keyword arguments.
"""
kwargs(; kwargs...) = Dict{Symbol, Any}(kwargs...)

"""
    wait_for_jobs(jobids, output_dir, iter, experiment_dir, model_interface, module_load_str, model_run_func; verbose, hpc_kwargs, reruns=1)

Wait for a set of jobs to complete. If a job fails, it will be rerun up to `reruns` times.

This function monitors the status of multiple jobs and handles failures by rerunning the failed jobs up to the specified number of `reruns`. It logs errors and job completion status, ensuring all jobs are completed before proceeding.

Arguments:
- `jobids`: Vector of job IDs.
- `output_dir`: Directory for output.
- `iter`: Iteration number.
- `experiment_dir`: Directory for the experiment.
- `model_interface`: Interface to the model.
- `module_load_str`: Commands to load necessary modules.
- `model_run_func`: Function to run the model.
- `verbose`: Print detailed logs if true.
- `hpc_kwargs`: HPC job parameters.
- `reruns`: Number of times to rerun failed jobs.
"""
function wait_for_jobs(
    jobids::AbstractVector,
    output_dir,
    iter,
    experiment_dir,
    model_interface,
    module_load_str,
    model_run_func;
    verbose,
    hpc_kwargs,
    reruns = 1,
)
    rerun_job_count = zeros(length(jobids))
    completed_jobs = Set{Int}()

    try
        while length(completed_jobs) < length(jobids)
            for (m, jobid) in enumerate(jobids)
                m in completed_jobs && continue

                if job_failed(jobid)
                    log_member_error(output_dir, iter, m, verbose)
                    if rerun_job_count[m] < reruns

                        @info "Rerunning ensemble member $m"
                        jobids[m] = model_run_func(
                            iter,
                            m,
                            output_dir,
                            experiment_dir,
                            model_interface,
                            module_load_str;
                            hpc_kwargs,
                        )
                        rerun_job_count[m] += 1
                    else
                        push!(completed_jobs, m)
                    end
                elseif job_success(jobid)
                    @info "Ensemble member $m complete"
                    push!(completed_jobs, m)
                end
            end
            sleep(5)
        end
    catch e
        kill_job.(jobids)
        if !(e isa InterruptException)
            @error "Pipeline crashed outside of a model run. Stacktrace:" exception =
                (e, catch_backtrace())
        end
    end

    report_iteration_status(jobids, output_dir, iter)
    return map(job_status, jobids)
end

"""
    log_member_error(output_dir, iteration, member, verbose=false)

Log a warning message when an error occurs. If verbose, includes the ensemble member's output.
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

job_running(status::Symbol) = status == :RUNNING
job_success(status::Symbol) = status == :COMPLETED
job_failed(status::Symbol) = status == :FAILED
job_pending(status::Symbol) = status == :PENDING
job_completed(status::Symbol) = job_failed(status) || job_success(status)

job_pending(jobid) = job_pending(job_status(jobid))
job_running(jobid) = job_running(job_status(jobid))
job_success(jobid) = job_success(job_status(jobid))
job_failed(jobid) = job_failed(job_status(jobid))
job_completed(jobid) = job_completed(job_status(jobid))

"""
    report_iteration_status(jobids, output_dir, iter)

Report the status of an iteration. See also [`wait_for_jobs`](@ref).
"""
function report_iteration_status(jobids, output_dir, iter)
    if !all(job_completed.(jobids))
        error("Some jobs are not complete: $(filter(job_completed, jobids))")
    elseif all(job_failed, jobids)
        error(
            """Full ensemble for iteration $iter has failed. See model logs in
$(abspath(path_to_iteration(output_dir, iter)))""",
        )
    elseif any(job_failed, jobids)
        @warn "Failed ensemble members: $(findall(job_failed, jobids))"
    end
end

# Slurm-specific functions

"""
    submit_slurm_job(sbatch_filepath; env=deepcopy(ENV))

Submit a job to the Slurm scheduler using sbatch, removing unwanted environment variables.

Unset variables: "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE"
"""
function submit_slurm_job(sbatch_filepath; env = ENV)
    clean_env = deepcopy(env)
    # List of SLURM environment variables to unset
    unset_env_vars = [
        "SLURM_MEM_PER_CPU",
        "SLURM_MEM_PER_GPU",
        "SLURM_MEM_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_NTASKS",
        "SLURM_JOB_NAME",
        "SLURM_SUBMIT_DIR",
        "SLURM_JOB_ID",
    ]
    # Create a new environment without the SLURM variables
    for var in unset_env_vars
        delete!(clean_env, var)
    end

    try
        cmd = `sbatch --parsable $sbatch_filepath`
        output = readchomp(setenv(cmd, clean_env))
        # Parse job ID, handling potential format issues
        jobid = match(r"^\d+", output)
        if jobid === nothing
            error("Failed to parse job ID from output: $output")
        end

        return parse(Int, jobid.match)
    catch e
        error("Failed to submit SLURM job: $e")
    end
end

"""
    generate_sbatch_directives(hpc_kwargs)

Generate Slurm sbatch directives from HPC kwargs. 
"""
function generate_sbatch_directives(hpc_kwargs)
    @assert haskey(hpc_kwargs, :time) "Slurm kwargs must include key :time"

    hpc_kwargs[:time] = format_slurm_time(hpc_kwargs[:time])
    slurm_directives = map(collect(hpc_kwargs)) do (k, v)
        "#SBATCH --$(replace(string(k), "_" => "-"))=$(replace(string(v), "_" => "-"))"
    end
    return join(slurm_directives, "\n")
end
"""
    generate_sbatch_script(iter, member, output_dir, experiment_dir, model_interface; module_load_str, hpc_kwargs, exeflags="")
Generate a string containing an sbatch script to run the forward model.
`hpc_kwargs` is turned into a series of sbatch directives using [`generate_sbatch_directives`](@ref).
`module_load_str` is used to load the necessary modules and can be obtained via [`module_load_string`](@ref).
`exeflags` is a string of flags to pass to the Julia executable (defaults to empty string).
"""
function generate_sbatch_script(
    iter::Int,
    member::Int,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str,
    hpc_kwargs,
    exeflags = "",
)
    member_log = path_to_model_log(output_dir, iter, member)
    slurm_directives = generate_sbatch_directives(hpc_kwargs)
    ntasks = get(hpc_kwargs, :ntasks, 1)
    gpus_per_task = get(hpc_kwargs, :gpus_per_task, 0)
    climacomms_device = gpus_per_task > 0 ? "CUDA" : "CPU"
    # TODO: Remove this exception for GCP
    mpiexec_string =
        get_backend() == GCPBackend ? "mpiexec -n $ntasks" :
        "srun --output=$member_log --open-mode=append"
    sbatch_contents = """
    #!/bin/bash
    #SBATCH --job-name=run_$(iter)_$(member)
    #SBATCH --output=$member_log
    $slurm_directives

    $module_load_str
    export CLIMACOMMS_DEVICE="$climacomms_device"
    export CLIMACOMMS_CONTEXT="MPI"

    $mpiexec_string julia $exeflags --project=$experiment_dir -e '

        import ClimaCalibrate as CAL
        iteration = $iter; member = $member
        model_interface = "$model_interface"; include(model_interface)
        experiment_dir = "$experiment_dir"
        CAL.forward_model(iteration, member)
        CAL.write_model_completed("$output_dir", iteration, member)
    '
    exit 0
    """
    return sbatch_contents
end

"""
    slurm_model_run(iter, member, output_dir, experiment_dir, model_interface, module_load_str; hpc_kwargs)

Construct and execute a command to run a single forward model on Slurm.
Helper function for [`model_run`](@ref).
"""
function slurm_model_run(
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs = Dict{Symbol, Any}(
        :time => 45,
        :ntasks => 1,
        :cpus_per_task => 1,
    ),
    exeflags = "",
)
    # Type and existence checks
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"
    @assert isdir(experiment_dir) "Experiment directory does not exist: $experiment_dir"
    @assert isfile(model_interface) "Model interface file does not exist: $model_interface"

    # Range checks
    @assert iter >= 0 "Iteration number must be non-negative"
    @assert member > 0 "Member number must be positive"

    sbatch_contents = generate_sbatch_script(
        iter,
        member,
        output_dir,
        experiment_dir,
        model_interface,
        module_load_str,
        hpc_kwargs,
        exeflags,
    )

    jobid = mktemp(output_dir) do sbatch_filepath, io
        write(io, sbatch_contents)
        close(io)
        submit_slurm_job(sbatch_filepath)
    end
    return jobid
end

# Type alias for dispatching on PBSJobID (String) vs SlurmJobID (Int)
const SlurmJobID = Int

wait_for_jobs(
    jobids::AbstractVector{SlurmJobID},
    output_dir,
    iter,
    experiment_dir,
    model_interface,
    module_load_str;
    verbose,
    hpc_kwargs,
    reruns = 1,
) = wait_for_jobs(
    jobids,
    output_dir,
    iter,
    experiment_dir,
    model_interface,
    module_load_str,
    slurm_model_run;
    verbose = verbose,
    hpc_kwargs,
    reruns = reruns,
)

"""
    job_status(job_id)

Parse the slurm job_id's state and return one of three status symbols: :PENDING, :RUNNING, or :COMPLETED.
"""
function job_status(job_id::SlurmJobID)
    cmd = `squeue -j $job_id --format=%T --noheader`
    # Obtain stderr, difficult to do otherwise
    stdout = Pipe()
    stderr = Pipe()
    process = run(pipeline(ignorestatus(cmd), stdout = stdout, stderr = stderr))
    close(stdout.in)
    close(stderr.in)
    status = String(read(stdout))
    stderr = String(read(stderr))
    exit_code = process.exitcode

    # https://slurm.schedmd.com/job_state_codes.html
    pending_statuses = [
        "PENDING",
        "CONFIGURING",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
        "RESIZING",
    ]
    running_statuses =
        ["RUNNING", "COMPLETING", "STAGED", "SUSPENDED", "STOPPED", "RESIZING"]
    invalid_job_err = "slurm_load_jobs error: Invalid job id specified"
    @debug job_id status exit_code stderr

    status == "" && exit_code == 0 && stderr == "" && return :COMPLETED
    exit_code != 0 && contains(stderr, invalid_job_err) && return :COMPLETED

    any(str -> contains(status, str), pending_statuses) && return :PENDING
    any(str -> contains(status, str), running_statuses) && return :RUNNING

    @warn "Job ID $job_id has unknown status `$status`. Marking as completed"
    return :COMPLETED
end

"""
    kill_job(jobid::SlurmJobID)
    kill_job(jobid::PBSJobID)

End a running job, catching errors in case the job can not be ended.
"""
function kill_job(jobid::SlurmJobID)
    try
        run(`scancel $jobid`)
        println("Cancelling slurm job $jobid")
    catch e
        println("Failed to cancel slurm job $jobid: ", e)
    end
end

"Format `minutes` to a Slurm time string (D-HH:MM or HH:MM)"
function format_slurm_time(minutes::Int)
    days, remaining_minutes = divrem(minutes, (60 * 24))
    hours, remaining_minutes = divrem(remaining_minutes, 60)
    if days > 0
        return string(
            days,
            "-",
            lpad(hours, 2, '0'),
            ":",
            lpad(remaining_minutes, 2, '0'),
            ":00",
        )
    else
        return string(
            lpad(hours, 2, '0'),
            ":",
            lpad(remaining_minutes, 2, '0'),
            ":00",
        )
    end
end

format_slurm_time(str::AbstractString) = str

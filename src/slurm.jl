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
            sleep(10)
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
job_completed(status::Symbol) = job_failed(status) || job_success(status)

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
function submit_slurm_job(sbatch_filepath; env = deepcopy(ENV))
    # Ensure that we don't inherit unwanted environment variables
    unset_env_vars =
        ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE")
    for k in unset_env_vars
        haskey(env, k) && delete!(env, k)
    end
    jobid = readchomp(setenv(`sbatch --parsable $sbatch_filepath`, env))
    return parse(Int, jobid)
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
    generate_sbatch_script(iter, member, output_dir, experiment_dir, model_interface; module_load_str, hpc_kwargs)

Generate a string containing an sbatch script to run the forward model.

`hpc_kwargs` is turned into a series of sbatch directives using [`generate_sbatch_directives`](@ref).
`module_load_str` is used to load the necessary modules and can be obtained via [`module_load_string`](@ref).
"""
function generate_sbatch_script(
    iter::Int,
    member::Int,
    output_dir::AbstractString,
    experiment_dir::AbstractString,
    model_interface::AbstractString,
    module_load_str::AbstractString;
    hpc_kwargs,
)
    member_log = path_to_model_log(output_dir, iter, member)
    slurm_directives = generate_sbatch_directives(hpc_kwargs)
    gpus_per_task = get(hpc_kwargs, :gpus_per_task, 0)
    climacomms_device = gpus_per_task > 0 ? "CUDA" : "CPU"

    sbatch_contents = """
    #!/bin/bash
    #SBATCH --job-name=run_$(iter)_$(member)
    #SBATCH --output=$member_log
    $slurm_directives
    $module_load_str

    export CLIMACOMMS_DEVICE="$climacomms_device"
    export CLIMACOMMS_CONTEXT="MPI"

    srun --output=$member_log --open-mode=append julia --project=$experiment_dir -e '
    import ClimaCalibrate as CAL
    iteration = $iter; member = $member
    model_interface = "$model_interface"; include(model_interface)

    experiment_dir = "$experiment_dir"
    CAL.run_forward_model(CAL.set_up_forward_model(member, iteration, experiment_dir))'
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
        module_load_str;
        hpc_kwargs,
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
    job_status(jobid)

Parse the slurm jobid's state and return one of three status symbols: :COMPLETED, :FAILED, or :RUNNING.
"""
function job_status(jobid::SlurmJobID)
    failure_statuses = ("FAILED", "CANCELLED+", "CANCELLED")
    output = readchomp(`sacct -j $jobid --format=State --noheader`)
    # Jobs usually have multiple statuses
    statuses = strip.(split(output, "\n"))
    if all(s -> s == "COMPLETED", statuses)
        return :COMPLETED
    elseif any(s -> s in failure_statuses, statuses)
        return :FAILED
    else
        return :RUNNING
    end
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

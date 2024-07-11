export kwargs, sbatch_model_run, wait_for_jobs

kwargs(; kwargs...) = Dict{Symbol, Any}(kwargs...)

function generate_sbatch_directives(slurm_kwargs)
    @assert haskey(slurm_kwargs, :time) "Slurm kwargs must include key :time"

    slurm_kwargs[:time] = format_slurm_time(slurm_kwargs[:time])
    slurm_directives = map(collect(slurm_kwargs)) do (k, v)
        "#SBATCH --$(replace(string(k), "_" => "-"))=$(replace(string(v), "_" => "-"))"
    end
    return join(slurm_directives, "\n")
end

"""
    generate_sbatch_script(iter, member,
        output_dir, experiment_dir, model_interface;
        module_load_str, slurm_kwargs,
    )

Generate a string containing an sbatch script to run the forward model.

Helper function for `sbatch_model_run`.
"""
function generate_sbatch_script(
    iter::Int,
    member::Int,
    output_dir::AbstractString,
    experiment_dir::AbstractString,
    model_interface::AbstractString,
    module_load_str::AbstractString;
    slurm_kwargs,
)
    member_log = path_to_model_log(output_dir, iter, member)
    slurm_directives = generate_sbatch_directives(slurm_kwargs)

    sbatch_contents = """
    #!/bin/bash
    #SBATCH --job-name=run_$(iter)_$(member)
    #SBATCH --output=$member_log
    $slurm_directives
    set -euo pipefail
    $module_load_str

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
    sbatch_model_run(
        iter,
        member,
        output_dir,
        experiment_dir;
        model_interface,
        verbose;
        slurm_kwargs,
    )

Construct and execute a command to run a forward model on a Slurm cluster for a single ensemble member.

Arguments:
- iter: Iteration number
- member: Member number
- output_dir: Calibration experiment output directory
- experiment_dir: Directory containing the experiment's Project.toml
- model_interface: File containing the model interface
- module_load_str: Commands which load the necessary modules
- slurm_kwargs: Dictionary containing the slurm resources for the job. Easily generated using `kwargs`.
"""
function sbatch_model_run(
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    slurm_kwargs = Dict{Symbol, Any}(
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
        slurm_kwargs,
    )

    jobid = mktemp(output_dir) do sbatch_filepath, io
        write(io, sbatch_contents)
        close(io)
        submit_sbatch_job(sbatch_filepath)
    end
    return jobid
end

function wait_for_jobs(
    jobids::Vector{Int},
    output_dir,
    iter,
    experiment_dir,
    model_interface,
    module_load_str;
    verbose,
    slurm_kwargs,
    reruns = 1,
)
    statuses = map(job_status, jobids)
    rerun_job_count = zeros(length(jobids))
    completed_jobs = Set{Int}()

    try
        while length(completed_jobs) < length(statuses)
            for (m, status) in enumerate(statuses)
                m in completed_jobs && continue

                if job_failed(status)
                    log_member_error(output_dir, iter, m, verbose)
                    if rerun_job_count[m] < reruns

                        @info "Rerunning ensemble member $m"
                        jobids[m] = sbatch_model_run(
                            iter,
                            m,
                            output_dir,
                            experiment_dir,
                            model_interface,
                            module_load_str;
                            slurm_kwargs,
                        )
                        rerun_job_count[m] += 1
                    else
                        push!(completed_jobs, m)
                    end
                elseif job_success(status)
                    @info "Ensemble member $m complete"
                    push!(completed_jobs, m)
                end
            end
            sleep(5)
            statuses = map(job_status, jobids)
        end
        return statuses
    catch e
        kill_all_jobs(jobids)
        if !(e isa InterruptException)
            @error "Pipeline crashed outside of a model run. Stacktrace for failed simulation" exception =
                (e, catch_backtrace())
        end
        return map(job_status, jobids)
    end
end

"""
    log_member_error(output_dir, iteration, member, verbose = false)

Log a warning message when an error occurs.
If verbose, includes the ensemble member's output.
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

function report_iteration_status(statuses, output_dir, iter)
    all(job_completed.(statuses)) || error("Some jobs are not complete")

    if all(job_failed, statuses)
        error(
            """Full ensemble for iteration $iter has failed. See model logs in
            $(abspath(path_to_iteration(output_dir, iter)))""",
        )
    elseif any(job_failed, statuses)
        @warn "Failed ensemble members: $(findall(job_failed, statuses))"
    end
end

function submit_sbatch_job(sbatch_filepath; env = deepcopy(ENV))
    # Ensure that we don't inherit unwanted environment variables
    unset_env_vars =
        ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE")
    for k in unset_env_vars
        haskey(env, k) && delete!(env, k)
    end
    jobid = readchomp(setenv(`sbatch --parsable $sbatch_filepath`, env))
    return parse(Int, jobid)
end

job_running(status) = status == :RUNNING
job_success(status) = status == :COMPLETED
job_failed(status) = status == :FAILED
job_completed(status) = job_failed(status) || job_success(status)

"""
    job_status(jobid)

Parse the slurm jobid's state and return one of three status symbols: :COMPLETED, :FAILED, or :RUNNING.
"""
function job_status(jobid::Int)
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
    kill_all_jobs(jobids)

Takes a list of slurm job IDs and runs `scancel` on them.
"""
function kill_all_jobs(jobids)
    for jobid in jobids
        try
            kill_slurm_job(jobid)
            println("Cancelling slurm job $jobid")
        catch e
            println("Failed to cancel slurm job $jobid: ", e)
        end
    end
end

kill_slurm_job(jobid) = run(`scancel $jobid`)

function format_slurm_time(minutes::Int)
    days, remaining_minutes = divrem(minutes, (60 * 24))
    hours, remaining_minutes = divrem(remaining_minutes, 60)
    # Format the string according to Slurm's time format
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

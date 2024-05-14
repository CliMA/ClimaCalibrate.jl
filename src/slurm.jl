
kwargs(; kwargs...) = Dict{Symbol, Any}(kwargs...)

"""
    generate_sbatch_script


"""
function generate_sbatch_script(
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface;
    module_load = """
    export MODULEPATH=/groups/esm/modules:\$MODULEPATH
    module purge
    module load climacommon/2024_04_30
    """,
    slurm_kwargs = Dict{Symbol, Any}(
        :time => 45,
        :ntasks => 1,
        :cpus_per_task => 1,
    ),
)
    member_log = path_to_model_log(output_dir, iter, member)

    # Format time in minutes to string for slurm
    slurm_kwargs[:time] = format_slurm_time(slurm_kwargs[:time])

    slurm_directives = map(collect(slurm_kwargs)) do (k, v)
        "#SBATCH --$(replace(string(k), "_" => "-"))=$(replace(string(v), "_" => "-"))"
    end
    slurm_directives_str = join(slurm_directives, "\n")

    sbatch_contents = """
    #!/bin/bash
    #SBATCH --job-name=run_$(iter)_$(member)
    #SBATCH --output=$member_log
    $slurm_directives_str

    $module_load

    srun --output=$member_log --open-mode=append julia --project=$experiment_dir -e '
    import ClimaCalibrate as CAL
    iteration = $iter; member = $member
    model_interface = "$model_interface"; include(model_interface)

    experiment_dir = "$experiment_dir"
    CAL.run_forward_model(CAL.set_up_forward_model(member, iteration, experiment_dir))'
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

Construct and execute a command to run a model simulation on a Slurm cluster for a single ensemble member.
"""
function sbatch_model_run(
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface;
    slurm_kwargs = Dict{Symbol, Any}(),
    kwargs...,
)
    sbatch_contents = generate_sbatch_script(
        iter,
        member,
        output_dir,
        experiment_dir,
        model_interface;
        slurm_kwargs,
        kwargs...,
    )

    sbatch_filepath, io = mktemp(output_dir)
    write(io, sbatch_contents)
    close(io)

    return submit_sbatch_job(sbatch_filepath)
end

function wait_for_jobs(
    jobids,
    output_dir,
    iter,
    experiment_dir,
    model_interface;
    verbose,
    slurm_kwargs,
)
    statuses = map(job_status, jobids)
    rerun_jobs = Set{Int}()
    completed_jobs = Set{Int}()

    try
        while !all(job_completed, statuses)
            for (m, status) in enumerate(statuses)
                m in completed_jobs && continue

                if job_failed(status)
                    log_member_error(output_dir, iter, m, verbose)
                    if !(m in rerun_jobs)

                        @info "Rerunning ensemble member $m"
                        jobids[m] = sbatch_model_run(
                            iter,
                            m,
                            output_dir,
                            experiment_dir,
                            model_interface;
                            slurm_kwargs,
                        )
                        push!(rerun_jobs, m)
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

Log a warning message when an error occurs in a specific ensemble member during a model run in a Slurm environment. 
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

function submit_sbatch_job(sbatch_filepath; debug = false, env = ENV)
    jobid = readchomp(setenv(`sbatch --parsable $sbatch_filepath`, env))
    debug || rm(sbatch_filepath)
    return parse(Int, jobid)
end

job_running(status) = status == "RUNNING"
job_success(status) = status == "COMPLETED"
job_failed(status) = status == "FAILED"
job_completed(status) = job_failed(status) || job_success(status)

"""
    job_status(jobid)

Parse the slurm jobid's state and return one of three status strings: "COMPLETED", "FAILED", or "RUNNING"
"""
function job_status(jobid)
    failure_statuses = ("FAILED", "CANCELLED+", "CANCELLED")
    output = readchomp(`sacct -j $jobid --format=State --noheader`)
    # Jobs usually have multiple statuses
    statuses = strip.(split(output, "\n"))
    if all(s -> s == "COMPLETED", statuses)
        return "COMPLETED"
    elseif any(s -> s in failure_statuses, statuses)
        return "FAILED"
    else
        return "RUNNING"
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

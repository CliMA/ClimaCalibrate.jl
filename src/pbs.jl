function generate_pbs_directives(pbs_kwargs)
    @assert haskey(pbs_kwargs, :walltime) "PBS kwargs must include key :walltime"
    pbs_kwargs[:walltime] = format_time(pbs_kwargs[:walltime])

    pbs_directives = map(collect(pbs_kwargs)) do (k, v)
        "#PBS -$k=$v"
    end

    return join(pbs_directives, "\n")
end

function extract_pbs_values(pbs_directives::AbstractString)
    select_match = match(r"select=(\d+)", pbs_directives)
    ncpus_match = match(r"ncpus=(\d+)", pbs_directives)
    ngpus_match = match(r"ngpus=(\d+)", pbs_directives)
    
    num_nodes = select_match !== nothing ? parse(Int, select_match.captures[1]) : error("PBS directives must include 'select'")
    ncpus_per_node = ncpus_match !== nothing ? parse(Int, ncpus_match.captures[1]) : error("PBS directives must include 'ncpus'")
    gpus_per_node = ngpus_match !== nothing ? parse(Int, ngpus_match.captures[1]) : 0
    
    return num_nodes, ncpus_per_node, gpus_per_node
end

"""
    generate_script(
        iter, member,
        output_dir, experiment_dir, model_interface;
        module_load_str, pbs_kwargs,
    )

Generate a string containing a PBS script to run the forward model.

Helper function for `model_run`.
"""
function generate_pbs_script(
    iter::Int,
    member::Int,
    output_dir::AbstractString,
    experiment_dir::AbstractString,
    model_interface::AbstractString,
    module_load_str::AbstractString;
    pbs_kwargs...
)
    member_log = path_to_model_log(output_dir, iter, member)
    pbs_directives = generate_pbs_directives(pbs_kwargs)
    num_nodes, ncpus_per_node, gpus_per_node = extract_pbs_values(pbs_directives)
    total_procs = num_nodes * ncpus_per_node

    qsub_contents = """
    #!/bin/bash
    #PBS -N run_${iter}_${member}
    #PBS -j oe
    #PBS -A UCIT0011
    #PBS -q preempt
    #PBS -o $member_log
    $pbs_directives
    set -euo pipefail
    $module_load_str

    \$MPITRAMPOLINE_EXEC -n $total_procs -ppn $ncpus_per_node set_gpu_rank julia --project=$experiment_dir -e '
    import ClimaCalibrate as CAL
    iteration = $iter; member = $member
    model_interface = "$model_interface"; include(model_interface)

    experiment_dir = "$experiment_dir"
    CAL.run_forward_model(CAL.set_up_forward_model(member, iteration, experiment_dir))'

    exit 0
    """
    return qsub_contents
end

"""
    model_run(
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
function model_run(
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
    kwargs...,
)
    script_contents = generate_script(
        iter,
        member,
        output_dir,
        experiment_dir,
        model_interface,
        module_load_str;
        slurm_kwargs,
        kwargs...,
    )

    sbatch_filepath, io = mktemp(output_dir)
    write(io, script_contents)
    close(io)

    return submit_sbatch_job(sbatch_filepath)
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
                        jobids[m] = model_run(
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

function submit_job(filepath; debug = false, env = deepcopy(ENV))
    unset_env_vars =
        ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE")
    for k in unset_env_vars
        haskey(env, k) && delete!(env, k)
    end
    jobid = readchomp(setenv(`qsub $filepath`, env))
    return parse(Int, jobid)
end

job_running(status) = status == :RUNNING
job_success(status) = status == :COMPLETED
job_failed(status) = status == :FAILED
job_completed(status) = job_failed(status) || job_success(status)

"""
    job_status(jobid)

Parse the jobid's state and return one of three status symbols: :COMPLETED, :FAILED, or :RUNNING.
"""
function job_status(jobid)
    status_str = readchomp(`qstat -f $jobid -x -F dsv`)
    job_state_match = match(r"job_state=([^|]+)", status_str)
    status = first(job_state_match.captures)
    substate_match = match(r"substate=([^|]+)", status_str)
    substate_number = parse(Int, (first(substate_match.captures)))
    status_dict = Dict(
        "Q" => :RUNNING,
        "F" => :COMPLETED,
    )
    status_symbol = get(status_dict, status, :RUNNING)
    # Check for failure in the substate number
    if status_symbol == :COMPLETED && substate_number in (91, 93)
        status_symbol = :FAILED
    end
    return status_symbol
end

"""
    kill_all_jobs(jobids)

Cancels a list of PBS jobids.
"""
function kill_all_jobs(jobids)
    for jobid in jobids
        try
            kill_pbs_job(jobid)
            println("Cancelling PBS job $jobid")
        catch e
            println("Failed to cancel PBS job $jobid: ", e)
        end
    end
end

kill_pbs_job(jobid) = run(`qdel -W force $jobid`)

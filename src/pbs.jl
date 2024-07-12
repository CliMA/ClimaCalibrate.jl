"""
generate_pbs_script(
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
    pbs_kwargs = Dict()
)
    member_log = path_to_model_log(output_dir, iter, member)

    walltime = format_pbs_time(get(pbs_kwargs, :time, 45))
    num_nodes = get(pbs_kwargs, :ntasks, 1)
    ncpus_per_node = get(pbs_kwargs, :ncpus_per_task, 1)
    ngpus_per_node = get(pbs_kwargs, :ngpus_per_task, 0)
    queue = get(pbs_kwargs, :ngpus_per_task, "develop")

    if ngpus_per_node > 0
        ranks_per_node = ngpus_per_node
        set_gpu_rank = "set_gpu_rank"
    else
        ranks_per_node = ncpus_per_node
        set_gpu_rank = ""
    end
    total_ranks = num_nodes * ranks_per_node

    qsub_contents = """
    #!/bin/bash
    #PBS -N run_$(iter)_$member
    #PBS -j oe
    #PBS -A UCIT0011
    #PBS -q $queue
    #PBS -o $member_log
    #PBS -l walltime=$walltime
    #PBS -l select=$num_nodes:ncpus=$ncpus_per_node:ngpus=$ngpus_per_node
    set -euo pipefail
    $module_load_str
    \$MPITRAMPOLINE_MPIEXEC -n $total_ranks -ppn $ranks_per_node $set_gpu_rank julia --project=$experiment_dir -e '
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
    pbs_kwargs = Dict{Symbol, Any}(
        :walltime => 45,
        :select => "1:ncpus=1:ngpus=0",
    ),
)
    script_contents = generate_pbs_script(
        iter,
        member,
        output_dir,
        experiment_dir,
        model_interface,
        module_load_str;
        pbs_kwargs,
    )

    jobid = mktemp(output_dir) do filepath, io
        write(io, script_contents)
        close(io)
        submit_pbs_job(filepath)
    end

    return jobid
end

function wait_for_jobs(
    jobids::Vector{<:AbstractString},
    output_dir,
    iter,
    experiment_dir,
    model_interface,
    module_load_str;
    verbose,
    slurm_kwargs = nothing,
    pbs_kwargs = nothing,
    reruns = 1,
)
    @assert !isnothing(slurm_kwargs) || !isnothing(pbs_kwargs)
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
                        if isnothing(slurm_kwargs) 
                            jobids[m] = model_run(
                                iter,
                                m,
                                output_dir,
                                experiment_dir,
                                model_interface,
                                module_load_str;
                                pbs_kwargs,
                            )
                        else
                            jobids[m] = model_run(
                                iter,
                                m,
                                output_dir,
                                experiment_dir,
                                model_interface,
                                module_load_str;
                                slurm_kwargs,
                            )
                        end
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
        kill_job.(jobids)
        if !(e isa InterruptException)
            @error "Pipeline crashed outside of a model run. Stacktrace for failed simulation" exception =
                (e, catch_backtrace())
        end
        return map(job_status, jobids)
    end
end

function submit_pbs_job(filepath; debug = false, env = deepcopy(ENV))
    unset_env_vars =
        ("PBS_MEM_PER_CPU", "PBS_MEM_PER_GPU", "PBS_MEM_PER_NODE")
    for k in unset_env_vars
        haskey(env, k) && delete!(env, k)
    end
    jobid = readchomp(setenv(`qsub $filepath`, env))
    return jobid
end

"""
    job_status(jobid)

Parse the jobid's state and return one of three status symbols: :COMPLETED, :FAILED, or :RUNNING.
"""
function job_status(jobid::AbstractString)
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

function kill_job(jobid::AbstractString)
    try
        run(`qdel -W force $jobid`)
        println("Cancelling PBS job $jobid")

    catch e
        println("Failed to cancel PBS job $jobid: ", e)
    end

end

function format_pbs_time(minutes::Int)
    hours, remaining_minutes = divrem(minutes, 60)
    # Format the string according to Slurm's time format
    return string(
        lpad(hours, 2, '0'),
        ":",
        lpad(remaining_minutes, 2, '0'),
        ":00",
    )
end
format_pbs_time(str::AbstractString) = str

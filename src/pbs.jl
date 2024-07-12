struct PBS end

"""
generate_pbs_script(
        iter, member,
        output_dir, experiment_dir, model_interface;
        module_load_str, hpc_kwargs,
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
    hpc_kwargs = Dict(),
)
    member_log = path_to_model_log(output_dir, iter, member)

    walltime = format_pbs_time(get(hpc_kwargs, :time, 45))
    num_nodes = get(hpc_kwargs, :ntasks, 1)
    ncpus_per_node = get(hpc_kwargs, :ncpus_per_task, 1)
    ngpus_per_node = get(hpc_kwargs, :ngpus_per_task, 0)
    queue = get(hpc_kwargs, :ngpus_per_task, "develop")

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
        hpc_kwargs,
    )

Construct and execute a command to run a forward model on a Slurm cluster for a single ensemble member.

Arguments:
- iter: Iteration number
- member: Member number
- output_dir: Calibration experiment output directory
- experiment_dir: Directory containing the experiment's Project.toml
- model_interface: File containing the model interface
- module_load_str: Commands which load the necessary modules
- hpc_kwargs: Dictionary containing the slurm resources for the job. Easily generated using `kwargs`.
"""
function pbs_model_run(
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
)
    # Type and existence checks
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"
    @assert isdir(experiment_dir) "Experiment directory does not exist: $experiment_dir"
    @assert isfile(model_interface) "Model interface file does not exist: $model_interface"

    # Range checks
    @assert iter >= 0 "Iteration number must be non-negative"
    @assert member > 0 "Member number must be positive"

    script_contents = generate_pbs_script(
        iter,
        member,
        output_dir,
        experiment_dir,
        model_interface,
        module_load_str;
        hpc_kwargs,
    )

    jobid = mktemp(output_dir) do filepath, io
        write(io, script_contents)
        close(io)
        submit_pbs_job(filepath)
    end

    return jobid
end

function submit_pbs_job(filepath; debug = false, env = deepcopy(ENV))
    unset_env_vars = ("PBS_MEM_PER_CPU", "PBS_MEM_PER_GPU", "PBS_MEM_PER_NODE")
    for k in unset_env_vars
        haskey(env, k) && delete!(env, k)
    end
    jobid = readchomp(setenv(`qsub $filepath`, env))
    return jobid
end

const PBSJobID = AbstractString

wait_for_jobs(
    jobids::Vector{<:PBSJobID},
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
    pbs_model_run;
    verbose,
    hpc_kwargs,
    reruns,
)

"""
    job_status(jobid)

Parse the jobid's state and return one of three status symbols: :COMPLETED, :FAILED, or :RUNNING.
"""
function job_status(jobid::PBSJobID)
    status_str = readchomp(`qstat -f $jobid -x -F dsv`)
    job_state_match = match(r"job_state=([^|]+)", status_str)
    status = first(job_state_match.captures)
    substate_match = match(r"substate=([^|]+)", status_str)
    substate_number = parse(Int, (first(substate_match.captures)))
    status_dict = Dict("Q" => :RUNNING, "F" => :COMPLETED)
    status_symbol = get(status_dict, status, :RUNNING)
    # Check for failure in the substate number
    if status_symbol == :COMPLETED && substate_number in (91, 93)
        status_symbol = :FAILED
    end
    return status_symbol
end

function kill_job(jobid::PBSJobID)
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

"""
generate_pbs_script(
        iter, member,
        output_dir, experiment_dir, model_interface;
        module_load_str, hpc_kwargs,
    )

Generate a string containing a PBS script to run the forward model.

Returns:
- `qsub_contents::Function`: A function generating the content of the PBS script based on the provided arguments. 
    This will run the contents of the `julia_script`, which have to be run from a file due to Derecho's `set_gpu_rank`.
- `julia_script::String`: The Julia script string to be executed by the PBS job. 

Helper function for [`pbs_model_run`](@ref).
"""
function generate_pbs_script(
    iter::Int,
    member::Int,
    output_dir::AbstractString,
    experiment_dir::AbstractString,
    model_interface::AbstractString,
    module_load_str::AbstractString;
    hpc_kwargs = Dict(),
    exeflags = "",
)
    member_log = path_to_model_log(output_dir, iter, member)

    queue = get(hpc_kwargs, :queue, "main")
    walltime = format_pbs_time(get(hpc_kwargs, :time, 45))
    num_nodes = get(hpc_kwargs, :ntasks, 1)
    cpus_per_node = get(hpc_kwargs, :cpus_per_task, 1)
    gpus_per_node = get(hpc_kwargs, :gpus_per_task, 0)

    if gpus_per_node > 0
        ranks_per_node = gpus_per_node
        set_gpu_rank = "set_gpu_rank"
        climacomms_device = "CUDA"
    else
        ranks_per_node = cpus_per_node
        set_gpu_rank = ""
        climacomms_device = "CPU"
    end
    total_ranks = num_nodes * ranks_per_node

    # This must match the filepath in `pbs_model_run`
    member_path = path_to_ensemble_member(output_dir, iter, member)
    julia_filepath = joinpath(member_path, "model_run.jl")

    pbs_script = """\
    #!/bin/bash
    #PBS -N run_$(iter)_$member
    #PBS -j oe
    #PBS -A UCIT0011
    #PBS -q $queue
    #PBS -o $member_log
    #PBS -l walltime=$walltime
    #PBS -l select=$num_nodes:ncpus=$cpus_per_node:ngpus=$gpus_per_node:mpiprocs=$ranks_per_node

    $(pbs_trap_block())

    $module_load_str

    export JULIA_MPI_HAS_CUDA=true
    export CLIMACOMMS_DEVICE="$climacomms_device"
    export CLIMACOMMS_CONTEXT="MPI"
    \$MPITRAMPOLINE_MPIEXEC -n $total_ranks -ppn $ranks_per_node $set_gpu_rank julia $exeflags --project=$experiment_dir $julia_filepath
    """

    julia_script = """\
    import ClimaCalibrate as CAL
    include("$(abspath(model_interface))")
    CAL.forward_model($iter, $member)
    CAL.write_model_completed("$output_dir", $iter, $member)
    """
    return pbs_script, julia_script
end

"""
    pbs_trap_block()

Return the bash snippet that makes PBS jobs requeue on preemption or near-walltime signals.

This is included in the generated PBS scripts and is tested for presence to avoid
duplicating the job snippet in tests.
"""
function pbs_trap_block()
    return """
    # Self-requeue on preemption or near-walltime signals:
    # Trap SIGTERM on job termination and call `qrerun` to requeue the same job ID 
    # so it can continue later with the same submission parameters.
    # Exiting with status 0 prevents the scheduler from marking the job as failed
    # due to the trap.
    handle_preterminate() {
        sig="\$1"
        echo "[ClimaCalibrate] Received \$sig on PBS job \${PBS_JOBID:-unknown}, attempting qrerun"
        if command -v qrerun >/dev/null 2>&1; then
            qrerun "\${PBS_JOBID}"
        else
            echo "qrerun not available on this system"
        fi
        exit 0
    }
    trap 'handle_preterminate TERM' TERM
    """
end

"""
    pbs_model_run(iter, member, output_dir, experiment_dir, model_interface, module_load_str; hpc_kwargs)

Construct and execute a command to run a single forward model on PBS Pro.
Helper function for [`model_run`](@ref).
"""
function pbs_model_run(
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
    debug = false,
    exeflags = "",
)
    # Type and existence checks
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"
    @assert isdir(experiment_dir) "Experiment directory does not exist: $experiment_dir"
    @assert isfile(model_interface) "Model interface file does not exist: $model_interface"

    # Range checks
    @assert iter >= 0 "Iteration number must be non-negative"
    @assert member > 0 "Member number must be positive"

    pbs_script_contents, julia_script_contents = generate_pbs_script(
        iter,
        member,
        output_dir,
        experiment_dir,
        model_interface,
        module_load_str;
        hpc_kwargs,
        exeflags,
    )
    # TODO: Improve and figure out how to clean up the model_run.jl file
    # julia_filepath does not get cleaned up. `mktemp() do` syntax is ideal 
    # but would remove the script file before the PBS script actually executes
    member_path = path_to_ensemble_member(output_dir, iter, member)
    julia_filepath = joinpath(member_path, "model_run.jl")
    write(julia_filepath, julia_script_contents)
    jobid = mktemp(output_dir) do pbs_filepath, io
        write(io, pbs_script_contents)
        close(io)
        submit_pbs_job(pbs_filepath)
    end
    if debug
        println("PBS Script:")
        println(pbs_script_contents(julia_filepath))
        println("Julia Script:")
        println(julia_script_contents)
    end
    return jobid
end

"""
    submit_pbs_job(sbatch_filepath; env=deepcopy(ENV))

Submit a job to the PBS Pro scheduler using qsub, removing unwanted environment variables.

Unset variables: "PBS_MEM_PER_CPU", "PBS_MEM_PER_GPU", "PBS_MEM_PER_NODE", "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"
"""
function submit_pbs_job(filepath; debug = false, env = deepcopy(ENV))
    # Clean env to avoid user overrides breaking system PBS utilities (e.g., python wrappers)
    unset_env_vars = (
        "PBS_MEM_PER_CPU",
        "PBS_MEM_PER_GPU",
        "PBS_MEM_PER_NODE",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
    )
    for k in unset_env_vars
        haskey(env, k) && delete!(env, k)
    end
    env["PYTHONNOUSERSITE"] = "1"
    jobid = readchomp(setenv(`qsub $filepath`, env))
    return jobid
end

# Type alias for dispatching on PBSJobID (String) vs SlurmJobID (Int)
const PBSJobID = AbstractString

wait_for_jobs(
    jobids::AbstractVector{<:PBSJobID},
    output_dir,
    iter,
    experiment_dir,
    model_interface,
    module_load_str;
    verbose,
    hpc_kwargs,
    reruns = 0,
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
    _qstat_output(jobid, env; retries=2, delay=0.25)

Best-effort qstat caller: tries dsv then plain format, with a few short retries.
Returns the output String or `nothing` if all attempts fail.
"""
function _qstat_output(jobid::PBSJobID, env)
    attempts = 3
    delay = 0.25
    for i in 1:attempts
        try
            out = readchomp(setenv(`qstat -f $jobid -x -F dsv`, env))
            if isempty(strip(out)) && i < attempts
                sleep(delay)
                continue
            end
            return out
        catch
            try
                out = readchomp(setenv(`qstat -f $jobid -x`, env))
                if isempty(strip(out)) && i < attempts
                    sleep(delay)
                    continue
                end
                return out
            catch
                i < attempts && sleep(delay)
            end
        end
    end
    return nothing
end

function job_status(jobid::PBSJobID)
    # Call qstat with a sanitized environment to avoid user Python interfering with PBS wrappers
    clean_env = deepcopy(ENV)
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE")
        haskey(clean_env, k) && delete!(clean_env, k)
    end
    clean_env["PYTHONNOUSERSITE"] = "1"

    status_str = _qstat_output(jobid, clean_env)
    if isnothing(status_str)
        @warn "qstat failed for job $jobid; assuming job is running"
        return :RUNNING
    end

    # Support both dsv and plain formats
    job_state_match = match(r"job_state\s*=\s*([^|\n\r]+)", status_str)
    substate_match = match(r"substate\s*=\s*(\d+)", status_str)

    status_code = if isnothing(job_state_match)
        @warn "Job status for $jobid not found in qstat output. Assuming job is running"
        "Q"
    else
        strip(first(job_state_match.captures))
    end

    substate_number =
        isnothing(substate_match) ? 0 :
        parse(Int, first(substate_match.captures))

    # Map PBS states to our symbols; default to :RUNNING while job exists
    status_symbol = get(
        Dict("Q" => :RUNNING, "R" => :RUNNING, "F" => :COMPLETED),
        status_code,
        :RUNNING,
    )

    if status_symbol == :COMPLETED && substate_number in (91, 93)
        return :FAILED
    end
    return status_symbol
end

function kill_job(jobid::PBSJobID)
    try
        run(`qdel $jobid`)
        println("Cancelling PBS job $jobid")
    catch e
        println("Failed to cancel PBS job $jobid: ", e)
    end
end

"Format `minutes` to a PBS time string (HH:MM:SS)"
function format_pbs_time(minutes::Int)
    hours, remaining_minutes = divrem(minutes, 60)
    return string(
        lpad(hours, 2, '0'),
        ":",
        lpad(remaining_minutes, 2, '0'),
        ":00",
    )
end

format_pbs_time(str::AbstractString) = str

"""
    make_job_script(
        backend::DerechoBackend,
        job_body;
        job_name = "pbs_job.txt",
        output = "output.txt",
    )

Make a job script with `job_body` for the `backend`.

The job body must be a single Julia command.
"""
function make_job_script(
    backend::DerechoBackend,
    job_body;
    job_name = "pbs_job.txt",
    output = "output.txt",
)
    (; hpc_config) = backend
    (; directives, env_vars) = hpc_config

    directives = Dict(directives)
    queue = directives[:queue]
    walltime = directives[:time]
    num_nodes = directives[:ntasks]
    cpus_per_node = directives[:cpus_per_task]
    gpus_per_node = directives[:gpus_per_task]
    job_priority = directives[:job_priority]

    if gpus_per_node > 0
        ranks_per_node = gpus_per_node
        set_gpu_rank = "set_gpu_rank"
    else
        ranks_per_node = cpus_per_node
        set_gpu_rank = ""
    end
    total_ranks = num_nodes * ranks_per_node

    env_var_str = join(("export $k=\"$v\"" for (k, v) in env_vars), "\n")
    # Change directory before starting the Julia process because PBS defaults to
    # the home directory instead of the submission directory, unlike Slurm
    pbs_script = """
    #!/bin/bash
    #PBS -N $job_name
    #PBS -j oe
    #PBS -A UCIT0011
    #PBS -q $queue
    #PBS -o $output
    #PBS -l job_priority=$job_priority
    #PBS -l walltime=$walltime
    #PBS -l select=$num_nodes:ncpus=$cpus_per_node:ngpus=$gpus_per_node:mpiprocs=$ranks_per_node

    $(module_load_string(backend))

    $env_var_str

    cd \$PBS_O_WORKDIR
    \$MPITRAMPOLINE_MPIEXEC -n $total_ranks -ppn $ranks_per_node $set_gpu_rank $job_body
    """
    return pbs_script
end

"""
    submit_job(backend::DerechoBackend, job_script::String)

Submit a `job` that run `job_script` with `backend`.

The `job_script` should be generated with `make_job_script`.
"""
function submit_job(backend::DerechoBackend, job_script::String)
    output_dir = pwd()
    mktemp(output_dir) do pbs_filepath, io
        write(io, job_script)
        close(io)

        clean_env = deepcopy(ENV)
        # List of PBS environment variables to unset
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
            haskey(clean_env, k) && delete!(clean_env, k)
        end
        # Disable user-site packages directory to prevent issues with Derecho's
        # `qstat` python backend https://github.com/NCAR/qstat-cache
        clean_env["PYTHONNOUSERSITE"] = "1"

        try
            # Pass all environment variables from the submitting process
            # to the job using -V
            cmd = `qsub -V $pbs_filepath`
            # readchomp return a substring
            job_id = String(readchomp(setenv(cmd, clean_env)))
            # Parse job ID, handling potential format issues
            if isnothing(job_id)
                error("Failed to parse job ID from output: $job_id")
            end
            job_info = JobInfo(backend, job_id, job_script)
            push!(backend.job_records, job_info)
            return job_info
        catch e
            error("Failed to submit PBS job: $e")
        end
    end
end

const QBS_CODE_TO_JOB_STATUS =
    Dict("Q" => RUNNING, "R" => RUNNING, "F" => COMPLETED)

"""
    job_status(::DerechoBackend, job::JobInfo)

Return the status of `job`.

See [`JobStatus`](@ref).
"""
function job_status(::DerechoBackend, job::JobInfo)
    (; id) = job
    # Call qstat with a sanitized environment to avoid user Python interfering
    # with PBS wrappers
    clean_env = deepcopy(ENV)
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE")
        haskey(clean_env, k) && delete!(clean_env, k)
    end
    clean_env["PYTHONNOUSERSITE"] = "1"

    status_str = _qstat_output(id, clean_env)
    if isnothing(status_str)
        @warn "qstat failed for job $id; assuming job is running"
        return RUNNING
    end

    # Support both dsv and plain formats
    job_state_match = match(r"job_state\s*=\s*([^|\n\r]+)", status_str)
    substate_match = match(r"substate\s*=\s*(\d+)", status_str)

    status_code = if isnothing(job_state_match)
        @warn "Job status for $id not found in qstat output. Assuming job is running"
        return RUNNING
    else
        strip(first(job_state_match.captures))
    end

    substate_number =
        isnothing(substate_match) ? 0 :
        parse(Int, first(substate_match.captures))

    # Map PBS states to our symbols; default to RUNNING while job exists
    status_symbol = get(QBS_CODE_TO_JOB_STATUS, status_code, RUNNING)

    if status_symbol == COMPLETED && substate_number == 93
        return FAILED
    end
    return status_symbol
end

"""
    _qstat_output(jobid, env; attempts=3, delay=0.25)

Best-effort qstat caller: tries dsv then plain format, with a few short retries.
Returns the output String or `nothing` if all attempts fail.
"""
function _qstat_output(id::String, env; attempts = 3, delay = 0.25)
    # Try different qstat formats in order of preference
    qstat_commands = [`qstat -f $id -x -F dsv`, `qstat -f $id -x`]
    for i in 1:attempts
        for cmd in qstat_commands
            try
                out = readchomp(setenv(cmd, env))
                !isempty(strip(out)) && return out
            catch
                continue
            end
        end
        i < attempts && sleep(delay)
    end
    return nothing
end

"""
    cancel_job(::SlurmBackend, job::JobInfo)

Cancel `job` by running the command `qdel`.
"""
function cancel_job(::DerechoBackend, job::JobInfo)
    (; id) = job
    try
        run(`qdel $id`)
        println("Cancelling PBS job $id")
    catch e
        println("Failed to cancel PBS job $id: ", e)
    end
end

function module_load_string(backend::DerechoBackend)
    module_loads =
        join(("module load $m" for m in backend.hpc_config.modules), "\n")
    return """export MODULEPATH="/glade/campaign/univ/ucit0011/ClimaModules-Derecho:\$MODULEPATH"
    module purge
    $module_loads"""
end

"""
    format_pbs_time(minutes::Int)

Format `minutes` into a string (HH:MM:SS) accecpted by PBS.
"""
function format_pbs_time(minutes::Int)
    hours, remaining_minutes = divrem(minutes, 60)
    return string(
        lpad(hours, 2, '0'),
        ":",
        lpad(remaining_minutes, 2, '0'),
        ":00",
    )
end

"""
    format_pbs_time(str::AbstractString)

Return `str`.

This function does not validate whether `str` is correct or not.
"""
format_pbs_time(str::AbstractString) = str

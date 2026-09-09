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
    (; directives) = hpc_config

    directives = Dict(directives)
    num_nodes = directives[:ntasks]
    cpus_per_node = directives[:cpus_per_task]
    gpus_per_node = directives[:gpus_per_task]

    if gpus_per_node > 0
        ranks_per_node = gpus_per_node
        # Use a bash script to set GPU ranks for each process, needed so that 
        # MPI can use multiple GPUs concurrently
        set_gpu_rank = joinpath(@__DIR__, "set_gpu_rank.sh")
    else
        ranks_per_node = cpus_per_node
        set_gpu_rank = ""
    end
    total_ranks = num_nodes * ranks_per_node

    # Change directory before starting the Julia process because PBS defaults to
    # the home directory instead of the submission directory, unlike Slurm
    pbs_script = """
    #!/bin/bash
    $(generate_directives(hpc_config))
    #PBS -N $job_name
    #PBS -o $output

    $(module_load_string(backend))

    $(generate_env_vars(hpc_config))

    cd \$PBS_O_WORKDIR
    \$MPITRAMPOLINE_MPIEXEC -n $total_ranks -ppn $ranks_per_node $set_gpu_rank $job_body
    """
    return pbs_script
end

"""
    submit_job(backend::DerechoBackend, job_script::String)

Submit a `job` that runs `job_script` with `backend`.

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

        # Pass all environment variables from the submitting process to the
        # job using -V
        cmd = setenv(`qsub -V $pbs_filepath`, clean_env)
        job_id, stderr_text, exit_code = _run_capturing_output(cmd)
        if !iszero(exit_code) || isempty(job_id)
            error("qsub exited with $exit_code and returned no job id: \
                  $stderr_text")
        end
        job_info = JobInfo(backend, String(job_id), job_script)
        push!(backend.job_records, job_info)
        return job_info
    end
end

# The `job_state` codes and the `substate` numbers, including substate 93 for a
# job that exited with an error, are documented in the PBS Professional
# Reference Guide, in its tables of job attributes and job substates:
# https://help.altair.com/2022.1.0/PBS%20Professional/PBSReferenceGuide2022.1.pdf
const PBS_CODE_TO_JOB_STATUS = Dict(
    "Q" => PENDING,   # queued
    "H" => PENDING,   # held
    "W" => PENDING,   # waiting for its execution time
    "T" => PENDING,   # being moved to a new location
    "M" => PENDING,   # moved to another server
    "R" => RUNNING,
    "S" => RUNNING,   # suspended
    "B" => RUNNING,   # array job with at least one subjob running
    "E" => RUNNING,   # exiting after having run
    "X" => COMPLETED, # subjob finished
    "F" => COMPLETED,
)

"""
    _parse_pbs_state(status_str)

Map the `job_state` and `substate` reported by `qstat` to a [`JobStatus`](@ref).

Return `nothing` if `status_str` carries no `job_state`, so that the caller can
distinguish "PBS says nothing" from a state we understand.

PBS reports a finished job as `F` whether it succeeded or not; substate 93
is the one that means the job exited with an error.

# Examples
```julia
ClimaCalibrate.Backend._parse_pbs_state("job_state = F|substate = 93")  # FAILED
```
"""
function _parse_pbs_state(status_str)
    job_state_match = match(r"job_state\s*=\s*([^|\n\r]+)", status_str)
    isnothing(job_state_match) && return nothing
    status_code = strip(first(job_state_match.captures))
    status = get(PBS_CODE_TO_JOB_STATUS, status_code, nothing)
    isnothing(status) && return nothing

    substate_match = match(r"substate\s*=\s*(\d+)", status_str)
    substate_number =
        isnothing(substate_match) ? 0 :
        parse(Int, first(substate_match.captures))
    status == COMPLETED && substate_number == 93 && return FAILED
    return status
end

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
    # Derecho's `qstat` is NCAR's qstat-cache, which answers from a snapshot
    # refreshed every few seconds. Polling one job for its state is exactly
    # what a snapshot gets wrong: a job submitted since the last refresh is
    # "Unknown Job Id" (exit 153), and a job that has since finished is still
    # reported in whatever state the snapshot caught it in. Ask the scheduler
    clean_env["QSCACHE_BYPASS"] = "true"

    status_str, qstat_error = _qstat_output(id, clean_env)
    if isnothing(status_str)
        # Reporting RUNNING here keeps the calibration polling. That is right
        # for a transient qstat outage, but `wait_for_jobs` has to time out
        # eventually, or a permanent outage hangs the run forever
        @warn "qstat failed for job $id; assuming it still runs. \
               Reason: $qstat_error" maxlog = 5
        return RUNNING
    end

    status = _parse_pbs_state(status_str)
    if isnothing(status)
        @warn "Could not determine the state of job $id from qstat. Assuming \
               it is still running" maxlog = 5
        return RUNNING
    end
    return status
end

"""
    _qstat_output(id, env; attempts=3, delay=0.25)

Best-effort qstat caller: tries dsv then plain format, with a few short retries.

`-x` makes `qstat` report finished jobs alongside queued and running ones, so
one query covers every state a job can be in.

Return `(output, nothing)` on success and `(nothing, reason)` if all attempts
fail, so that the caller can report why qstat could not be reached rather than
silently treating the job as running.
"""
function _qstat_output(id::String, env; attempts = 3, delay = 0.25)
    # Try different qstat formats in order of preference
    qstat_commands = [`qstat -f $id -x -F dsv`, `qstat -f $id -x`]
    last_error = nothing
    for i in 1:attempts
        for cmd in qstat_commands
            out, err, exit_code = _run_capturing_output(setenv(cmd, env))
            iszero(exit_code) && !isempty(out) && return out, nothing
            last_error = isempty(err) ? "qstat exited with $exit_code" : err
        end
        i < attempts && sleep(delay)
    end
    return nothing, last_error
end

"""
    cancel_job(::DerechoBackend, job::JobInfo)

Cancel `job` by running the command `qdel`.
"""
function cancel_job(::DerechoBackend, job::JobInfo)
    (; id) = job
    @info "Cancelling PBS job $id"
    # `qdel` exits nonzero for a job that has already finished, which is not
    # something the caller can act on
    process = run(pipeline(ignorestatus(`qdel $id`); stderr = devnull))
    success(process) || @warn "qdel exited with $(process.exitcode) for job \
                               $id, which may already have finished"
    return nothing
end

function module_load_string(backend::DerechoBackend)
    module_loads = generate_modules(backend.hpc_config)
    return """module purge
    module use /glade/campaign/univ/ucit0011/ClimaModules-Derecho
    $module_loads
    module list 2>&1"""
end

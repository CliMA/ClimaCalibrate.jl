"""
    make_job_script(
        backend::SlurmBackend,
        job_body;
        job_name = "slurm_job",
        output = "output.txt",
    )

Make a job script with `job_body` for the `backend`.

The job body must be a single Julia command.
"""
function make_job_script(
    backend::SlurmBackend,
    job_body;
    job_name = "slurm_job",
    output = "output.txt",
)
    (; hpc_config) = backend
    (; directives) = hpc_config

    # Note that ntasks = 1 cannot be included as a default when making the
    # config because the cpus-per-task directive can change the default
    ntasks = get(directives, :ntasks, 1)
    mpiexec_string = _generate_mpiexec_string(backend, ntasks, output)

    slurm_script = """
    #!/bin/bash
    $(generate_directives(hpc_config))
    #SBATCH --job-name=$job_name
    #SBATCH --output=$output

    $(module_load_string(backend))
    $(generate_env_vars(hpc_config))

    $mpiexec_string $job_body
    """
    return slurm_script
end

"""
    submit_job(backend::SlurmBackend, job_script::String)

Submit a `job` that runs `job_script` with `backend`.

The `job_script` should be generated with `make_job_script`.
"""
function submit_job(backend::SlurmBackend, job_script::String)
    output_dir = pwd()
    mktemp(output_dir) do sbatch_filepath, io
        write(io, job_script)
        close(io)

        clean_env = deepcopy(ENV)
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
            job_id = match(r"^\d+", output)
            if isnothing(job_id)
                error("Failed to parse job ID from output: $output")
            end
            job_id = parse(Int, job_id.match)
            job_info = JobInfo(backend, job_id, job_script)
            push!(backend.job_records, job_info)
            return job_info
        catch e
            error("Failed to submit SLURM job: $e")
        end
    end
end

# https://slurm.schedmd.com/job_state_codes.html
const PENDING_STATUSES = Set([
    "PENDING",
    "CONFIGURING",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUED",
    "RESV_DEL_HOLD",
])
const RUNNING_STATUSES = Set([
    "RUNNING",
    "COMPLETING",
    "STAGE_OUT",
    "SIGNALING",
    "SUSPENDED",
    "STOPPED",
    "RESIZING",
])
const COMPLETED_STATUSES = Set(["COMPLETED"])
const FAILED_STATUSES = Set([
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "LAUNCH_FAILED",
])

"""
    _parse_slurm_state(output)

Map a job state reported by `sacct` or `squeue` to a [`JobStatus`](@ref).

Return `nothing` if `output` is empty or names a state we do not recognize, so
that the caller can decide what an unrecognized state means.

Slurm appends explanatory text to some states (`CANCELLED by 1234`), so only the
first word is matched.

# Examples
```julia
ClimaCalibrate.Backend._parse_slurm_state("CANCELLED by 40826")  # FAILED
```
"""
function _parse_slurm_state(output)
    tokens = split(strip(String(output)))
    isempty(tokens) && return nothing
    state = uppercase(first(tokens))
    state in COMPLETED_STATUSES && return COMPLETED
    state in FAILED_STATUSES && return FAILED
    state in RUNNING_STATUSES && return RUNNING
    state in PENDING_STATUSES && return PENDING
    return nothing
end

"""
    _sacct_state(id)

Return the state `sacct` reports for the job allocation `id`, or `nothing` if
`sacct` is unavailable or has no record of the job.

`-X` restricts the output to the allocation itself, so that the `.batch` and
`.extern` steps do not appear as extra lines.
"""
function _sacct_state(id)
    isnothing(Sys.which("sacct")) && return nothing
    cmd = `sacct -j $id -X --noheader --parsable2 --format=State`
    output = try
        readchomp(pipeline(ignorestatus(cmd); stderr = devnull))
    catch e
        @debug "sacct failed for job $id" exception = e
        return nothing
    end
    isempty(strip(output)) && return nothing
    return first(eachsplit(output, '\n'))
end

"""
    job_status(::SlurmBackend, job::JobInfo)

Return the status of `job`.

`squeue` only lists jobs that are still queued or running, so a job that has
left the queue is looked up with `sacct`, which is the only way to distinguish a
job that succeeded from one that failed, timed out, or was cancelled.

See [`JobStatus`](@ref).
"""
function job_status(::SlurmBackend, job::JobInfo)
    (; id) = job
    cmd = `squeue -j $id --format=%T --noheader`
    # Obtain stderr, difficult to do otherwise
    stdout = Pipe()
    stderr = Pipe()
    process = run(pipeline(ignorestatus(cmd), stdout = stdout, stderr = stderr))
    close(stdout.in)
    close(stderr.in)
    status = String(read(stdout))
    stderr = String(read(stderr))
    exit_code = process.exitcode

    @debug id status exit_code stderr

    # The job is still in the queue, so squeue is authoritative
    queued_status = _parse_slurm_state(status)
    if !isnothing(queued_status) && queued_status in (PENDING, RUNNING)
        return queued_status
    end

    # The job has left the queue. Only sacct can say whether it succeeded
    sacct_state = _sacct_state(id)
    if !isnothing(sacct_state)
        sacct_status = _parse_slurm_state(sacct_state)
        if isnothing(sacct_status)
            @warn "Job ID $id has unknown state `$(strip(sacct_state))`. \
                   Treating it as failed"
            return FAILED
        end
        return sacct_status
    end

    !isnothing(queued_status) && return queued_status

    # A squeue that could not reach the controller returns nothing at all,
    # which is indistinguishable from a job that has left the queue. The job
    # stays RUNNING so the loop keeps polling: a running member reported as
    # complete would be counted as a failure by the checkpoint cross-check in
    # `report_status`
    if exit_code != 0 && !occursin("Invalid job id", stderr)
        @warn "squeue failed for job $id with exit code $exit_code \
               ($(strip(stderr))). Treating the job as still running" maxlog = 1
        return RUNNING
    end

    # Neither squeue nor sacct knows about this job. Accounting may be disabled
    # or the record may have been purged; the calibration cross-checks the
    # member's checkpoint file, so report completion rather than blocking
    @warn "Neither squeue nor sacct has a record of job $id. Assuming it \
           finished; check the model log to see whether it succeeded" maxlog = 1
    return COMPLETED
end

"""
    cancel_job(::SlurmBackend, job::JobInfo)

Cancel `job` by running the command `scancel`.
"""
function cancel_job(::SlurmBackend, job::JobInfo)
    (; id) = job
    @info "Cancelling Slurm job $id"
    # `scancel` exits nonzero for a job that has already finished, which is not
    # something the caller can act on
    process = run(pipeline(ignorestatus(`scancel $id`); stderr = devnull))
    success(process) || @warn "scancel exited with $(process.exitcode) for \
                               job $id, which may already have finished"
    return nothing
end

"""
    _generate_mpiexec_string(backend, ntasks, output)

Return an mpiexec string: `mpiexec -n \$ntasks` for `GCPBackend`, or an `srun` string
logging to `output` for other backends.
"""
function _generate_mpiexec_string(backend, ntasks, output)
    # TODO: Remove this exception for GCPBackend
    return backend isa GCPBackend ? "mpiexec -n $ntasks" :
           "srun --output=$output --open-mode=append"
end

"""
    module_load_string(backend::HPCBackend)

Return a string that loads the correct modules for a given backend when executed
via bash.
"""
function module_load_string(backend::CaltechHPCBackend)
    module_loads = generate_modules(backend.hpc_config)
    return """export MODULEPATH="/resnick/groups/esm/modules:\$MODULEPATH"
    module purge
    $module_loads"""
end

function module_load_string(backend::ClimaGPUBackend)
    module_loads = generate_modules(backend.hpc_config)
    return """module purge
    $module_loads"""
end

function module_load_string(backend::GCPBackend)
    isempty(backend.hpc_config.modules) ||
        @warn "Loading modules is not supported by the backend. Not loading any modules specified by the backend"
    return """
    unset CUDA_ROOT
    unset NVHPC_CUDA_HOME
    unset CUDA_INC_DIR
    unset CPATH
    unset NVHPC_ROOT

    # NVHPC and HPC-X paths
    export NVHPC="/sw/nvhpc/Linux_x86_64/24.5"
    export HPCX_PATH="\${NVHPC}/comm_libs/12.4/hpcx/hpcx-2.19"

    # CUDA environment
    export CUDA_HOME="\${NVHPC}/cuda/12.4"
    export CUDA_PATH="\${CUDA_HOME}"
    export CUDA_ROOT="\${CUDA_HOME}"

    # MPI via MPIwrapper
    export MPITRAMPOLINE_LIB="/sw/mpiwrapper/lib/libmpiwrapper.so"
    export OPAL_PREFIX="\${HPCX_PATH}/ompi"

    # Library paths - CUDA first, then HPC-X
    export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${HPCX_PATH}/ompi/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

    # Executable paths
    export PATH="/sw/mpiwrapper/bin:\${CUDA_HOME}/bin:\${PATH}"
    export PATH="\${NVHPC}/profilers/Nsight_Systems/target-linux-x64:\${PATH}"

    # Julia
    export PATH="/sw/julia/julia-1.11.5/bin:\${PATH}"
    export JULIA_MPI_HAS_CUDA="true"
    """
end

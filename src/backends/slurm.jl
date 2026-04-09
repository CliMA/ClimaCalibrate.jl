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
    module_load_str = module_load_string(backend)
    (; hpc_kwargs) = backend
    slurm_directives = _generate_sbatch_directives(hpc_kwargs)
    ntasks = get(hpc_kwargs, :ntasks, 1)
    gpus_per_task = get(hpc_kwargs, :gpus_per_task, 0)
    climacomms_device = gpus_per_task > 0 ? "CUDA" : "CPU"
    mpiexec_string = _generate_mpiexec_string(backend, ntasks, output)

    slurm_script = """
    #!/bin/bash
    #SBATCH --job-name=$job_name
    #SBATCH --output=$output
    $slurm_directives

    $module_load_str
    export CLIMACOMMS_DEVICE="$climacomms_device"
    export CLIMACOMMS_CONTEXT="MPI"

    $mpiexec_string $job_body
    exit 0
    """
    return slurm_script
end

"""
    submit_job(backend::SlurmBackend, job_script::String)

Submit a `job` that run `job_script` with `backend`.

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
const PENDING_STATUSES = [
    "PENDING",
    "CONFIGURING",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUED",
    "RESIZING",
]
const RUNNING_STATUSES =
    ["RUNNING", "COMPLETING", "STAGED", "SUSPENDED", "STOPPED", "RESIZING"]

"""
    job_status(::SlurmBackend, job::JobInfo)

Return the status of `job`.

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

    invalid_job_err = "slurm_load_jobs error: Invalid job id specified"
    @debug job_id status exit_code stderr

    if status == "" && exit_code == 0 && stderr == ""
        return COMPLETED
    end
    if exit_code != 0 && contains(stderr, invalid_job_err)
        return COMPLETED
    end

    if any(str -> contains(status, str), PENDING_STATUSES)
        return PENDING
    end

    if any(str -> contains(status, str), RUNNING_STATUSES)
        return RUNNING
    end

    @warn "Job ID $job_id has unknown status `$status`. Marking as completed"
    return COMPLETED
end

"""
    cancel_job(::SlurmBackend, job::JobInfo)

Cancel `job` by running the command `scancel`.
"""
function cancel_job(::SlurmBackend, job::JobInfo)
    (; id) = job
    try
        run(`scancel $id`)
        println("Cancelling slurm job $id")
        return nothing
    catch e
        println("Failed to cancel slurm job $id: ", e)
        return nothing
    end
end

"""
    generate_sbatch_directives(hpc_kwargs)

Generate Slurm sbatch directives from HPC kwargs.
"""
function _generate_sbatch_directives(hpc_kwargs)
    @assert haskey(hpc_kwargs, :time) "Slurm kwargs must include key :time"

    hpc_kwargs[:time] = format_slurm_time(hpc_kwargs[:time])
    slurm_directives = map(collect(hpc_kwargs)) do (k, v)
        "#SBATCH --$(replace(string(k), "_" => "-"))=$(replace(string(v), "_" => "-"))"
    end
    return join(slurm_directives, "\n")
end

"""
    _generate_mpiexec_string(backend, ntasks, output)

Modify the job body to log to `output` or run with MPI with `ntasks` depending
on the `backend`.
"""
function _generate_mpiexec_string(backend, ntasks, output)
    # TODO: Remove this exception for GCPBackend
    return backend isa GCPBackend ? "mpiexec -n $ntasks" :
           "srun --output=$output --open-mode=append"
end

"""
    format_slurm_time(minutes::Int)

Format `minutes` into a string accecpted by slurm.
"""
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

"""
    format_slurm_time(str::AbstractString)

Return `str`.

This function does not validate whether `str` is correct or not.
"""
format_slurm_time(str::AbstractString) = str


"""
    module_load_string(backend::HPCBackend)

Return a string that loads the correct modules for a given backend when executed
via bash.
"""
function module_load_string(::CaltechHPCBackend)
    return """export MODULEPATH="/resnick/groups/esm/modules:\$MODULEPATH"
    module purge
    module load climacommon/2025_03_18"""
end

function module_load_string(::ClimaGPUBackend)
    return """module purge
    module load climacommon/2026_02_18"""
end

function module_load_string(::GCPBackend)
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

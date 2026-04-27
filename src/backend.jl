module Backend

import Distributed

export HPCBackend,
    SlurmBackend,
    JuliaBackend,
    WorkerBackend,
    DerechoBackend,
    GCPBackend,
    ClimaGPUBackend,
    CaltechHPCBackend,
    get_backend,
    JobInfo,
    JobStatus,
    job_status,
    ispending,
    isrunning,
    issuccess,
    isfailed,
    iscompleted,
    submit_job,
    requeue_job,
    cancel_job,
    make_job_script

include("backends/config.jl")

abstract type AbstractBackend end

"""
    JobStatus

An enum representing the current status of a job.

## Values
- `PENDING`: The job is queued and waiting to be scheduled.
- `RUNNING`: The job is currently executing.
- `COMPLETED`: The job finished running.
- `FAILED`: The job terminated with an error as reported by the scheduler.

Use [`ispending`](@ref), [`isrunning`](@ref), [`issuccess`](@ref),
[`isfailed`](@ref), and [`iscompleted`](@ref) to query the status of a
[`JobInfo`](@ref).

See also: [`job_status`](@ref).
"""
@enum JobStatus begin
    PENDING
    RUNNING
    COMPLETED
    FAILED
end

"""
    JuliaBackend

The simplest backend to use.

This is a singleton type and is meant for use in dispatch.
"""
struct JuliaBackend <: AbstractBackend end

"""
    DEFAULT_FAILURE_RATE

The default failure rate used by the `WorkerBackend`.
"""
const DEFAULT_FAILURE_RATE = 0.5

"""
    WorkerBackend

Used to run calibrations on Distributed.jl's workers.
For use on a Slurm cluster, see [`SlurmManager`](@ref) and for use on a PBS
cluster, see [`PBSManager`](@ref).

## Keyword Arguments for `WorkerBackend`
- `failure_rate::Float64`: The threshold for the percentage of workers that can
  fail before an iteration is stopped. The default is
  $DEFAULT_FAILURE_RATE.
- `worker_pool`: A worker pool created from the workers available.
"""
Base.@kwdef struct WorkerBackend{WORKERPOOL <: Distributed.WorkerPool} <:
                   AbstractBackend
    failure_rate::Float64 = DEFAULT_FAILURE_RATE
    worker_pool::WORKERPOOL = default_worker_pool()
end

"""
    HPCBackend <: AbstractBackend

An abstract type for high performance cluster backends.
"""
abstract type HPCBackend <: AbstractBackend end
abstract type SlurmBackend <: HPCBackend end

"""
    JobInfo

A struct containing the backend, job ID, and the job script that was run.
"""
struct JobInfo
    """The backend that the job was submitted with."""
    backend::HPCBackend

    """Job ID of the Slurm (integer) or PBS (string) job."""
    id::Union{Int64, String}

    """Job script that was submitted"""
    job_script::String
end

"""
    Base.show(io::IO, job::JobInfo)

Pretty print the backend and job id of `job`.
"""
function Base.show(io::IO, job::JobInfo)
    labels = ("Backend", "Job ID")
    (; backend, id) = job
    values = (nameof(typeof(backend)), string(id))
    width = maximum(length.(labels))
    for (i, (label, value)) in enumerate(zip(labels, values))
        i > 1 && print(io, "\n")
        print(io, rpad(label, width))
        print(io, ": ")
        print(io, value)
    end
end

"""
    CaltechHPCBackend

Used for Caltech's
[high-performance computing cluster](https://www.hpc.caltech.edu/).
"""
struct CaltechHPCBackend <: SlurmBackend
    hpc_config::SlurmConfig
    job_records::Vector{JobInfo}
end

"""
    CaltechHPCBackend(config::SlurmConfig)

Construct a `CaltechHPCBackend` for submitting jobs to Caltech's
[high-performance computing cluster](https://www.hpc.caltech.edu/).

See [`SlurmConfig`](@ref).
"""
function CaltechHPCBackend(config::SlurmConfig)
    return CaltechHPCBackend(config, [])
end

"""
    CaltechHPCBackend(; directives, modules, env_vars)

Construct a `SlurmConfig` from the keyword arguments and use it to construct a
`CaltechHPCBackend`.
"""
function CaltechHPCBackend(; directives = [], modules = [], env_vars = [])
    return CaltechHPCBackend(SlurmConfig(; directives, modules, env_vars), [])
end

"""
    ClimaGPUBackend

Used for CliMA's private GPU server.
"""
struct ClimaGPUBackend <: SlurmBackend
    hpc_config::SlurmConfig
    job_records::Vector{JobInfo}
end

"""
    ClimaGPUBackend(config::SlurmConfig)

Construct a `ClimaGPUBackend` for submitting jobs to CliMA's private GPU server.

See [`SlurmConfig`](@ref).
"""
function ClimaGPUBackend(config::SlurmConfig)
    return ClimaGPUBackend(config, [])
end

"""
    ClimaGPUBackend(; directives, modules, env_vars)

Construct a `SlurmConfig` from the keyword arguments and use it to construct a
`ClimaGPUBackend`.
"""
function ClimaGPUBackend(; directives = [], modules = [], env_vars = [])
    return ClimaGPUBackend(SlurmConfig(; directives, modules, env_vars), [])
end

"""
    GCPBackend

Used for CliMA's private GCP server.
"""
struct GCPBackend <: SlurmBackend
    hpc_config::SlurmConfig
    job_records::Vector{JobInfo}
end

"""
    GCPBackend(config::SlurmConfig)

Construct a `GCPBackend` for submitting jobs to CliMA's private GCP server.

See [`SlurmConfig`](@ref).
"""
function GCPBackend(config::SlurmConfig)
    return GCPBackend(config, [])
end

"""
    GCPBackend(; directives, modules, env_vars)

Construct a `SlurmConfig` from the keyword arguments and use it to construct a
`GCPBackend`.
"""
function GCPBackend(; directives = [], modules = [], env_vars = [])
    return GCPBackend(SlurmConfig(; directives, modules, env_vars), [])
end

"""
    DerechoBackend

Used for NSF NCAR's
[Derecho supercomputing system](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/).
"""
struct DerechoBackend <: HPCBackend
    hpc_config::PBSConfig
    job_records::Vector{JobInfo}
end

"""
    DerechoBackend(config::PBSConfig)

Construct a `DerechoBackend` for submitting jobs to the Derecho supercomputing
system.

See [`PBSConfig`](@ref).
"""
function DerechoBackend(config::PBSConfig)
    return DerechoBackend(config, [])
end

"""
    DerechoBackend(; directives, modules, env_vars)

Construct a `PBSConfig` from the keyword arguments and use it to construct a
`DerechoBackend`.
"""
function DerechoBackend(; directives = [], modules = [], env_vars = [])
    return DerechoBackend(PBSConfig(; directives, modules, env_vars), [])
end

"""
    job_records(backend::HPCBackend)

Return a vector of `JobInfo`s that were requested with `backend`.
"""
function job_records(backend::HPCBackend)
    return backend.job_records
end

"""
    write_job_script(filepath, job::JobInfo)

Write the job scheduler script for `job` to `filepath`.

This is useful for debugging the script that was submitted to the backend.
"""
function write_job_script(filepath, job::JobInfo)
    write(filepath, job.job_script)
    return nothing
end

# Generic functions for getting the status of a job and cancelling and requeuing
# jobs.
#
# It is the responsibility of the `HPCBackend`s to implement these functions
# themselves. See `slurm.jl` and `pbs.jl` for these implementations.
"""
    job_status(job::JobInfo)

Return the current job status.

See [`JobStatus`](@ref).
"""
job_status(job::JobInfo) = job_status(job.backend, job)

"""
    cancel_job(job::JobInfo)

Cancel the `job`.
"""
cancel_job(job::JobInfo) = cancel_job(job.backend, job)

"""
    requeue_job(job::JobInfo)

Requeue `job` by cancelling the job and resubmitting it again.

This function will requeue the job even if the `job` is completed.
"""
function requeue_job(job::JobInfo)
    # For slurm jobs, one option is to use scontrol requeue, but this would
    # involve defining extra methods for requeue_job to dispatch on the backend
    (; id) = job
    try
        # For both slurm (scancel) and PBS (qdel), cancel_job is a no-op if the
        # job is already completed
        cancel_job(job)
        job_info = submit_job(job.backend, job.job_script)
        println("Requeuing scheduled job $id")
        return job_info
    catch e
        println("Failed to requeue job $id: ", e)
        return nothing
    end
end

"""
    ispending(job::JobInfo)

Return `true` if `job` is pending (i.e. waiting to be scheduled).
"""
ispending(job) = job_status(job) == PENDING

"""
    isrunning(job::JobInfo)

Return `true` if `job` is currently running.
"""
isrunning(job) = job_status(job) == RUNNING

"""
    issuccess(job::JobInfo)

Return `true` if `job` completed successfully.
"""
issuccess(job) = job_status(job) == COMPLETED

"""
    isfailed(job::JobInfo)

Return `true` if `job` failed.
"""
isfailed(job) = job_status(job) == FAILED

"""
    iscompleted(job::JobInfo)

Return `true` if `job` has finished, either successfully or with a failure.
"""
iscompleted(job) = isfailed(job) || issuccess(job)

"""
    cancel_jobs_at_exit(backend::HPCBackend)

Register an exit hook to cancel all jobs submitted by `backend` when the Julia
process exits.
"""
function cancel_jobs_at_exit(backend::HPCBackend)
    cancel_backend_jobs = () -> begin
        for job in job_records(backend)
            # For PBS jobs, checking if the job is completed can take a while,
            # so we call cancel_job on every job even if the job is completed
            cancel_job(job)
        end
    end
    atexit(cancel_backend_jobs)
    return nothing
end

"""
    get_backend()

Get the ideal backend for running work and jobs.

Each backend is found via `gethostname()`. Defaults to `JuliaBackend` if none is
found.
"""
function get_backend()
    # TODO: Add WorkerBackend as default if there are multiple workers
    HOSTNAMES = [
        (r"^clima.gps.caltech.edu$", ClimaGPUBackend),
        (r"^login[1-4].cm.cluster$", CaltechHPCBackend),
        (r"^hpc-(\d\d)-(\d\d).cm.cluster$", CaltechHPCBackend),
        (r"^hpc\d+-slurm-login-\d+$", GCPBackend),
        (r"^hpc\d+-a\d+nodeset-\d+$", GCPBackend),
        (r"^cron$", DerechoBackend),  # Buildkite job launcher on Derecho
        (r"derecho([1-8])$", DerechoBackend),
        (r"dec\d+$", DerechoBackend),  # CPU nodes
        (r"deg(\d\d\d\d)$", DerechoBackend), # GPU nodes
    ]

    for (pattern, backend) in HOSTNAMES
        !isnothing(match(pattern, gethostname())) && return backend
    end

    return JuliaBackend
end

include("backends/slurm.jl")
include("backends/pbs.jl")
include("backends/workers.jl")

end

module BetterBackend # TODO: Find a better name for this

import Distributed

abstract type AbstractBackend end

"""
    JuliaBackend

The simplest backend to use.

This is a singleton type and is meant for use in dispatch.
"""
struct JuliaBackend <: AbstractBackend end

"""
    DEFAULT_FAILURE_RATE

The default failture rate used for the `WorkerBackend`.
"""
const DEFAULT_FAILURE_RATE = 0.5

"""
    WorkerBackend

Used to run calibrations on Distributed.jl's workers.
For use on a Slurm cluster, see [`SlurmManager`](@ref) and for use on a PBS
cluster, see [`PBSManager`](@ref).

## Keyword Arguments for `WorkerBackend`
- `failure_rate::Float64 `: The threshold for the percentage of workers that can
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

All concrete types of `HPCBackend` share the same keyword arguments for the
constructors.

## Keyword Arguments for HPC backends
- `hpc_kwargs::Dict{Symbol, String}`: Dictionary of arguments passed to the job
  scheduler (e.g., Slurm or PBS).
- `verbose::Bool`: Enable verbose logging output. The default is `false`.
"""
abstract type HPCBackend <: AbstractBackend end
abstract type SlurmBackend <: HPCBackend end

"""
    JobInfo

A struct containing the job ID and the script that was ran.
"""
struct JobInfo # TODO: Come up with a better name for this
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

See [`HPCBackend`](@ref) for the keyword arguments to construct a
`CaltechHPCBackend`.
"""
Base.@kwdef struct CaltechHPCBackend <: SlurmBackend
    hpc_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
    job_records::Vector{JobInfo} = []
    verbose::Bool = false # TODO: This keyword argument isn't used anywhere...
end

"""
    ClimaGPUBackend

Used for CliMA's private GPU server.

See [`HPCBackend`](@ref) for the keyword arguments to construct a
`ClimaGPUBackend`.
"""
Base.@kwdef struct ClimaGPUBackend <: SlurmBackend
    hpc_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
    job_records::Vector{JobInfo} = []
    verbose::Bool = false
end

"""
    GCPBackend

Used for CliMA's private GPU server.

See [`HPCBackend`](@ref) for the keyword arguments to construct a
`GCPBackend`.
"""
Base.@kwdef struct GCPBackend <: SlurmBackend
    hpc_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
    job_records::Vector{JobInfo} = []
    verbose::Bool = false
end

"""
    DerechoBackend

Used for NSF NCAR's
[Derecho supercomputing system](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/).

See [`HPCBackend`](@ref) for the keyword arguments to construct a
`DerechoBackend`.
"""
Base.@kwdef struct DerechoBackend <: HPCBackend
    hpc_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
    job_records::Vector{JobInfo} = []
    verbose::Bool = false
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

# Generic functions for getting the status of a job and killing and requeuing a
# jobs.
#
# It is the responsibility of the `HPCBackend`s to implement these functions
# themselves. See `slurm.jl` and `derecho.jl` for these implementations.
"""
    job_status(job::JobInfo)

Return the current job status.
"""
job_status(job::JobInfo) = job_status(job.backend, job)

"""
    kill_job(job::JobInfo)

Kill the `job`.
"""
kill_job(job::JobInfo) = kill_job(job.backend, job)

"""
    requeue_job(::SlurmBackend, job::JobInfo)

Requeue `job` by killing the job and resubmitting it again.

This function will requeue the job even if the `job` is completed.
"""
function requeue_job(job::JobInfo)
    # For slurm jobs, one option is to use scontrol requeue, but this would
    # involve defining extra methods for requeue_job to dispatch on the backend
    (; id) = job
    try
        # For both slurm (scancel) and PBS (qdel), kill_job is a no-op if the
        # job is already completed
        kill_job(job)
        job_info = submit_job(job.backend, job.job_script)
        println("Requeuing slurm job $id")
        return job_info
    catch e
        println("Failed to requeue slurm job $id: ", e)
        return nothing
    end
end

"""
    ispending(job::JobInfo)

Return `true` if `job` is pending (i.e. waiting to be scheduled).
"""
ispending(job) = job_status(job) == :PENDING

"""
    isrunning(job::JobInfo)

Return `true` if `job` is currently running.
"""
isrunning(job) = job_status(job) == :RUNNING

"""
    issuccess(job::JobInfo)

Return `true` if `job` completed successfully.
"""
issuccess(job) = job_status(job) == :COMPLETED

"""
    isfailed(job::JobInfo)

Return `true` if `job` failed.
"""
isfailed(job) = job_status(job) == :FAILED

"""
    iscompleted(job::JobInfo)

Return `true` if `job` has finished, either successfully or with a failure.
"""
iscompleted(job) = isfailed(job) || issuccess(job)

"""
    kill_jobs_at_exit(backend::HPCBackend)

Register an exit hook to cancel all jobs submitted by `backend` when the Julia
process exits.
"""
function kill_jobs_at_exit(backend::HPCBackend)
    cancel_backend_jobs = () -> begin
        for job in job_records(backend)
            # TODO: Investigate whether iscompleted is needed
            # Might be okay to call kill_job on completed jobs
            iscompleted(job) || kill_job(job)
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

"""
    ClimaCalibrate.Backend

Where an ensemble member's forward model runs, and how it is submitted.

A backend is the only thing that changes when a calibration moves from a laptop
to a cluster: [`JuliaBackend`](@ref) runs members one at a time in the current
process, [`WorkerBackend`](@ref) distributes them over Distributed.jl workers,
and each [`HPCBackend`](@ref) submits one scheduler job per member.

This module also holds the scheduler plumbing the HPC backends need: the job
configs ([`SlurmConfig`](@ref), [`PBSConfig`](@ref)), job submission and
cancellation ([`submit_job`](@ref), [`cancel_job`](@ref)), and job status
([`JobStatus`](@ref), [`job_status`](@ref)).
"""
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
    backend_type,
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
    cancel_jobs_at_exit,
    job_records,
    write_job_script,
    make_job_script

include("backends/config.jl")

"""
    AbstractBackend

Where an ensemble member's forward model runs.

The backend is the only thing that changes when a calibration moves from a
laptop to a cluster; `calibrate` dispatches on it to run an iteration's members.

Subtypes:
- [`JuliaBackend`](@ref): one member at a time, in the current process.
- [`WorkerBackend`](@ref): members distributed over Distributed.jl workers.
- [`HPCBackend`](@ref): one scheduler job per member.

# Interface
Subtypes must implement `Calibration.run_iteration(backend, interface, iter,
ensemble_size, output_dir)` and have a `failure_rate` field. See
[`failure_rate`](@ref).
"""
abstract type AbstractBackend end

"""
    JobStatus

An enum representing the current status of a job.

# Values
- `PENDING`: The job is queued and waiting to be scheduled.
- `RUNNING`: The job is currently executing.
- `COMPLETED`: The job finished running.
- `FAILED`: The job terminated with an error as reported by the scheduler.

Use [`ispending`](@ref), [`isrunning`](@ref), [`issuccess`](@ref),
[`isfailed`](@ref), and [`iscompleted`](@ref) to query the status of a
[`JobInfo`](@ref). Each of those queries the scheduler, so ask once and test the
result rather than calling several of them on the same job.

# Examples
```julia
status = ClimaCalibrate.job_status(job)
ClimaCalibrate.iscompleted(status) && ClimaCalibrate.isfailed(status)
```

See also [`job_status`](@ref).
"""
@enum JobStatus begin
    PENDING
    RUNNING
    COMPLETED
    FAILED
end

"""
    DEFAULT_FAILURE_RATE

The fraction of an iteration's ensemble members that may fail before the
calibration is halted.
"""
const DEFAULT_FAILURE_RATE = 0.5

"""
    JuliaBackend(; failure_rate = $DEFAULT_FAILURE_RATE)

Run the ensemble members one at a time in the current process.

# Keyword Arguments
- `failure_rate::Float64`: The fraction of an iteration's ensemble members that
  may fail before the calibration is halted `[-]`. The default is
  $DEFAULT_FAILURE_RATE.

# Examples
```julia
backend = ClimaCalibrate.JuliaBackend()
tolerant = ClimaCalibrate.JuliaBackend(; failure_rate = 0.9)
```
"""
Base.@kwdef struct JuliaBackend <: AbstractBackend
    failure_rate::Float64 = DEFAULT_FAILURE_RATE
end

"""
    failure_rate(backend)

Return the fraction of an iteration's ensemble members that may fail before
`backend` halts the calibration.
"""
failure_rate(backend::AbstractBackend) = backend.failure_rate

"""
    EMPTY_POOL_TIMEOUT

The default number of seconds a `WorkerBackend` iteration will wait on an empty
worker pool before erroring, so an asynchronous calibration cannot hang forever
when no workers ever start.
"""
const EMPTY_POOL_TIMEOUT = 6 * 3600

"""
    WorkerBackend(; failure_rate, worker_pool, empty_pool_timeout)

Run each ensemble member's forward model on a Distributed.jl worker.

Members are handed to workers as they become free, so a calibration can start
before every worker has connected. Add workers with [`add_workers`](@ref) on a
cluster, or with `Distributed.addprocs` locally; see [`SlurmManager`](@ref) and
[`PBSManager`](@ref) for the Slurm and PBS cluster managers.

# Keyword Arguments
- `failure_rate::Float64`: The fraction of an iteration's ensemble members that
  may fail before the calibration is halted `[-]`. The default is
  $DEFAULT_FAILURE_RATE.
- `worker_pool`: A worker pool created from the workers available.
- `empty_pool_timeout::Int`: How long (in seconds) an iteration will wait on an
  empty worker pool before erroring, so an asynchronous calibration cannot hang
  forever when no workers ever start. Defaults to `$EMPTY_POOL_TIMEOUT`.

# Examples
```julia
wait(ClimaCalibrate.add_workers(4; cluster = :local))
ClimaCalibrate.@worker_setup include("my_model.jl")
backend = ClimaCalibrate.WorkerBackend()
```
"""
Base.@kwdef struct WorkerBackend{WORKERPOOL <: Distributed.WorkerPool} <:
                   AbstractBackend
    failure_rate::Float64 = DEFAULT_FAILURE_RATE
    worker_pool::WORKERPOOL = calibration_worker_pool()
    empty_pool_timeout::Int = EMPTY_POOL_TIMEOUT
end

"""
    HPCBackend <: AbstractBackend

Backend that submits one scheduler job per ensemble member.

Each job starts a fresh Julia process, includes the file returned by
`ClimaCalibrate.model_interface_filepath`, and runs one member's forward model.
Prefer this over [`WorkerBackend`](@ref) when forward models are long-running,
need internal parallelism, or will not all fit in one allocation.

Subtypes:
- [`SlurmBackend`](@ref): the Slurm clusters.
- [`DerechoBackend`](@ref): NSF NCAR Derecho, which uses PBS.
"""
abstract type HPCBackend <: AbstractBackend end

"""
    SlurmBackend <: HPCBackend

Abstract supertype for the clusters that use the Slurm scheduler:
[`CaltechHPCBackend`](@ref), [`ClimaGPUBackend`](@ref), and
[`GCPBackend`](@ref).

Job submission, status queries, and cancellation are implemented once for this
type; the concrete backends differ only in which modules they load and how they
launch MPI.
"""
abstract type SlurmBackend <: HPCBackend end

"""
    JOB_TIMEOUT

The default number of seconds an `HPCBackend` iteration waits for a running job
before giving up, so that a scheduler which stops reporting cannot hang a
calibration indefinitely.
"""
const JOB_TIMEOUT = 24 * 3600

"""
    job_timeout(backend::HPCBackend)

Return the number of seconds `backend` waits for a running job before giving up.

The clock starts when a job leaves the queue, so time spent waiting for an
allocation does not count against it.
"""
job_timeout(backend::HPCBackend) = backend.job_timeout

"""
    JobInfo

A submitted scheduler job: which backend submitted it, its scheduler ID, and the
script it runs.

Returned by [`submit_job`](@ref), and the argument to [`job_status`](@ref),
[`cancel_job`](@ref), and [`requeue_job`](@ref).

# Fields
- `backend`: The backend the job was submitted with.
- `id`: The scheduler's job ID, an `Int64` for Slurm and a `String` for PBS.
- `job_script`: The script that was submitted. [`write_job_script`](@ref)
  writes it to a file, which shows what the scheduler was asked to run.
"""
struct JobInfo
    backend::HPCBackend
    id::Union{Int64, String}
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
    CaltechHPCBackend(config::SlurmConfig)
    CaltechHPCBackend(; directives, modules, env_vars, failure_rate, job_timeout)

Submit one scheduler job per ensemble member to Caltech's [high-performance computing cluster](https://www.hpc.caltech.edu/).

The second form builds the `SlurmConfig` from its keyword arguments.

# Fields
- `hpc_config`: Scheduler directives, modules, and environment variables for
  each ensemble member's job. See [`SlurmConfig`](@ref).
- `job_records`: The jobs submitted with this backend, in submission order.
- `failure_rate`: The fraction of an iteration's ensemble members that may fail
  before the calibration is halted `[-]`.
- `job_timeout`: How long (in seconds) an iteration waits for a running job
  before giving up `[s]`. The default is `$JOB_TIMEOUT` (24 hours).

# Examples
```julia
backend = ClimaCalibrate.CaltechHPCBackend(;
    directives = [:time => 60, :ntasks => 1, :cpus_per_task => 8],
    modules = ["climacommon"],
)
```

See also [`failure_rate`](@ref).
"""
struct CaltechHPCBackend <: SlurmBackend
    hpc_config::SlurmConfig
    job_records::Vector{JobInfo}
    failure_rate::Float64
    job_timeout::Int
end

CaltechHPCBackend(
    config::SlurmConfig;
    failure_rate = DEFAULT_FAILURE_RATE,
    job_timeout = JOB_TIMEOUT,
) = CaltechHPCBackend(config, JobInfo[], failure_rate, job_timeout)

CaltechHPCBackend(;
    directives = [],
    modules = [],
    env_vars = [],
    failure_rate = DEFAULT_FAILURE_RATE,
    job_timeout = JOB_TIMEOUT,
) = CaltechHPCBackend(
    SlurmConfig(; directives, modules, env_vars);
    failure_rate,
    job_timeout,
)

"""
    ClimaGPUBackend(config::SlurmConfig)
    ClimaGPUBackend(; directives, modules, env_vars, failure_rate, job_timeout)

Submit one scheduler job per ensemble member to CliMA's private GPU server.

The second form builds the `SlurmConfig` from its keyword arguments.

# Fields
- `hpc_config`: Scheduler directives, modules, and environment variables for
  each ensemble member's job. See [`SlurmConfig`](@ref).
- `job_records`: The jobs submitted with this backend, in submission order.
- `failure_rate`: The fraction of an iteration's ensemble members that may fail
  before the calibration is halted `[-]`.
- `job_timeout`: How long (in seconds) an iteration waits for a running job
  before giving up `[s]`. The default is `$JOB_TIMEOUT` (24 hours).

# Examples
```julia
backend = ClimaCalibrate.ClimaGPUBackend(;
    directives = [:time => 60, :ntasks => 1, :cpus_per_task => 8],
    modules = ["climacommon"],
)
```

See also [`failure_rate`](@ref).
"""
struct ClimaGPUBackend <: SlurmBackend
    hpc_config::SlurmConfig
    job_records::Vector{JobInfo}
    failure_rate::Float64
    job_timeout::Int
end

ClimaGPUBackend(
    config::SlurmConfig;
    failure_rate = DEFAULT_FAILURE_RATE,
    job_timeout = JOB_TIMEOUT,
) = ClimaGPUBackend(config, JobInfo[], failure_rate, job_timeout)

ClimaGPUBackend(;
    directives = [],
    modules = [],
    env_vars = [],
    failure_rate = DEFAULT_FAILURE_RATE,
    job_timeout = JOB_TIMEOUT,
) = ClimaGPUBackend(
    SlurmConfig(; directives, modules, env_vars);
    failure_rate,
    job_timeout,
)

"""
    GCPBackend(config::SlurmConfig)
    GCPBackend(; directives, modules, env_vars, failure_rate, job_timeout)

Submit one scheduler job per ensemble member to CliMA's private GCP server.

The second form builds the `SlurmConfig` from its keyword arguments.

# Fields
- `hpc_config`: Scheduler directives, modules, and environment variables for
  each ensemble member's job. See [`SlurmConfig`](@ref).
- `job_records`: The jobs submitted with this backend, in submission order.
- `failure_rate`: The fraction of an iteration's ensemble members that may fail
  before the calibration is halted `[-]`.
- `job_timeout`: How long (in seconds) an iteration waits for a running job
  before giving up `[s]`. The default is `$JOB_TIMEOUT` (24 hours).

# Examples
```julia
backend = ClimaCalibrate.GCPBackend(;
    directives = [:time => 60, :ntasks => 1, :cpus_per_task => 8],
    modules = ["climacommon"],
)
```

See also [`failure_rate`](@ref).
"""
struct GCPBackend <: SlurmBackend
    hpc_config::SlurmConfig
    job_records::Vector{JobInfo}
    failure_rate::Float64
    job_timeout::Int
end

GCPBackend(
    config::SlurmConfig;
    failure_rate = DEFAULT_FAILURE_RATE,
    job_timeout = JOB_TIMEOUT,
) = GCPBackend(config, JobInfo[], failure_rate, job_timeout)

GCPBackend(;
    directives = [],
    modules = [],
    env_vars = [],
    failure_rate = DEFAULT_FAILURE_RATE,
    job_timeout = JOB_TIMEOUT,
) = GCPBackend(
    SlurmConfig(; directives, modules, env_vars);
    failure_rate,
    job_timeout,
)

"""
    DerechoBackend(config::PBSConfig)
    DerechoBackend(; directives, modules, env_vars, failure_rate, job_timeout)

Submit one scheduler job per ensemble member to NSF NCAR's [Derecho supercomputing system](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/).

The second form builds the `PBSConfig` from its keyword arguments.

# Fields
- `hpc_config`: Scheduler directives, modules, and environment variables for
  each ensemble member's job. See [`PBSConfig`](@ref).
- `job_records`: The jobs submitted with this backend, in submission order.
- `failure_rate`: The fraction of an iteration's ensemble members that may fail
  before the calibration is halted `[-]`.
- `job_timeout`: How long (in seconds) an iteration waits for a running job
  before giving up `[s]`. The default is `$JOB_TIMEOUT` (24 hours).

# Examples
```julia
backend = ClimaCalibrate.DerechoBackend(;
    directives = [:time => 60, :ntasks => 1, :cpus_per_task => 8],
    modules = ["climacommon"],
)
```

See also [`failure_rate`](@ref).
"""
struct DerechoBackend <: HPCBackend
    hpc_config::PBSConfig
    job_records::Vector{JobInfo}
    failure_rate::Float64
    job_timeout::Int
end

DerechoBackend(
    config::PBSConfig;
    failure_rate = DEFAULT_FAILURE_RATE,
    job_timeout = JOB_TIMEOUT,
) = DerechoBackend(config, JobInfo[], failure_rate, job_timeout)

DerechoBackend(;
    directives = [],
    modules = [],
    env_vars = [],
    failure_rate = DEFAULT_FAILURE_RATE,
    job_timeout = JOB_TIMEOUT,
) = DerechoBackend(
    PBSConfig(; directives, modules, env_vars);
    failure_rate,
    job_timeout,
)

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

# Each predicate has a `JobStatus` method as well as a `JobInfo` one. Querying a
# `JobInfo` shells out to the scheduler, so a caller that needs more than one
# predicate should call `job_status` once and ask about the result.
"""
    ispending(job::JobInfo)
    ispending(status::JobStatus)

Return `true` if `job` is pending (i.e. waiting to be scheduled).
"""
ispending(status::JobStatus) = status == PENDING
ispending(job::JobInfo) = ispending(job_status(job))

"""
    isrunning(job::JobInfo)
    isrunning(status::JobStatus)

Return `true` if `job` is currently running.
"""
isrunning(status::JobStatus) = status == RUNNING
isrunning(job::JobInfo) = isrunning(job_status(job))

"""
    issuccess(job::JobInfo)
    issuccess(status::JobStatus)

Return `true` if `job` completed successfully.
"""
issuccess(status::JobStatus) = status == COMPLETED
issuccess(job::JobInfo) = issuccess(job_status(job))

"""
    isfailed(job::JobInfo)
    isfailed(status::JobStatus)

Return `true` if `job` failed.
"""
isfailed(status::JobStatus) = status == FAILED
isfailed(job::JobInfo) = isfailed(job_status(job))

"""
    iscompleted(job::JobInfo)
    iscompleted(status::JobStatus)

Return `true` if `job` has finished, either successfully or with a failure.
"""
iscompleted(status::JobStatus) = isfailed(status) || issuccess(status)
iscompleted(job::JobInfo) = iscompleted(job_status(job))

"""
    cancel_jobs_at_exit(backend::HPCBackend)

Register an exit hook to cancel all jobs submitted by `backend` when the Julia
process exits.
"""
function cancel_jobs_at_exit(backend::HPCBackend)
    # `calibrate` registers this, and a session may call `calibrate` more than
    # once with the same backend. Registering the hook twice would cancel every
    # job twice
    backend in ATEXIT_REGISTERED_BACKENDS && return nothing
    push!(ATEXIT_REGISTERED_BACKENDS, backend)

    cancel_backend_jobs = () -> begin
        for job in job_records(backend)
            # Asking the scheduler which jobs are still alive runs one `qstat`
            # per job on PBS, so the polling loop records the ones it saw finish
            job_finished(job) && continue
            cancel_job(job)
        end
    end
    atexit(cancel_backend_jobs)
    return nothing
end

"""
    mark_job_finished!(job::JobInfo)
    job_finished(job::JobInfo)

Record that `job` has left the scheduler, and read that record back.

[`cancel_jobs_at_exit`](@ref) uses this to cancel the jobs that are still
running when the process exits. Cancelling a finished job runs a `scancel` or
`qdel` that reports an error the user can do nothing about.
"""
mark_job_finished!(job::JobInfo) = (push!(FINISHED_JOBS, job); nothing)

job_finished(job::JobInfo) = job in FINISHED_JOBS

# Held by identity: `JobInfo` is compared field by field, and two members can
# share a job script
const FINISHED_JOBS = Base.IdSet{JobInfo}()

# Backends whose jobs are already scheduled for cancellation at exit. Held by
# identity: two backends with equal configs are still separate job pools
const ATEXIT_REGISTERED_BACKENDS = Base.IdSet{HPCBackend}()

"""
    backend_type()

Return the `AbstractBackend` *type* that suits the current machine, identified
by `gethostname()`. Defaults to [`JuliaBackend`](@ref) when the host matches no
known cluster.

This returns a type, not a backend that `calibrate` accepts: the `HPCBackend`s
need a config, so construct one with
`backend_type()(; directives, modules, env_vars)`.
"""
function backend_type()
    # TODO: Add WorkerBackend as default if there are multiple workers
    HOSTNAMES = [
        (r"^clima\.gps\.caltech\.edu$", ClimaGPUBackend),
        (r"^login[1-4]\.cm\.cluster$", CaltechHPCBackend),
        (r"^hpc-\d\d-\d\d\.cm\.cluster$", CaltechHPCBackend),
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

"""
    get_backend()

Deprecated alias for [`backend_type`](@ref).
"""
function get_backend()
    Base.depwarn(
        "`get_backend` returns a backend *type*, not a backend. It has been \
        renamed to `backend_type` to say so. Construct a backend with \
        `backend_type()(; directives, modules, env_vars)`.",
        :get_backend,
    )
    return backend_type()
end

include("backends/slurm.jl")
include("backends/pbs.jl")
include("backends/workers.jl")

end

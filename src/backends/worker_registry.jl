import TOML

# Worker registry
#
# In-memory table of workers and launch groups, written to TOML after every
# reconcile pass and at exit. Each worker row links a Distributed pid to its
# scheduler job and tracks its state. A launch group is one `add_workers` call,
# or a direct `addprocs(SlurmManager(n))`. This file only stores and updates
# rows. `worker_reconcile.jl` syncs them with the scheduler and Distributed.

"""
    WorkerState

State of a worker, written to the registry file as its lowercase name:

- `STARTING`: submitted, queued, running, or connected and loading code. A
  worker with a `pid` has connected.
- `READY`: in the worker pool, idle
- `BUSY`: checked out of the pool, running a member
- `ENDED`: terminal, never left. `exit_reason` says why, and a missing
  `ready_at` means the worker never made it into the pool.
"""
@enum WorkerState begin
    STARTING
    READY
    BUSY
    ENDED
end
const LIVE_STATES = (STARTING, READY, BUSY)

# Seconds between full `reconcile_workers!` passes. The dispatch loop calls it
# on every pass, which can be several times a second.
const RECONCILE_INTERVAL = 5.0

# Seconds between scheduler queries in `reconcile_workers!`.
const SCHEDULER_POLL_INTERVAL = 30.0

# Seconds a worker may spend starting before it is ended. Queue time counts,
# and queues on shared clusters routinely run to hours, so the default is
# generous; `add_workers(; startup_timeout)` sets it per group.
const DEFAULT_STARTUP_TIMEOUT = 12 * 3600.0

# Seconds after submission during which a job missing from the scheduler's
# listing is not treated as finished. A job can take a while to show up: on
# Derecho `qstat` answers from NCAR's cache, which lags the server by several
# seconds even with the bypass, and a busy Slurm controller can return a
# truncated `squeue` listing. Without the grace a worker could be ended right
# after `qsub`, then start anyway and burn a relaunch.
const UNKNOWN_JOB_GRACE = 60.0

# An `add_workers` call. `launcher` is the tuple `(; manager, kwargs)`,
# specifying a worker launch, with `manager = nothing` for local workers. It is
# `nothing` for a group this session cannot launch. It is not written to the
# file.
Base.@kwdef mutable struct LaunchGroup
    id::Int
    cluster::Symbol
    desired::Int
    max_relaunches::Int
    relaunches_used::Int = 0
    startup_timeout::Float64
    launcher::Any = nothing
end

# Whether this session may launch workers for `g` and cancel its jobs
owned(g::LaunchGroup) = !isnothing(g.launcher)
# The cluster manager of an owned scheduler group, else `nothing`
scheduler_manager(g::LaunchGroup) = owned(g) ? g.launcher.manager : nothing

# One worker. `job_id` is the scheduler's, `pid` is Distributed's
Base.@kwdef mutable struct WorkerRecord
    id::Int
    group_id::Int
    state::WorkerState = STARTING
    job_id::Union{Nothing, String} = nothing
    pid::Union{Nothing, Int} = nothing
    output_file::Union{Nothing, String} = nothing
    exit_reason::Union{Nothing, String} = nothing
    requested_at::Float64
    submitted_at::Union{Nothing, Float64} = nothing
    ready_at::Union{Nothing, Float64} = nothing
    ended_at::Union{Nothing, Float64} = nothing
end

# The registry holds the session's groups and workers, and the file they are
# written to. Ids are positions in the vectors and nothing is ever removed.
# `lock` guards the vectors and `set_state!`; single-field writes to a record
# happen unlocked. No lock here is held while taking `POOL_LOCK`, running a
# scheduler command, or calling `addprocs`. Mutable for the two timestamps
# `reconcile_workers!` updates to rate limit itself.
mutable struct WorkerRegistry
    path::String
    lock::ReentrantLock
    groups::Vector{LaunchGroup}
    workers::Vector{WorkerRecord}
    last_scheduler_poll::Float64
    last_reconcile::Float64
end

WorkerRegistry(path) = WorkerRegistry(
    path,
    ReentrantLock(),
    LaunchGroup[],
    WorkerRecord[],
    -Inf,
    -Inf,
)

# The process's registry, empty until `add_workers` runs. Its path is set in
# the module's `__init__`, since it depends on the working directory and pid
const WORKER_REGISTRY = WorkerRegistry("")

# `.climacalibrate/workers-<driver pid>.toml` under the working directory
default_registry_path() =
    joinpath(pwd(), ".climacalibrate", "workers-$(getpid()).toml")

"""
    set_registry_path!(path)

Write the session's worker registry to `path` instead of
`.climacalibrate/workers-<pid>.toml` in the working directory. Call it before
`add_workers`; the path cannot change once the registry has launch groups.
"""
function set_registry_path!(path, reg = WORKER_REGISTRY)
    isempty(reg.groups) || error(
        "The worker registry already has launch groups; its path cannot change",
    )
    reg.path = abspath(path)
    return reg
end

"""
    worker_registry()

The session's worker registry. Read it with [`worker_rows`](@ref), or read the
TOML file at its `path` after the run.
"""
worker_registry() = WORKER_REGISTRY

# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------

# A record as a TOML table: fields that are `nothing` are left out, enums and
# symbols become their lowercase names, and `launcher` is not written
function record_to_dict(record)
    table = Dict{String, Any}()
    for name in fieldnames(typeof(record))
        name == :launcher && continue
        value = getfield(record, name)
        isnothing(value) && continue
        value isa WorkerState && (value = lowercase(string(value)))
        value isa Symbol && (value = string(value))
        table[string(name)] = value
    end
    return table
end

# Write the groups and workers to `reg.path` through a temporary file, so a
# reader never sees a partial file. A failed write is logged.
function write_registry_file(reg::WorkerRegistry)
    tables = lock(reg.lock) do
        Dict(
            "groups" => map(record_to_dict, reg.groups),
            "workers" => map(record_to_dict, reg.workers),
        )
    end
    try
        mkpath(dirname(reg.path))
        tmp, io = mktemp(dirname(reg.path); cleanup = false)
        try
            TOML.print(io, tables; sorted = true)
        finally
            close(io)
        end
        mv(tmp, reg.path; force = true)
    catch e
        @warn "Could not write the worker registry to $(reg.path)" exception = e maxlog =
            1
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------------

"""
    worker_rows(reg = worker_registry(); states = nothing, group_id = nothing)

The workers, filtered by `states` (a collection of [`WorkerState`](@ref)) and
`group_id`. Returns a new vector of the live records, so the registry may grow
while it is iterated.
"""
function worker_rows(
    reg::WorkerRegistry = WORKER_REGISTRY;
    states = nothing,
    group_id = nothing,
)
    lock(reg.lock) do
        filter(reg.workers) do w
            (isnothing(states) || w.state in states) &&
                (isnothing(group_id) || w.group_id == group_id)
        end
    end
end

# One worker by id.
worker_row(reg::WorkerRegistry, id) = lock(() -> reg.workers[id], reg.lock)

# One worker by Distributed id, or `nothing`.
worker_row_by_pid(reg::WorkerRegistry, pid) = lock(reg.lock) do
    i = findfirst(w -> w.pid == pid, reg.workers)
    isnothing(i) ? nothing : reg.workers[i]
end

# All launch groups, oldest first, in a new vector.
group_rows(reg::WorkerRegistry) = lock(reg.lock) do
    copy(reg.groups)
end

# One launch group by id.
group_row(reg::WorkerRegistry, id) = lock(() -> reg.groups[id], reg.lock)

# ----------------------------------------------------------------------------
# Writes
# ----------------------------------------------------------------------------

# Add a launch group and return its id. `launcher` is `(; manager, kwargs)`, or
# `nothing` for a group this session cannot launch for.
function insert_group!(
    reg::WorkerRegistry;
    cluster,
    desired,
    max_relaunches,
    startup_timeout,
    launcher = nothing,
    now = time(),
)
    lock(reg.lock) do
        id = length(reg.groups) + 1
        push!(
            reg.groups,
            LaunchGroup(;
                id,
                cluster = Symbol(cluster),
                desired,
                max_relaunches,
                startup_timeout,
                launcher,
            ),
        )
        return id
    end
end

# Add a `STARTING` worker to launch group `group_id` and return its id.
function insert_worker!(reg::WorkerRegistry; group_id, now = time())
    lock(reg.lock) do
        id = length(reg.workers) + 1
        push!(reg.workers, WorkerRecord(; id, group_id, requested_at = now))
        return id
    end
end

# Charge `n` launches to the group's relaunch budget.
add_relaunches_used!(reg, group_id, n) =
    lock(() -> (reg.groups[group_id].relaunches_used += n; nothing), reg.lock)

function set_job_id!(reg, id, job_id; now = time())
    w = worker_row(reg, id)
    w.job_id = string(job_id)
    w.submitted_at = now
    return nothing
end
set_pid!(reg, id, pid) = (worker_row(reg, id).pid = pid; nothing)
set_output_file!(reg, id, path) =
    (worker_row(reg, id).output_file = string(path); nothing)

# Move worker `id` to state `to`. Return `false` without writing when the
# worker is not in one of the states `from`. Ending a worker records
# `reason` in `exit_reason`, and `ready_at` is the first time it became `READY`.
function set_state!(
    reg::WorkerRegistry,
    id,
    to::WorkerState;
    from = LIVE_STATES,
    reason = nothing,
    now = time(),
)
    lock(reg.lock) do
        w = reg.workers[id]
        w.state in from || return false
        w.state = to
        to == READY && isnothing(w.ready_at) && (w.ready_at = now)
        if to == ENDED
            w.ended_at = now
            w.exit_reason = reason
        end
        return true
    end
end

# `set_state!` for the worker with Distributed id `pid`. Return `false` when
# no worker has that pid.
function set_state_by_pid!(reg, pid, to::WorkerState; kwargs...)
    w = worker_row_by_pid(reg, pid)
    isnothing(w) && return false
    return set_state!(reg, w.id, to; kwargs...)
end

# State changes from the pool, keyed by Distributed id. No-ops for a worker
# without a row.
mark_worker_busy!(pid) = set_state_by_pid!(WORKER_REGISTRY, pid, BUSY)
mark_worker_ready!(pid) = set_state_by_pid!(WORKER_REGISTRY, pid, READY)
mark_worker_exited!(pid; reason = "deregistered") =
    set_state_by_pid!(WORKER_REGISTRY, pid, ENDED; reason)

# Worker registry
#
# In-memory table of workers and launch groups. Each worker row links a
# Distributed pid to its scheduler job and tracks its state. A launch group is
# one `add_workers` call, or a direct `addprocs(SlurmManager(n))`. This file
# only stores and updates rows. `worker_reconcile.jl` syncs them with the
# scheduler and Distributed.

"""
    WorkerState

State of a worker, stored in its [`WorkerRecord`](@ref):

- `STARTING`: submitted, queued, running, or connected and loading code. A
  worker whose record has a `pid` has connected.
- `READY`: in the worker pool, idle.
- `BUSY`: checked out of the pool, running a member.
- `ENDED`: stopped. The record's `exit_reason` says why, and its `ready_at` is
  `nothing` if the worker never joined the pool.

A worker moves from `STARTING` to `READY`, back and forth between `READY` and
`BUSY`, and from any state to `ENDED`, which it never leaves. Other moves are
refused.
"""
@enum WorkerState begin
    STARTING
    READY
    BUSY
    ENDED
end
const LIVE_STATES = (STARTING, READY, BUSY)

# The states `set_state!` may move a worker to from each state
const ALLOWED_MOVES = Dict{WorkerState, Tuple{Vararg{WorkerState}}}(
    STARTING => (READY, ENDED),
    READY => (BUSY, ENDED),
    BUSY => (READY, ENDED),
    ENDED => (),
)

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

# One `add_workers` call, or a direct `addprocs(SlurmManager(n))`
Base.@kwdef mutable struct LaunchGroup
    # Position in the registry's `groups`
    const id::Int
    # Where the workers run, e.g. `:slurm`, `:pbs`, or `:local`
    const cluster::Symbol
    # Workers the group keeps live. `rmprocs` of a worker lowers it.
    desired::Int
    # Replacements the group may launch, not counting its first workers
    max_relaunches::Int
    # Replacements launched so far
    relaunches_used::Int = 0
    # Seconds after a worker is requested that it may stay `STARTING`
    const startup_timeout::Float64
    # Config `launch_rows!` needs to start more workers: `(; manager, kwargs,
    # resources_for)`, with `manager = nothing` for local workers.
    launcher::Any = nothing
end

# Whether `cancel_worker_jobs` has stopped the group, or it never had a launcher
torn_down(g::LaunchGroup) = isnothing(g.launcher)
# The cluster manager of a scheduler group not torn down, else `nothing`
scheduler_manager(g::LaunchGroup) = torn_down(g) ? nothing : g.launcher.manager

"""
    WorkerRecord

One worker in the worker registry, as returned by [`worker_rows`](@ref). Times
are Unix timestamps in seconds, from `time()`, and are `nothing` until the
event happens.

# Fields
- `id`: position in the registry
- `group_id`: launch group, one per [`add_workers`](@ref) call
- `state`: the worker's [`WorkerState`](@ref)
- `job_id`: scheduler job id, `nothing` for a local worker or before submission
- `pid`: Distributed worker id, `nothing` until the worker connects
- `output_file`: where the job writes the worker's startup output, `nothing`
  for a local worker
- `exit_reason`: why the worker ended, `nothing` while it is live
- `requested_at`: when the worker was requested
- `submitted_at`: when its job was submitted
- `ready_at`: when it first joined the pool
- `ended_at`: when it ended
"""
Base.@kwdef mutable struct WorkerRecord
    const id::Int
    const group_id::Int
    state::WorkerState = STARTING
    job_id::Union{Nothing, String} = nothing
    pid::Union{Nothing, Int} = nothing
    output_file::Union{Nothing, String} = nothing
    exit_reason::Union{Nothing, String} = nothing
    const requested_at::Float64
    submitted_at::Union{Nothing, Float64} = nothing
    ready_at::Union{Nothing, Float64} = nothing
    ended_at::Union{Nothing, Float64} = nothing
end

# One line per worker, e.g. `Worker 3 in group 1: ended (disconnected), job 123,
# pid 4, output /path/to/log`. Fields that are `nothing` are left out.
function Base.show(io::IO, w::WorkerRecord)
    print(io, "Worker ", w.id, " in group ", w.group_id, ": ")
    print(io, lowercase(string(w.state)))
    isnothing(w.exit_reason) || print(io, " (", w.exit_reason, ")")
    isnothing(w.job_id) || print(io, ", job ", w.job_id)
    isnothing(w.pid) || print(io, ", pid ", w.pid)
    isnothing(w.output_file) || print(io, ", output ", w.output_file)
    return nothing
end

# The session's groups and workers. Ids are positions in the vectors and
# nothing is ever removed. `lock` guards the vectors and `set_state!`;
# single-field writes to a record happen unlocked. No lock here is held while
# taking `POOL_LOCK`, running a scheduler command, or calling `addprocs`.
mutable struct WorkerRegistry
    const lock::ReentrantLock
    const groups::Vector{LaunchGroup}
    const workers::Vector{WorkerRecord}
    # `time()` of the last scheduler query and the last `reconcile_workers!`
    # pass, used to space them out
    last_scheduler_poll::Float64
    last_reconcile::Float64
end

WorkerRegistry() =
    WorkerRegistry(ReentrantLock(), LaunchGroup[], WorkerRecord[], -Inf, -Inf)

# The process's registry, empty until `add_workers` runs
const WORKER_REGISTRY = WorkerRegistry()

"""
    worker_registry()

The session's worker registry. List its workers with [`worker_rows`](@ref).
"""
worker_registry() = WORKER_REGISTRY

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
worker_row(reg::WorkerRegistry, id) = lock(reg.lock) do
    reg.workers[id]
end

# One worker by Distributed id, or `nothing`.
worker_row_by_pid(reg::WorkerRegistry, pid) = lock(reg.lock) do
    i = findfirst(w -> w.pid == pid, reg.workers)
    isnothing(i) ? nothing : reg.workers[i]
end

# All launch groups, oldest first. A copy, so callers can iterate it without
# the lock while other tasks add groups.
group_rows(reg::WorkerRegistry) = lock(reg.lock) do
    copy(reg.groups)
end

# One launch group by id.
group_row(reg::WorkerRegistry, id) = lock(reg.lock) do
    reg.groups[id]
end

# ----------------------------------------------------------------------------
# Writes
# ----------------------------------------------------------------------------

# Add a launch group and return its id. See `LaunchGroup` for `launcher`.
function insert_group!(
    reg::WorkerRegistry;
    cluster,
    desired,
    max_relaunches,
    startup_timeout,
    launcher = nothing,
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
add_relaunches_used!(reg, group_id, n) = lock(reg.lock) do
    reg.groups[group_id].relaunches_used += n
    nothing
end

function set_job_id!(reg, id, job_id; now = time())
    w = worker_row(reg, id)
    w.job_id = string(job_id)
    w.submitted_at = now
    return nothing
end
set_pid!(reg, id, pid) = (worker_row(reg, id).pid = pid; nothing)
set_output_file!(reg, id, path) =
    (worker_row(reg, id).output_file = string(path); nothing)

# Move worker `id` to state `to`. Return `false` without writing when the move
# is not in `ALLOWED_MOVES`, or when `from` is given and the worker is not in
# one of its states. Ending a worker records `reason` in `exit_reason`, and
# `ready_at` is the first time it became `READY`.
function set_state!(
    reg::WorkerRegistry,
    id,
    to::WorkerState;
    from = nothing,
    reason = nothing,
    now = time(),
)
    lock(reg.lock) do
        w = reg.workers[id]
        to in ALLOWED_MOVES[w.state] || return false
        isnothing(from) || w.state in from || return false
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

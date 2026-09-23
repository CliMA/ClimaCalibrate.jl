# Reconcile the worker registry with the scheduler
#
# The registry is updated in two ways:
# - Push: events that reach the driver write rows directly. `Distributed.manage`
#   sets a worker's pid on `:register` and records its exit on `:deregister`.
#   The pool sets `READY` once code is loaded and `BUSY` while a member runs.
# - Poll: `reconcile_workers!` checks what the driver must ask about. It ends
#   workers whose job finished or left the queue, whose process vanished
#   without a deregister, or that stayed starting too long. It launches
#   replacements for groups below their desired count. It runs on every pass
#   of the `WorkerBackend` dispatch loop and every iteration of the launch poll.
#
# Locking rule: nothing here holds `reg.lock` while running a scheduler
# command or `addprocs`. Row reads and writes each take the lock briefly.

# Serializes `reconcile_workers!` calls. A call that finds it held returns.
const RECONCILE_LOCK = ReentrantLock()

# Whether the worker on `row` is connected. The pid is read before the process
# list, since Distributed lists a process before its `:register` hook sets the
# pid. `alive_pids` replaces `Distributed.procs()` when given.
function _worker_connected(row, alive_pids)
    pid = row.pid
    isnothing(pid) && return false
    return pid in (isnothing(alive_pids) ? Distributed.procs() : alive_pids)
end

# Ask each scheduler about the live jobs of every group submitting to it, in
# one query per manager type, and end the workers whose job is finished.
# `query_scheduler_states` returns `Dict(id => JobStatus | nothing)`, or
# `nothing` when the query failed. A job the scheduler no longer lists, or
# reports as completed or failed, is finished, unless the worker's process is
# still connected: then Distributed's deregister hook reports the real exit.
function _update_scheduler_states!(reg::WorkerRegistry, alive_pids, now)
    now - reg.last_scheduler_poll >= SCHEDULER_POLL_INTERVAL || return nothing
    reg.last_scheduler_poll = now
    by_type = Dict{Type, Tuple{Any, Vector{WorkerRecord}}}()
    for g in group_rows(reg)
        manager = scheduler_manager(g)
        isnothing(manager) && continue
        live = filter(
            r -> !isnothing(r.job_id),
            worker_rows(reg; states = LIVE_STATES, group_id = g.id),
        )
        isempty(live) && continue
        _, rows = get!(by_type, typeof(manager), (manager, WorkerRecord[]))
        append!(rows, live)
    end
    for (manager, live) in values(by_type)
        states = query_scheduler_states(manager, unique(r.job_id for r in live))
        isnothing(states) && continue
        for row in live
            status = get(states, row.job_id, nothing)
            listed = haskey(states, row.job_id)
            if listed
                status in (Backend.COMPLETED, Backend.FAILED) || continue
                reason = "scheduler reported $status"
            else
                # Missing from the listing: finished, unless it was submitted
                # moments ago and has not shown up yet
                submitted = something(row.submitted_at, row.requested_at)
                isnothing(row.pid) &&
                    now - submitted <= UNKNOWN_JOB_GRACE &&
                    continue
                reason = "job left the queue"
            end
            _worker_connected(row, alive_pids) ||
                set_state!(reg, row.id, ENDED; reason, now)
        end
    end
    return nothing
end

# End live rows whose connected process is gone. Scheduler workers usually get
# here first through the deregister hook; local workers only get here, since
# `LocalManager` does not call our `Distributed.manage`.
function _end_disconnected_workers!(reg::WorkerRegistry, alive_pids, now)
    for row in worker_rows(reg; states = LIVE_STATES)
        isnothing(row.pid) && continue
        _worker_connected(row, alive_pids) && continue
        set_state!(reg, row.id, ENDED; reason = "disconnected", now)
    end
    return nothing
end

# End rows still starting after their group's startup timeout and cancel the
# job when no live sibling shares it.
function _apply_startup_timeouts!(reg::WorkerRegistry, now)
    to_cancel = Dict{Any, Vector{String}}()
    for row in worker_rows(reg; states = (STARTING,))
        g = group_row(reg, row.group_id)
        elapsed = now - row.requested_at
        elapsed > g.startup_timeout || continue
        set_state!(
            reg,
            row.id,
            ENDED;
            from = (STARTING,),
            reason = "not ready after $(round(Int, elapsed))s",
            now,
        ) || continue
        isnothing(row.job_id) && continue
        manager = scheduler_manager(g)
        isnothing(manager) && continue
        siblings = worker_rows(reg; states = LIVE_STATES)
        any(w -> w.job_id == row.job_id, siblings) ||
            push!(get!(to_cancel, manager, String[]), row.job_id)
    end
    for (manager, ids) in to_cancel
        cancel_scheduler_jobs(manager, ids)
    end
    return nothing
end

# Bring each group with a launcher up to its desired count within budget.
# `submit(group_id, n)` must insert the new rows before it returns, so the next
# pass does not see the same deficit.
function _fill_deficits!(reg::WorkerRegistry, submit)
    for g in group_rows(reg)
        owned(g) || continue
        live = length(worker_rows(reg; states = LIVE_STATES, group_id = g.id))
        deficit = g.desired - live
        deficit > 0 || continue
        budget = g.max_relaunches - g.relaunches_used
        if budget <= 0
            @warn "Launch group $(g.id) is $deficit worker(s) short but has used all $(g.max_relaunches) relaunches" maxlog =
                1 _id = Symbol("relaunch_budget_$(g.id)")
            continue
        end
        n = min(deficit, budget)
        add_relaunches_used!(reg, g.id, n)
        @info "Launch group $(g.id) has $live/$(g.desired) live worker(s), launching $n more"
        submit(g.id, n)
    end
    return nothing
end

"""
    reconcile_workers!(reg = worker_registry(); kwargs...)

Bring the registry in line with the cluster and the groups' desired counts:

1. Query each scheduler (at most every `SCHEDULER_POLL_INTERVAL` seconds) and
   end workers whose job left the queue or finished. A job missing from the
   listing within `UNKNOWN_JOB_GRACE` seconds of submission is left alone, and
   so is a worker whose process is still connected.
2. End workers whose connected process is gone.
3. End workers still starting after their group's startup timeout.
4. Launch replacements for each group short of its desired count, within its
   relaunch budget. Groups from another session's registry file have no
   launcher and are left alone.

Runs at most every `RECONCILE_INTERVAL` seconds, and
writes the registry file afterwards. No-op while another call is in progress
or when the registry has no launch groups. Errors are logged, not thrown.

Keyword arguments exist for testing: `now`, `alive_pids`, and
`submit(group_id, n)`.
"""
function reconcile_workers!(
    reg::WorkerRegistry = WORKER_REGISTRY;
    now = time(),
    alive_pids = nothing,
    submit = launch_group!,
)
    isempty(group_rows(reg)) && return nothing
    trylock(RECONCILE_LOCK) || return nothing
    try
        now - reg.last_reconcile >= RECONCILE_INTERVAL || return nothing
        reg.last_reconcile = now
        _update_scheduler_states!(reg, alive_pids, now)
        _end_disconnected_workers!(reg, alive_pids, now)
        _apply_startup_timeouts!(reg, now)
        _fill_deficits!(reg, submit)
        write_registry_file(reg)
    catch e
        @error "Reconciling the worker registry failed" exception =
            (e, catch_backtrace()) maxlog = 5
    finally
        unlock(RECONCILE_LOCK)
    end
    return nothing
end

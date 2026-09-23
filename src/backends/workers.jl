"""
    ClimaCalibrate.Backend.Workers

Distributed.jl workers for the
[`WorkerBackend`](@ref ClimaCalibrate.Backend.WorkerBackend): launching them
on a scheduler or locally ([`add_workers`](@ref), [`SlurmManager`](@ref),
[`PBSManager`](@ref)), loading code on them ([`@worker_setup`](@ref)), the pool
they join ([`calibration_worker_pool`](@ref)), and the worker registry that
tracks and relaunches them ([`worker_registry`](@ref)).
"""
module Workers

using Distributed
using Logging

import ...ClimaCalibrate: project_dir
import ..Backend
import ..Backend:
    AbstractBackend,
    DerechoBackend,
    GCPBackend,
    SlurmBackend,
    backend_type,
    format_slurm_time,
    format_pbs_time,
    slurm_flag_args,
    _parse_slurm_state,
    _parse_pbs_state,
    _run_capturing_output,
    scheduler_env,
    SLURM_INHERITED_VARS,
    PBS_INHERITED_VARS

include("worker_registry.jl")
include("worker_reconcile.jl")

function __init__()
    WORKER_REGISTRY.path = default_registry_path()
    get!(ENV, "JULIA_WORKER_TIMEOUT", worker_timeout())
    return nothing
end

export add_workers,
    calibration_worker_pool,
    set_worker_loggers,
    set_worker_logger,
    cancel_worker_jobs,
    SlurmManager,
    PBSManager,
    get_manager,
    map_remotecall_fetch,
    foreach_remotecall_wait,
    @worker_setup

# Set the time limit for the Julia worker to be contacted by the main process, default = "60.0s"
# https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_WORKER_TIMEOUT
worker_timeout() = "300.0"

# ----------------------------------------------------------------------------
# Global worker pool for asynchronous calibration
#
# Workers are submitted as individual allocations and add themselves to this
# pool (via the `Distributed.manage` `:register` hook) only after loading the
# model code. A calibration starts with an empty pool and assigns model runs to
# workers as they join.
# ----------------------------------------------------------------------------

"""
    GLOBAL_WORKER_POOL

The process-wide [`Distributed.WorkerPool`](@ref) that workers add themselves to
when they start. Used as the default pool for
[`WorkerBackend`](@ref ClimaCalibrate.Backend.WorkerBackend).
"""
const GLOBAL_WORKER_POOL = WorkerPool()

# Guards mutation of GLOBAL_WORKER_POOL, INITIALIZING_WORKERS, and WORKER_SETUP.
const POOL_LOCK = ReentrantLock()

# Workers that have connected but are still loading code: present in `workers()`
# but not yet schedulable. Keeps `calibration_worker_pool` from pooling them
# early.
const INITIALIZING_WORKERS = Set{Int}()

"""
    n_initializing_workers()

Number of workers that have connected but are still loading code (and so are not
yet in the pool). Used to distinguish "workers are on the way" from "no workers
are coming".
"""
n_initializing_workers() = lock(POOL_LOCK) do
    length(INITIALIZING_WORKERS)
end

"""
    calibration_worker_pool()

Return the process-wide `GLOBAL_WORKER_POOL`, which is what a
[`WorkerBackend`](@ref ClimaCalibrate.Backend.WorkerBackend) draws ensemble
members from.

Cluster workers add themselves via the `:register` hook. Workers added by other
means (e.g. plain `addprocs`/`LocalManager` or pre-existing workers) are picked
up here: each is claimed with `_claim_worker` and initialized in the
background, so it joins the pool once it has the code to run a forward model.

A worker enters `workers()` when `Distributed` registers it, which is before the
`:register` hook runs, so pooling ids straight from `workers()` can hand a
member to a worker that has yet to load `ClimaCalibrate`. Claiming is what keeps
the two paths from either racing or initializing the same worker twice.

The name is package-specific because `Distributed` exports a
`default_worker_pool` of its own, which makes the unqualified name ambiguous
under `using Distributed, ClimaCalibrate`.
"""
function calibration_worker_pool()
    # id 1 is the main process, not a worker
    unclaimed = filter(id -> id != 1 && _claim_worker(id), workers())
    for id in unclaimed
        @async _initialize_claimed_worker(id)
    end
    return GLOBAL_WORKER_POOL
end

"""
    default_worker_pool()

Deprecated name for [`calibration_worker_pool`](@ref).
"""
function default_worker_pool()
    Base.depwarn(
        "`default_worker_pool` is now `calibration_worker_pool`, since \
        `Distributed` exports a `default_worker_pool` of its own. It is not \
        exported, so call it as `ClimaCalibrate.calibration_worker_pool()`.",
        :default_worker_pool,
    )
    return calibration_worker_pool()
end

# ----------------------------------------------------------------------------
# Worker code-loading registry
#
# `@everywhere` only runs on workers that exist when it is called, so workers
# that join later (asynchronously) would not have the model code. Instead, we
# record setup expressions on the main process and replay them on each worker as it
# joins (see `initialize_worker`). `@worker_setup` is a drop-in replacement for
# `@everywhere` that both applies now and persists for future workers.
# ----------------------------------------------------------------------------

# Ordered list of SOURCE_PATH-wrapped toplevel expressions to run on a worker.
const WORKER_SETUP = Expr[]

# Reimplementation of Distributed's internal `extract_imports`: pull out
# `using`/`import` statements so they can be run locally first (to precompile
# once on the main process rather than racing across joining workers).
_extract_imports!(imports, x) = imports
function _extract_imports!(imports, ex::Expr)
    if Meta.isexpr(ex, (:import, :using))
        push!(imports, ex)
    elseif Meta.isexpr(ex, :let)
        _extract_imports!(imports, ex.args[2])
    elseif Meta.isexpr(ex, (:toplevel, :block))
        foreach(a -> _extract_imports!(imports, a), ex.args)
    end
    return imports
end
_extract_imports(x) = _extract_imports!(Any[], x)

"""
    register_worker_setup!(ex::Expr, source_path)

Record `ex` to run on all current and future workers, then apply it to all
current processes. `source_path` is propagated so relative `include` resolves on
the workers. Used by [`@worker_setup`](@ref).
"""
function register_worker_setup!(ex::Expr, source_path)
    wrapped = Expr(
        :toplevel,
        :(task_local_storage()[:SOURCE_PATH] = $source_path),
        ex,
    )
    lock(POOL_LOCK) do
        push!(WORKER_SETUP, wrapped)
    end
    # Apply to processes that already exist (main process + connected workers).
    Distributed.remotecall_eval(Main, procs(), wrapped)
    return nothing
end

"""
    @worker_setup expr

Like `Distributed.@everywhere`, but the expression is also recorded and replayed
on any worker that joins later.

!!! tip
    Use `@worker_setup`, not `@everywhere`, to set up workers for a
    `WorkerBackend`. Workers join asynchronously, and `@everywhere` skips any
    that connect after it runs, leaving them without the model code.

`using`/`import` statements run on the main process first (to precompile once), and
the current source path is propagated so relative `include` works on workers.
As with `@everywhere`, local variables must be interpolated with `\$`.
"""
macro worker_setup(ex)
    imps = _extract_imports(ex)
    return quote
        $(isempty(imps) ? nothing : Expr(:toplevel, map(esc, imps)...))
        # `esc(Expr(:quote, ex))` (rather than `QuoteNode(ex)`) so that `$`
        # interpolations are resolved in the caller's scope
        $(register_worker_setup!)(
            $(esc(Expr(:quote, ex))),
            get(task_local_storage(), :SOURCE_PATH, nothing),
        )
    end
end

"""
    initialize_worker(id)

Prepare worker `id` and add it to `GLOBAL_WORKER_POOL`. Loads
`ClimaCalibrate`, sets the working directory and logger, and replays all
recorded [`@worker_setup`](@ref) expressions. The worker is pushed to the pool
*only after* code loading completes, so it is never scheduled before it is
ready. Failures (e.g. a worker dying mid-init) are logged and the worker is not
pooled.

Does nothing if the worker is already pooled or is being initialized by
[`calibration_worker_pool`](@ref).
"""
function initialize_worker(id)
    _claim_worker(id) || return nothing
    return _initialize_claimed_worker(id)
end

"""
    _claim_worker(id)

Claim worker `id` for initialization, returning whether this caller is the one
that has to initialize it.

Adding `id` to `INITIALIZING_WORKERS` under `POOL_LOCK` is what makes the claim
exclusive: a worker is claimed by whichever of the `:register` hook and
[`calibration_worker_pool`](@ref) reaches it first, and the other leaves it
alone. Replaying the `@worker_setup` expressions twice would fail on the first
`struct` among them.
"""
function _claim_worker(id)
    lock(POOL_LOCK) do
        id in INITIALIZING_WORKERS && return false
        id in GLOBAL_WORKER_POOL.workers && return false
        push!(INITIALIZING_WORKERS, id)
        return true
    end
end

# Initialize a worker already claimed with `_claim_worker`, releasing the claim
# when it is either pooled or given up on.
function _initialize_claimed_worker(id)
    try
        Distributed.remotecall_wait(cd, id, pwd())
        Distributed.remotecall_eval(Main, id, :(using ClimaCalibrate, Logging))
        Distributed.remotecall_wait(set_worker_logger, id)
        # Snapshot the registry so we don't hold POOL_LOCK across remote calls
        # (lets workers initialize concurrently)
        setup = lock(POOL_LOCK) do
            copy(WORKER_SETUP)
        end
        for wrapped in setup
            Distributed.remotecall_wait(Core.eval, id, Main, wrapped)
        end
        # Only now is the worker schedulable. Its row is `READY` before it is
        # pooled, so it cannot overwrite the `BUSY` of a member already
        # running on it. A row ended while the worker loaded code (startup
        # timeout) already has a replacement on the way, so the worker is
        # stopped instead. The membership check keeps a worker that
        # reconnects under the same id out of the pool's channel twice
        row = worker_row_by_pid(WORKER_REGISTRY, id)
        if !isnothing(row) && !set_state!(WORKER_REGISTRY, row.id, READY)
            @warn "Worker $id was ended while loading its code ($(row.exit_reason)), stopping it"
            stop_process(id)
            return nothing
        end
        lock(POOL_LOCK) do
            id in GLOBAL_WORKER_POOL.workers || push!(GLOBAL_WORKER_POOL, id)
        end
        @info "Worker $id initialized and added to pool"
    catch e
        @warn "Worker $id failed to initialize; not added to pool" exception = e
        stop_worker_after_setup_failure!(id, e)
    finally
        lock(POOL_LOCK) do
            delete!(INITIALIZING_WORKERS, id)
        end
    end
    return nothing
end

# First line of the error's message, unwrapped from a `RemoteException`.
function error_summary(e)
    e isa RemoteException && (e = e.captured.ex)
    msg = first(split(sprint(showerror, e), '\n'))
    return length(msg) > 200 ? first(msg, 200) * "..." : msg
end

# End the row of worker `id`, which failed to load its code, and stop it. A
# worker without a row (from the caller's own `addprocs`) is only left out of
# the pool. A worker that disconnected during setup is not counted against its
# group. Once a group has two setup failures and no worker that ever loaded
# its code, the setup code is assumed broken and the group launches no more
# workers.
function stop_worker_after_setup_failure!(id, e)
    reg = WORKER_REGISTRY
    row = worker_row_by_pid(reg, id)
    isnothing(row) && return nothing
    if !(id in Distributed.procs())
        mark_worker_exited!(id; reason = "disconnected during setup")
        return nothing
    end
    mark_worker_exited!(id; reason = "setup failed: $(error_summary(e))")
    stop_process(id)
    rows = worker_rows(reg; group_id = row.group_id)
    any(r -> !isnothing(r.ready_at), rows) && return nothing
    failures = count(
        r -> startswith(something(r.exit_reason, ""), "setup failed"),
        rows,
    )
    failures >= 2 || return nothing
    lock(reg.lock) do
        g = group_row(reg, row.group_id)
        if g.relaunches_used < g.max_relaunches
            @warn "Launch group $(g.id) will not relaunch workers, since $failures workers failed to load their code and none succeeded"
            g.relaunches_used = g.max_relaunches
        end
    end
    return nothing
end

# Ask worker `id` to exit. Errors are ignored, since the worker may already be
# gone.
function stop_process(id)
    try
        Distributed.remote_do(exit, id)
    catch
    end
    return nothing
end

"""
    remove_worker_from_pool(id)

Remove worker `id` from `GLOBAL_WORKER_POOL`. Called when a worker deregisters
(e.g. walltime expiry or crash).

A `Distributed.WorkerPool` holds its workers both in a `Set` and in an internal
`Channel` that `take!` draws from. This only removes `id` from the `Set`, so a
copy of `id` may still sit in the channel. That copy is harmless because
`WorkerPool`'s `take!` discards any id that is no longer a live process before
returning one.
"""
function remove_worker_from_pool(id)
    lock(POOL_LOCK) do
        delete!(GLOBAL_WORKER_POOL.workers, id)
        delete!(INITIALIZING_WORKERS, id)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Cluster managers
#
# One `ClusterManager` per scheduler. Their `launch` and `manage` methods are
# further down, next to the submission code they share.
# ----------------------------------------------------------------------------

"""
    SlurmManager(ntasks = 1)

The ClusterManager for Slurm clusters, taking in the number of workers to
request. Each worker is submitted as its own batch job with `sbatch`.

To submit the jobs, run `addprocs(SlurmManager(ntasks))`.

Keyword arguments can be passed to `sbatch`: `addprocs(SlurmManager(ntasks),
gpus_per_task=1)`.

By default the workers will inherit the running Julia environment.

To run a calibration, call `calibrate(WorkerBackend(), ...)`.

To run functions on a worker, call `remotecall(func, worker_id, args...)`.
"""
struct SlurmManager <: ClusterManager
    ntasks::Integer

    SlurmManager(ntasks = 1) = new(ntasks)
end

"""
    PBSManager(ntasks)

The ClusterManager for PBS Pro clusters, taking in the number of workers to
request. Each allocation is submitted as its own job with `qsub`.

To submit the jobs, run `addprocs(PBSManager(ntasks))`.

Keyword arguments can be passed to `qsub`: `addprocs(PBSManager(ntasks),
nodes=2)`

By default, the workers will inherit the running Julia environment.

To run a calibration, call `calibrate(WorkerBackend(), ...)`

To run functions on a worker, call `remotecall(func, worker_id, args...)`
"""
struct PBSManager <: ClusterManager
    ntasks::Integer
end

# Managers that submit workers as scheduler jobs. Their workers carry their
# registry row id in `config.userdata`.
const SchedulerManager = Union{SlurmManager, PBSManager}

# Name of the scheduler a manager submits to, as stored in the worker registry.
cluster_name(::SlurmManager) = :slurm
cluster_name(::PBSManager) = :pbs

# ----------------------------------------------------------------------------
# Scheduler queries and job teardown
#
# Workers are submitted as individual batch jobs (see `launch`) and recorded in
# the worker registry with their job ids. `reconcile_workers!` asks the
# scheduler about live jobs through `query_scheduler_states`. If the main
# process exits while jobs are pending or running, the `atexit` hook cancels
# them by id through `cancel_worker_jobs`.
#
# Locking rule: nothing here runs a scheduler command or `addprocs`/`rmprocs`
# while holding the registry lock. `Distributed` calls the `:deregister` hook
# from its message task, and that hook writes to the registry.
# ----------------------------------------------------------------------------

# Ensures the teardown `atexit` hook is registered at most once per session.
const ATEXIT_HOOK_REGISTERED = Ref(false)

# Run `cmd`, discarding its output. Used for the scheduler cancellation commands
# below (`scancel`/`qdel`); this only silences those commands' own output and has
# no effect on forward-model logs, which workers write via `set_worker_logger`.
_run_quiet(cmd) = run(pipeline(cmd; stdout = devnull, stderr = devnull))

# Copy of `ENV` without the variables in `SLURM_INHERITED_VARS`.
function scheduler_env(::Union{SlurmBackend, SlurmManager})
    clean_env = Dict{String, String}(ENV)
    for var in SLURM_INHERITED_VARS
        delete!(clean_env, var)
    end
    return clean_env
end

# The user site-packages directory is disabled and NCAR's qstat-cache bypassed.
# The cache answers from a snapshot refreshed every few seconds, which reports
# a job submitted since the last refresh as "Unknown Job Id" (exit 153) and a
# finished job in whatever state the snapshot caught it in.
function scheduler_env(::Union{DerechoBackend, PBSManager})
    clean_env = Dict{String, String}(ENV)
    for k in PBS_INHERITED_VARS
        delete!(clean_env, k)
    end
    clean_env["PYTHONNOUSERSITE"] = "1"
    clean_env["QSCACHE_BYPASS"] = "true"
    return clean_env
end

"""
    query_scheduler_states(manager, job_ids)

States of `job_ids` submitted through `manager` as
`Dict(id => JobStatus | nothing)`, or `nothing` when the scheduler could not be
queried. Jobs the scheduler no longer lists are absent from the result.

One query covers every job, so the cost does not grow with the number of
workers.
"""
function query_scheduler_states end

# One `squeue` call for every job of this session, matched by the shared job
# name. `squeue -j` aborts when any listed id has already left the controller.
function query_scheduler_states(::SlurmManager, job_ids)
    cmd = `squeue --name $(worker_jobname()) -h -o "%i %T"`
    out, err, code = try
        _run_capturing_output(cmd)
    catch e
        @warn "squeue could not be run" exception = e maxlog = 5
        return nothing
    end
    if !iszero(code)
        @warn "squeue failed with exit code $code: $err" maxlog = 5
        return nothing
    end
    return _parse_squeue_output(out)
end

# Parse `squeue -h -o "%i %T"` output, one `<id> <STATE>` per line.
function _parse_squeue_output(out)
    states = Dict{String, Union{Backend.JobStatus, Nothing}}()
    for line in eachline(IOBuffer(out))
        tokens = split(line)
        length(tokens) >= 2 || continue
        states[String(first(tokens))] = _parse_slurm_state(tokens[2])
    end
    return states
end

# One `qstat` call for `job_ids`. `-x` includes finished jobs. `qstat` exits
# nonzero when any id is unknown but still reports the others.
function query_scheduler_states(pm::PBSManager, job_ids)
    cmd = setenv(`qstat -x -f -F dsv $job_ids`, scheduler_env(pm))
    out, err, code = try
        _run_capturing_output(cmd)
    catch e
        @warn "qstat could not be run" exception = e maxlog = 5
        return nothing
    end
    if isempty(out) && !iszero(code)
        @warn "qstat failed with exit code $code: $err" maxlog = 5
        return nothing
    end
    return _parse_qstat_output(out)
end

# Parse `qstat -f -F dsv` output, one `Job Id: <id>|key = value|...` per line.
function _parse_qstat_output(out)
    states = Dict{String, Union{Backend.JobStatus, Nothing}}()
    for line in eachline(IOBuffer(out))
        m = match(r"^Job Id:\s*([^|\s]+)", line)
        isnothing(m) && continue
        states[String(m[1])] = _parse_pbs_state(line)
    end
    return states
end

"""
    cancel_scheduler_jobs(manager, job_ids)

Cancel `job_ids` submitted through `manager` with `scancel` or `qdel`.
Failures are logged.
"""
function cancel_scheduler_jobs end

cancel_scheduler_jobs(::SlurmManager, job_ids) =
    cancel_quietly(`scancel $job_ids`, job_ids)
cancel_scheduler_jobs(::PBSManager, job_ids) =
    cancel_quietly(`qdel $job_ids`, job_ids)

function cancel_quietly(cmd, job_ids)
    isempty(job_ids) && return nothing
    try
        _run_quiet(cmd)
    catch e
        @warn "Failed to cancel jobs $job_ids" exception = e
    end
    return nothing
end

"""
    cancel_worker_jobs()

Cancel every scheduler job this session submitted for the worker registry and
end the live workers with `exit_reason = "cancel_worker_jobs"`. This tears
down connected workers (by cancelling their job), jobs still in the queue, and
jobs of workers already ended, and forgets how to launch more so nothing is
relaunched. Local workers are
ended in the registry but their processes are left to `rmprocs`, so they stay
in the pool. Safe to call with no live jobs or no registry.

Registered as an `atexit` hook whenever workers are launched, so that jobs are
not orphaned when the main process exits. It may also be called directly to
tear down workers early.

This does not call `rmprocs`, which waits on Distributed's global worker lock.
A direct `addprocs(SlurmManager(n))` or `addprocs(PBSManager(n))` holds that
lock until each of its jobs has connected or ended. Cancelling the jobs
releases the workers directly.
"""
function cancel_worker_jobs(; reg::WorkerRegistry = WORKER_REGISTRY)
    for g in group_rows(reg)
        # Not this session's group, so not its jobs to cancel
        owned(g) || continue
        rows = worker_rows(reg; group_id = g.id)
        manager = scheduler_manager(g)
        if !isnothing(manager)
            ids = unique(r.job_id for r in rows if !isnothing(r.job_id))
            cancel_scheduler_jobs(manager, ids)
        end
        for r in rows
            set_state!(reg, r.id, ENDED; reason = "cancel_worker_jobs")
        end
        # Without a launcher nothing is relaunched on the next reconcile pass
        g.launcher = nothing
    end
    write_registry_file(reg)
    return nothing
end

# Register `cancel_worker_jobs` as an `atexit` hook exactly once, so jobs submitted
# in this session are cleaned up if the main process exits before the caller
# tears them down. Guarded by `POOL_LOCK` against a double registration.
# `init_multi` registers Distributed's own hook first. Hooks run last in, first
# out, so jobs are cancelled before Distributed's `rmprocs`, which can block on
# its worker lock.
function ensure_worker_atexit_hook!()
    lock(POOL_LOCK) do
        if !ATEXIT_HOOK_REGISTERED[]
            Distributed.init_multi()
            atexit(cancel_worker_jobs)
            ATEXIT_HOOK_REGISTERED[] = true
        end
    end
    return nothing
end

worker_cookie() = begin
    Distributed.init_multi()
    cluster_cookie()
end

# Attach the Distributed id to the worker's registry row (stored in
# `config.userdata` by `worker_config`) and load its code on a task of its own,
# so `addprocs` returns, and releases Distributed's worker lock, once the
# worker has connected. On deregistration, drop it from the pool and record
# the exit.
function Distributed.manage(
    ::SchedulerManager,
    id::Integer,
    config::WorkerConfig,
    op::Symbol,
)
    if op == :register
        set_pid!(WORKER_REGISTRY, config.userdata, id)
        _claim_worker(id) && errormonitor(@async _initialize_claimed_worker(id))
    elseif op == :deregister
        remove_worker_from_pool(id)
        mark_worker_exited!(id)
    end
    return nothing
end

# Local workers go through `LocalManager`, whose `manage` is Distributed's own,
# so their exits are noticed by `reconcile_workers!` instead, and `rmprocs` on
# one is relaunched like any other exit.

# `rmprocs` of a scheduler worker: end its row and lower its group's desired
# count so it is not relaunched, then stop it as Distributed would. Distributed
# also calls this when a connection fails, before the `:register` hook sets the
# row's pid. That row is left to `connect_worker`, so it is relaunched.
function Distributed.kill(
    manager::SchedulerManager,
    id::Int,
    config::WorkerConfig,
)
    reg = WORKER_REGISTRY
    row = config.userdata isa Int ? worker_row(reg, config.userdata) : nothing
    if !isnothing(row) &&
       row.pid == id &&
       set_state!(reg, row.id, ENDED; reason = "removed with rmprocs")
        g = group_row(reg, row.group_id)
        g.desired = max(g.desired - 1, 0)
    end
    return invoke(
        Distributed.kill,
        Tuple{ClusterManager, Int, WorkerConfig},
        manager,
        id,
        config,
    )
end


# Where a worker's startup output goes, default to a temp dir under `exehome`.
# The job writes this file on the node it runs on, so a caller passing `o` or
# `output` must give a path on the shared filesystem, not `/tmp`.
function default_worker_output_base(params, exehome, jobname)
    haskey(params, :o) && return params[:o]
    haskey(params, :output) && return params[:output]
    # Keep the worker logs after the main process exit to make it easier to
    # debug a calibration
    dir = mktempdir(exehome; prefix = ".julia_worker_", cleanup = false)
    return joinpath(dir, jobname)
end

"""
    submit_jobs!(manager, params, row_ids)

Submit one batch job per allocation for the registry rows `row_ids` and return
each row's output file, where its worker prints its address. `params` are
`addprocs`-style keywords: scheduler flags plus Distributed's own.

Workers are grouped into allocations of `workers_per_node` (one per allocation
by default). Each allocation gets a script under the output base that runs its
workers (see `single_worker_script` and `multi_worker_script`). An allocation
whose rows have all ended is not submitted, and one whose rows end while it is
submitted is cancelled.
"""
function submit_jobs!(manager, params, row_ids)
    reg = WORKER_REGISTRY
    params = add_default_worker_params(params)
    exehome = params[:dir]
    exename = params[:exename]
    exeflags = worker_exeflags(manager, params[:exeflags])
    env = Dict{String, String}(params[:env])
    propagate_env_vars!(env)
    env = merge(scheduler_env(manager), env)
    jobname = worker_jobname()
    output_base = default_worker_output_base(params, exehome, jobname)
    base = submit_command(manager, params, jobname)
    workers_per_node = get(params, :workers_per_node, 1)
    all_ended(ids) = all(id -> worker_row(reg, id).state == ENDED, ids)
    ntasks = length(row_ids)
    output_files = String[]
    counts = workers_per_allocation(ntasks, workers_per_node)
    njobs = length(counts)
    next = 1
    for (j, nworkers) in enumerate(counts)
        ids = row_ids[next:(next + nworkers - 1)]
        next += nworkers
        if workers_per_node == 1
            outputs = ["$output_base-$j.out"]
        else
            outputs = [abspath("$output_base-$j-$g.out") for g in 1:nworkers]
        end
        append!(output_files, outputs)
        all_ended(ids) && continue
        if workers_per_node == 1
            script_path = "$output_base-$j.sh"
            write(script_path, single_worker_script(exename, exeflags))
            cmd = `$base -o $(only(outputs)) $script_path`
        else
            script_path = "$output_base-multiworker-$j.sh"
            write(script_path, multi_worker_script(exename, exeflags, outputs))
            cmd = `$base -o $output_base-job$j.log $script_path`
        end
        chmod(script_path, 0o700)
        foreach(
            ((id, output_file),) -> set_output_file!(reg, id, output_file),
            zip(ids, outputs),
        )
        @info "Submitting worker job [$j/$njobs] with $nworkers worker(s): $cmd"
        out, err, code = _run_capturing_output(setenv(cmd, env))
        job_id = iszero(code) ? parse_job_id(manager, out) : nothing
        if isnothing(job_id)
            reason = "submission exited with $code: $(isempty(err) ? out : err)"
            @warn "Worker job [$j/$njobs] was not submitted. $reason"
            foreach(id -> set_state!(reg, id, ENDED; reason), ids)
        else
            foreach(id -> set_job_id!(reg, id, job_id), ids)
            all_ended(ids) && cancel_scheduler_jobs(manager, [job_id])
        end
    end
    return output_files
end

# Rows for a direct `addprocs(manager)`, in a group of its own. That group
# carries a launcher so the session owns it and cancels its jobs at exit, but
# `desired = 0` means nothing is ever relaunched for it.
function rows_for_launch!(manager)
    group_id = insert_group!(
        WORKER_REGISTRY;
        cluster = cluster_name(manager),
        desired = 0,
        max_relaunches = 0,
        startup_timeout = DEFAULT_STARTUP_TIMEOUT,
        launcher = (; manager, kwargs = (;)),
    )
    return [insert_worker!(WORKER_REGISTRY; group_id) for _ in 1:manager.ntasks]
end

# Job id from `sbatch --parsable` output, which is `<id>` or `<id>;<cluster>`.
function _parse_sbatch_output(out)
    m = match(r"^\d+", out)
    return isnothing(m) ? nothing : String(m.match)
end

# Job id from `qsub` output, which is the id alone.
_parse_qsub_output(out) = isempty(out) ? nothing : String(out)

parse_job_id(::SlurmManager, out) = _parse_sbatch_output(out)
parse_job_id(::PBSManager, out) = _parse_qsub_output(out)

# Submission command for one allocation, without its output file and script.
function submit_command(::SlurmManager, params, jobname)
    worker_args = parse_slurm_worker_params(params)
    return `sbatch --parsable -J $jobname -n 1 -D $(params[:dir]) $worker_args`
end
# -V inherit env, -N job name, -j oe merge stdout/stderr
submit_command(
    ::PBSManager,
    params,
    jobname,
) = `qsub -V -N $jobname -j oe $(parse_pbs_worker_params(params))`

# PBS workers default to the driver's project.
worker_exeflags(::SlurmManager, exeflags) = exeflags
worker_exeflags(::PBSManager, exeflags) =
    exeflags == `` ? `--project=$(project_dir())` : exeflags

# With the `connect` keyword, hand Distributed the `WorkerConfig` of a worker
# that has already started, so `addprocs` holds Distributed's worker lock only
# while it connects. `add_workers` submits its jobs and waits for them to start
# outside `addprocs`, then connects each worker this way (see `launch_rows!`).
# Otherwise this is a direct `addprocs(manager)`: submit the jobs and wait for
# every worker inside `launch`, holding the lock throughout.
function Distributed.launch(
    manager::SchedulerManager,
    params::Dict,
    instances_arr::Array,
    launch_condition::Condition,
)
    if haskey(params, :connect)
        push!(instances_arr, params[:connect])
        notify(launch_condition)
        return nothing
    end
    # Ensure submitted jobs are cancelled if the main process exits.
    ensure_worker_atexit_hook!()
    row_ids = rows_for_launch!(manager)
    output_files = submit_jobs!(manager, params, row_ids)
    nstarted = 0
    t_waited = await_worker_addresses(row_ids, output_files) do _, config
        nstarted += 1
        push!(instances_arr, config)
        notify(launch_condition)
    end
    report_startup(nstarted, length(row_ids), t_waited)
    return nothing
end


workers_per_allocation(ntasks, max_per_node) =
    [min(max_per_node, ntasks - i) for i in 0:max_per_node:(ntasks - 1)]

"""
    parse_slurm_worker_params(params::Dict)

Parse params into string arguments for the worker launch command.

Uses all keys that are not in `Distributed.default_addprocs_params()`.
"""
function parse_slurm_worker_params(params::Dict)
    stdkeys = keys(Distributed.default_addprocs_params())
    excepted_keys = ADDPROCS_PASSTHROUGH_KEYS
    worker_params =
        filter(x -> !(x[1] in stdkeys || x[1] in excepted_keys), params)
    worker_args = []

    for (k, v) in worker_params
        if string(k) == "o" || string(k) == "output"
            continue
        end
        append!(worker_args, slurm_flag_args(k, v))
    end
    return worker_args
end

worker_jobname() = "julia-$(getpid())"

# `addprocs` keywords that are not scheduler flags
const ADDPROCS_PASSTHROUGH_KEYS = (:job_file_loc, :workers_per_node, :connect)

function add_default_worker_params(params)
    default_params = Distributed.default_addprocs_params()
    params = merge(default_params, Dict{Symbol, Any}(params))
    return params
end

function propagate_env_vars!(env)
    # Taken from Distributed.jl
    if get(env, "JULIA_LOAD_PATH", nothing) === nothing
        env["JULIA_LOAD_PATH"] = join(LOAD_PATH, ":")
    end
    if get(env, "JULIA_DEPOT_PATH", nothing) === nothing
        env["JULIA_DEPOT_PATH"] = join(DEPOT_PATH, ":")
    end
    project = Base.ACTIVE_PROJECT[]
    if project !== nothing && get(env, "JULIA_PROJECT", nothing) === nothing
        env["JULIA_PROJECT"] = project
    end
end

# This regex will match the worker's socket, ex: julia_worker:9015#169.254.3.1
const JULIA_WORKER_REGEX = r"([\w]+):([\d]+)#(\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3})"

# The address line a worker printed to `file`, or `nothing`
function read_worker_address(file)
    (isfile(file) && filesize(file) > 0) || return nothing
    for line in eachline(file)
        m = match(JULIA_WORKER_REGEX, line)
        isnothing(m) || return m
    end
    return nothing
end

# Poll one output file per worker and call `found(i, config)` with the
# `WorkerConfig` of each worker that prints its address. `row_ids[i]` is the
# registry row that owns `output_files[i]`. A worker whose row has ended (job
# failed, timed out, cancelled) is logged and skipped. The loop ends when every
# row has printed its address or ended; `reconcile_workers!` is what ends rows
# that never start, through the scheduler query and the group's startup
# timeout. Returns the seconds waited.
function await_worker_addresses(found, row_ids, output_files)
    reg = WORKER_REGISTRY
    @assert length(output_files) == length(row_ids)
    ntasks = length(output_files)
    t_start = time()
    delay = 0.0
    pending = collect(1:ntasks)
    while true
        t_waited = round(Int, time() - t_start)
        reconcile_workers!(reg)
        filter!(pending) do i
            row = worker_row(reg, row_ids[i])
            if row.state == ENDED
                @warn "Worker $i/$ntasks ended before connecting ($(row.exit_reason)); skipping. Check $(output_files[i])."
                return false
            end
            address = read_worker_address(output_files[i])
            isnothing(address) && return true
            config = worker_config(address, row_ids[i])
            @info "Worker ready after $(t_waited)s on host $(config.host), port $(config.port) (worker $i/$ntasks)"
            found(i, config)
            return false
        end
        isempty(pending) && return t_waited
        # Back off to limit resource usage while waiting for jobs to start
        sleep(delay)
        delay = min(max(1.0, 1.5 * delay), 30.0)
    end
end

# Warn when only some of `ntasks` workers started, and throw when none did.
function report_startup(nstarted, ntasks, t_waited)
    if 0 < nstarted < ntasks
        @warn "After $t_waited s, $nstarted/$ntasks workers started. Continuing with available workers."
    end
    nstarted == 0 && error(
        "No workers started after $t_waited s. Check the job scheduler output.",
    )
    return nothing
end


# `userdata` carries the registry row id to the `:register` hook.
function worker_config(worker_launch_details, row_id)
    config = WorkerConfig()
    config.port = parse(Int, worker_launch_details[2])
    config.host = strip(worker_launch_details[3])
    config.userdata = row_id
    return config
end


# Connect the worker described by `config` through `launch`'s `connect`
# keyword. Returns its Distributed id, or
# `nothing` after ending its row when it cannot be reached.
function connect_worker(manager, config)
    try
        return only(addprocs(manager; connect = config))
    catch e
        @warn "Could not connect to the worker at $(config.host):$(config.port)" exception =
            e
        reason = "connection failed: $(nameof(typeof(e)))"
        set_state!(WORKER_REGISTRY, config.userdata, ENDED; reason)
        return nothing
    end
end

# Wait for the workers of `row_ids`, whose jobs are already submitted, to print
# their addresses, connecting each as it does. Returns the Distributed ids of
# the workers that connected; throws if none did.
function connect_workers!(manager, row_ids, output_files)
    tasks = Task[]
    t_waited = await_worker_addresses(row_ids, output_files) do _, config
        push!(tasks, @async connect_worker(manager, config))
    end
    ids = filter(!isnothing, fetch.(tasks))
    report_startup(length(ids), length(row_ids), t_waited)
    return ids
end



# Quote `s` for bash. Wrap the string in single quotes and replace each
# existing single quote ' with its escaped version '\''
shell_quote(s) = "'" * replace(string(s), "'" => "'\\''") * "'"

# Shell-quoted command line that starts a Julia worker.
_worker_command_string(exename, exeflags) = join(
    shell_quote.([
        string(exename),
        exeflags.exec...,
        "--worker=$(worker_cookie())",
    ]),
    ' ',
)

"""
    single_worker_script(exename, exeflags)

Bash script that runs one Julia worker in the foreground, so the job lives as
long as the worker.
"""
single_worker_script(exename, exeflags) = """
#!/bin/bash
exec $(_worker_command_string(exename, exeflags))
"""

"""
    multi_worker_script(exename, exeflags, worker_outputs)

Bash script that starts one Julia worker process per entry of
`worker_outputs` on a single node. Worker `g` sees only GPU `g - 1` through
`CUDA_VISIBLE_DEVICES` (harmless on CPU nodes) and redirects its output to
`worker_outputs[g]`, where the master polls for the `julia_worker` startup
line. The script waits on all workers so the allocation stays alive while
any of them runs.
"""
function multi_worker_script(exename, exeflags, worker_outputs)
    worker_cmd = _worker_command_string(exename, exeflags)
    lines = ["#!/bin/bash"]
    for (g, output) in enumerate(worker_outputs)
        push!(
            lines,
            "CUDA_VISIBLE_DEVICES=$(g - 1) $worker_cmd > $(shell_quote(output)) 2>&1 &",
        )
    end
    push!(lines, "wait")
    return join(lines, "\n") * "\n"
end

"""
    parse_pbs_worker_params(params::Dict)

Parse params into string arguments for the worker launch command.

Uses all keys that are not in `Distributed.default_addprocs_params()`. Keys that
start with `l_` will be treated as `-l` arguments to `qsub`. For example,
l_walltime = "00:10:00" is transformed into `-l walltime=00:10:00`.
"""
function parse_pbs_worker_params(params::Dict)
    stdkeys = keys(Distributed.default_addprocs_params())
    excepted_keys = ADDPROCS_PASSTHROUGH_KEYS
    worker_params =
        filter(x -> !(x[1] in stdkeys || x[1] in excepted_keys), params)
    worker_args = []

    for (k, v) in worker_params
        # Exceptions for `-l` and `-o` options
        if startswith(string(k), "l_")
            str_k = string(k)[3:end]
            # Special handling for ` -l select=...` parameter
            # Each job can only have one task
            if str_k == "select"
                v = "$v"
            end
            append!(worker_args, ["-l", "$str_k=$v"])
            continue
        elseif string(k) == "o"
            continue
        end

        k2 = replace(string(k), "_" => "-")
        if length(v) > 0
            append!(worker_args, ["-$k2", "$v"])
        else
            push!(worker_args, "-$k2")
        end
    end
    return worker_args
end

"""
    map_remotecall_fetch(f::Function, args...; workers = workers())

Call function `f` from each worker and wait for the results to return.
"""
function map_remotecall_fetch(f::Function, args...; workers = workers())
    return map(workers) do worker
        remotecall_fetch(worker) do
            if isempty(args)
                f()
            else
                f(args...)
            end
        end
    end
end

"""
    foreach_remotecall_wait(f::Function, args...; workers = workers())

Call function `f` from each worker.
"""
function foreach_remotecall_wait(f::Function, args...; workers = workers())
    foreach(workers) do worker
        remotecall_wait(worker) do
            if isempty(args)
                f()
            else
                f(args...)
            end
        end
    end
end

"""
    set_worker_logger()

Set the worker's global logger to write to `worker_\$worker_id.log` in its
working directory.

Call this from the worker process. [`add_workers`](@ref) does so for each
worker it starts.

# Returns
The `SimpleLogger` that was installed.
"""
function set_worker_logger()
    @eval Main using Logging
    io = open("worker_$(myid()).log", "w")
    logger = SimpleLogger(io)
    Base.global_logger(logger)
    @info "Logging from worker $(myid())"
    flush(io)
    return logger
end

"""
    set_worker_loggers(workers = workers())

Set the global logger to a simple file logger for the given workers.
"""
function set_worker_loggers(workers = workers())
    # `workers` has to be passed as the keyword argument: as a positional
    # argument it would become the argument forwarded to the closure, and the
    # target list would silently fall back to all workers
    return map_remotecall_fetch(; workers) do
        @eval Main begin
            using ClimaCalibrate
            set_worker_logger()
        end
    end
end


function is_pbs_available()
    return all([
        !isnothing(Sys.which("qstat")),
        !isnothing(Sys.which("pbsnodes")),
        !isnothing(Sys.which("qsub")),
    ])
end


function is_slurm_available()
    return all([
        !isnothing(Sys.which("sinfo")),
        !isnothing(Sys.which("srun")),
        !isnothing(Sys.which("sbatch")),
    ])
end

function is_cluster_environment()
    return is_pbs_available() || is_slurm_available()
end

const DEFAULT_WALLTIME = 60

default_cpu_kwargs(::SlurmManager) = (;
    cpus_per_task = 1,
    time = format_slurm_time(DEFAULT_WALLTIME),
    backend_worker_kwargs(backend_type())...,
)
default_cpu_kwargs(::PBSManager) = (;
    l_select = "ncpus=1",
    l_walltime = format_pbs_time(DEFAULT_WALLTIME),
    backend_worker_kwargs(backend_type())...,
)

default_gpu_kwargs(::SlurmManager) = (;
    gpus_per_task = 1,
    cpus_per_task = 4,
    time = format_slurm_time(DEFAULT_WALLTIME),
    backend_worker_kwargs(backend_type())...,
)
default_gpu_kwargs(::PBSManager) = (;
    l_select = "ngpus=1:ncpus=4",
    l_walltime = format_pbs_time(DEFAULT_WALLTIME),
    backend_worker_kwargs(backend_type())...,
)

# Resources for one allocation of `n` workers. The workers run as `n` background
# processes in a single task, so that task needs all `n` workers' resources.
# Each worker gets 4 CPUs per GPU, matching the single-worker defaults above
# (`ngpus=1:ncpus=4`), so `n` GPU workers need `4n` CPUs.
allocation_resource_kwargs(::PBSManager, device, n) = Dict{Symbol, Any}(
    :l_select => device == :gpu ? "ngpus=$n:ncpus=$(4n)" : "ncpus=$n",
)
allocation_resource_kwargs(::SlurmManager, device, n) =
    device == :gpu ?
    Dict{Symbol, Any}(:gpus_per_task => n, :cpus_per_task => 4n) :
    Dict{Symbol, Any}(:cpus_per_task => n)

backend_worker_kwargs(::Type{DerechoBackend}) =
    (; q = "main@desched1", A = "UCIT0011")
backend_worker_kwargs(::Type{GCPBackend}) = (; partition = "a3")
backend_worker_kwargs(::Type{<:AbstractBackend}) = (;)

"""
    get_manager(cluster = :auto, nworkers = 1)

Return the `ClusterManager` for `cluster`, which is one of `:slurm`, `:pbs`, or
`:auto` to pick whichever scheduler's commands are on `PATH`.

`:local` workers do not need a manager, so [`add_workers`](@ref) handles that
case before calling this.
"""
function get_manager(cluster = :auto, nworkers = 1)
    if cluster == :slurm || (cluster == :auto && is_slurm_available())
        SlurmManager(nworkers)
    elseif cluster == :pbs || (cluster == :auto && is_pbs_available())
        PBSManager(nworkers)
    elseif cluster == :auto
        error("Neither Slurm nor PBS was detected on this machine. Pass \
              `cluster = :local` to `add_workers` to start workers locally.")
    else
        error(
            "Unknown cluster type: $cluster. Valid options are :auto, :pbs, :slurm, or :local",
        )
    end
end

"""
    add_workers(
        nworkers;
        device = :gpu,
        cluster = :auto,
        time = DEFAULT_WALLTIME,
        max_relaunches = 2 * nworkers,
        startup_timeout = DEFAULT_STARTUP_TIMEOUT,
        kwargs...
    )

Add `nworkers` worker processes to the current Julia session, automatically
detecting and configuring for the available computing environment.

This does not wait for the workers to connect. Each worker is submitted as an
individual allocation and adds itself to `GLOBAL_WORKER_POOL` once it
has started and loaded its code, so a calibration can begin with an empty pool
and pick up workers as they join.

The call is recorded as a launch group in the worker registry (see
[`worker_registry`](@ref), written to a TOML file under `.climacalibrate/`
in the working directory) with `nworkers` as its desired count. While a
calibration runs, a worker that exits, fails, or is not ready within
`startup_timeout` seconds is replaced, up to `max_relaunches` replacements for
the group. To add workers later, call `add_workers` again.

The returned `Task` runs the (blocking) submission; `wait` on it to block until
all submissions have been processed. Submitted jobs are cancelled automatically
when the process exits (via an `atexit` hook); call [`cancel_worker_jobs`](@ref)
to tear them down earlier.

Use [`@worker_setup`](@ref) (instead of `@everywhere`) to load model code so
that workers joining later get the same setup.

# Arguments
- `nworkers::Int`: The number of worker processes to add.
- `device::Symbol = :gpu`: The target compute device type, either `:gpu` (1 GPU,
  4 CPU cores) or `:cpu` (1 CPU core).
- `cluster::Symbol = :auto`: The cluster management system to use. Options:
  * `:auto`: Auto-detect available cluster environment (SLURM, PBS, or local)
  * `:slurm`: Force use of SLURM scheduler
  * `:pbs`: Force use of PBS scheduler
  * `:local`: Force use of local processing (standard `addprocs`)
- `time::Int = DEFAULT_WALLTIME`: Walltime in minutes, will be formatted
  appropriately for the cluster system
- `workers_per_node::Int = 1`: Number of workers to run per node.
- `max_relaunches::Int = 2 * nworkers`: Replacement workers this call may launch
  in total.
- `startup_timeout::Real = DEFAULT_STARTUP_TIMEOUT`: Seconds from submission
  until a worker that has not joined the pool is ended and replaced. Queue
  time counts, so the default is 12 hours; lower it on a cluster with short
  queues to replace a stuck worker sooner.
- `o`: Base path for the workers' output files, which the job writes on the
  node it runs on, so it must be on the shared filesystem. Defaults to a
  `.julia_worker_*` directory in the working directory.
- `kwargs`: Other kwargs can be passed directly through to `addprocs`.

# Returns
A `Task` running the submission. `wait` on it to block until all workers have
been submitted; the workers themselves join the pool as they connect.

# Examples
```julia
# On a cluster: four GPU workers, each its own allocation
wait(ClimaCalibrate.add_workers(4; time = 120))

# Locally, for debugging
wait(ClimaCalibrate.add_workers(2; cluster = :local))

# On a cluster that charges for whole nodes, four workers per allocation
wait(ClimaCalibrate.add_workers(8; workers_per_node = 4))
```

See also [`@worker_setup`](@ref), [`cancel_worker_jobs`](@ref),
[`calibration_worker_pool`](@ref).
"""
function add_workers(
    nworkers;
    device = :gpu,
    cluster = :auto,
    time = DEFAULT_WALLTIME,
    max_relaunches = 2 * nworkers,
    startup_timeout = DEFAULT_STARTUP_TIMEOUT,
    kwargs...,
)
    return errormonitor(
        Threads.@spawn _add_workers(
            nworkers;
            device,
            cluster,
            time,
            max_relaunches,
            startup_timeout,
            kwargs...,
        )
    )
end

function _add_workers(
    nworkers;
    device,
    cluster,
    time,
    max_relaunches,
    startup_timeout,
    kwargs...,
)
    reg = WORKER_REGISTRY
    workers_per_node = get(kwargs, :workers_per_node, 1)
    if cluster == :local || (cluster == :auto && !is_cluster_environment())
        @info "Using local processing mode, adding $nworkers worker$(nworkers == 1 ? "" : "s")"
        workers_per_node > 1 && throw(
            ArgumentError(
                "workers_per_node = $workers_per_node groups several workers " *
                "into one scheduler allocation, which only applies on a " *
                "cluster. Local processing starts each worker as its own " *
                "process, so use cluster = :slurm or :pbs, or drop " *
                "workers_per_node.",
            ),
        )
        manager = nothing
        cluster = :local
        merged_kwargs = (; kwargs...)
        resources_for = nothing
    else
        manager = get_manager(cluster, nworkers)
        cluster = cluster_name(manager)
        @info "Using $(nameof(typeof(manager))) to add $nworkers workers"

        default_kwargs =
            device == :gpu ? default_gpu_kwargs(manager) :
            device == :cpu ? default_cpu_kwargs(manager) :
            throw(
                ArgumentError(
                    "device must be :gpu or :cpu, got $(repr(device))",
                ),
            )

        normalized_kwargs = process_time_parameter(manager, time, kwargs)
        merged_kwargs = merge(default_kwargs, normalized_kwargs)

        # Resources for an allocation of `n` workers. Explicit resource
        # requests from the caller win.
        resources_for =
            n -> filter(
                p -> !haskey(kwargs, first(p)),
                allocation_resource_kwargs(manager, device, n),
            )
        if workers_per_node > 1
            merged_kwargs =
                merge(merged_kwargs, resources_for(workers_per_node))
        else
            resources_for = nothing
        end
    end

    ensure_worker_atexit_hook!()
    # The group and its rows appear together, so a concurrent
    # `reconcile_workers!` never sees the group short of workers
    group_id, row_ids = lock(reg.lock) do
        gid = insert_group!(
            reg;
            cluster,
            desired = nworkers,
            max_relaunches,
            startup_timeout,
            launcher = (; manager, kwargs = merged_kwargs, resources_for),
        )
        gid, [insert_worker!(reg; group_id = gid) for _ in 1:nworkers]
    end
    return launch_rows!(group_id, row_ids)
end

# Insert `n` rows for launch group `group_id` and launch replacement workers for
# them on a background task. The rows exist before this returns, so the next
# `reconcile_workers!` sees the group as full.
function launch_group!(group_id, n)
    reg = WORKER_REGISTRY
    row_ids = [insert_worker!(reg; group_id) for _ in 1:n]
    errormonitor(
        Threads.@spawn launch_rows!(group_id, row_ids; relaunch = true)
    )
    return nothing
end

# Launch workers for the rows `row_ids` of launch group `group_id` with the
# group's recorded manager and keywords, and return their Distributed ids once
# every row has connected or ended. Scheduler jobs are submitted and awaited
# outside `addprocs` (see `connect_worker`). A relaunch writes its output
# files under a base of its own, so it never reads an earlier worker's file.
# Fewer workers than `workers_per_node` get an allocation sized for that many.
function launch_rows!(group_id, row_ids; relaunch = false)
    reg = WORKER_REGISTRY
    launcher = group_row(reg, group_id).launcher
    # `cancel_worker_jobs` has ended the rows
    isnothing(launcher) && return nothing
    (; manager, kwargs) = launcher
    n = length(row_ids)
    resources_for = get(launcher, :resources_for, nothing)
    if !isnothing(resources_for) && n < get(kwargs, :workers_per_node, 1)
        kwargs = merge(kwargs, resources_for(n))
    end
    if relaunch
        for key in (:o, :output)
            haskey(kwargs, key) || continue
            kwargs = merge(
                kwargs,
                Dict(key => "$(kwargs[key])-relaunch$(first(row_ids))"),
            )
        end
    end
    ids = try
        if isnothing(manager)
            addprocs(n; kwargs...)
        else
            params = Dict{Symbol, Any}(pairs(kwargs))
            output_files = submit_jobs!(manager, params, row_ids)
            connect_workers!(manager, row_ids, output_files)
        end
    catch e
        reason = "launch failed: $(nameof(typeof(e)))"
        for id in row_ids
            isnothing(worker_row(reg, id).pid) &&
                set_state!(reg, id, ENDED; reason)
        end
        rethrow()
    end
    # Scheduler workers get their pid and load their code through `manage`
    isnothing(manager) || return ids
    for (row, id) in zip(row_ids, ids)
        set_pid!(reg, row, id)
    end
    @sync for id in ids
        @async initialize_worker(id)
    end
    return ids
end

"""
    process_time_parameter(manager, time, kwargs)

Process the time parameter and convert it to the appropriate format for the
specific cluster manager. This function translates a simple `time = minutes`
parameter into the appropriate format for each system.

Priority rules:
1. If system-specific time parameter exists in kwargs (e.g., `l_walltime` for
   PBS), use that directly
2. If `time` parameter is provided, convert it to the appropriate
   system-specific format
3. If neither is specified, defaults will be used from default_*_kwargs
   functions
"""
function process_time_parameter(::SlurmManager, time::Int, kwargs)
    # If time already exists in kwargs in Slurm format, use that (highest priority)
    if haskey(kwargs, :time)
        return kwargs
    end
    # Otherwise, use the time parameter and convert it to Slurm format
    return merge(kwargs, Dict(:time => format_slurm_time(time)))
end

function process_time_parameter(::PBSManager, time::Int, kwargs)
    # If l_walltime already exists in kwargs in PBS format, use that (highest priority)
    if haskey(kwargs, :l_walltime)
        return kwargs
    end
    # Otherwise, use the time parameter and convert it to PBS format
    return merge(kwargs, Dict(:l_walltime => format_pbs_time(time)))
end

# Fallback for other manager types
function process_time_parameter(_, time::Int, kwargs)
    # For other manager types, pass through the kwargs unchanged
    return kwargs
end

end # module Workers

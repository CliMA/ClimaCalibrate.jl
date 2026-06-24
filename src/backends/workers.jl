using Distributed
using Logging

import ..ClimaCalibrate: project_dir

export add_workers,
    default_worker_pool,
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
ENV["JULIA_WORKER_TIMEOUT"] = worker_timeout()

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
when they start. Used as the default pool for [`WorkerBackend`](@ref).
"""
const GLOBAL_WORKER_POOL = WorkerPool()

# Guards mutation of GLOBAL_WORKER_POOL, INITIALIZING, and WORKER_SETUP.
const POOL_LOCK = ReentrantLock()

# Workers that have connected but are still loading code: present in `workers()`
# but not yet schedulable. Keeps `default_worker_pool` from pooling them early.
const INITIALIZING = Set{Int}()

# Generous bound (seconds) on how long a calibration will wait on an empty
# worker pool before erroring, so an asynchronous calibration cannot hang
# forever when no workers ever start.
const EMPTY_POOL_TIMEOUT = 7200

"""
    n_initializing_workers()

Number of workers that have connected but are still loading code (and so are not
yet in the pool). Used to distinguish "workers are on the way" from "no workers
are coming".
"""
n_initializing_workers() = lock(POOL_LOCK) do
    length(INITIALIZING)
end

"""
    default_worker_pool()

Return the process-wide `GLOBAL_WORKER_POOL`.

Cluster workers add themselves via the `:register` hook. Workers added by other
means (e.g. plain `addprocs`/`LocalManager` or pre-existing workers) are
reconciled into the pool here. Workers still loading code (in `INITIALIZING`)
are skipped so they are not scheduled before they are ready.
"""
function default_worker_pool()
    lock(POOL_LOCK) do
        for id in workers()
            id == 1 && continue
            id in INITIALIZING && continue
            id in GLOBAL_WORKER_POOL.workers || push!(GLOBAL_WORKER_POOL, id)
        end
    end
    return GLOBAL_WORKER_POOL
end

# ----------------------------------------------------------------------------
# Worker code-loading registry
#
# `@everywhere` only runs on workers that exist when it is called, so workers
# that join later (asynchronously) would not have the model code. Instead, we
# record setup expressions on the master and replay them on each worker as it
# joins (see `initialize_worker`). `@worker_setup` is a drop-in replacement for
# `@everywhere` that both applies now and persists for future workers.
# ----------------------------------------------------------------------------

# Ordered list of SOURCE_PATH-wrapped toplevel expressions to run on every worker.
const WORKER_SETUP = Expr[]

# Reimplementation of Distributed's internal `extract_imports`: pull out
# `using`/`import` statements so they can be run locally first (to precompile
# once on the master rather than racing across joining workers).
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

Record `ex` to run on every current and future worker, then apply it to all
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
    # Apply to processes that already exist (master + connected workers).
    Distributed.remotecall_eval(Main, procs(), wrapped)
    return nothing
end

"""
    @worker_setup expr

Like `Distributed.@everywhere`, but the expression is also recorded and replayed
on any worker that joins later. Use this instead of `@everywhere` to load model
code for an asynchronous [`WorkerBackend`](@ref) calibration, where workers may
join after the calibration has started.

`using`/`import` statements run on the master first (to precompile once), and
the current source path is propagated so relative `include` works on workers.
As with `@everywhere`, local variables must be interpolated with `\$`.
"""
macro worker_setup(ex)
    imps = _extract_imports(ex)
    return quote
        $(isempty(imps) ? nothing : Expr(:toplevel, map(esc, imps)...))
        $(register_worker_setup!)(
            $(QuoteNode(ex)),
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
"""
function initialize_worker(id)
    lock(POOL_LOCK) do
        push!(INITIALIZING, id)
    end
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
        # Only now is the worker schedulable. Guard against a duplicate channel
        # entry in case `default_worker_pool` already reconciled this worker.
        lock(POOL_LOCK) do
            id in GLOBAL_WORKER_POOL.workers || push!(GLOBAL_WORKER_POOL, id)
        end
        @info "Worker $id initialized and added to pool"
    catch e
        @warn "Worker $id failed to initialize; not added to pool" exception = e
    finally
        lock(POOL_LOCK) do
            delete!(INITIALIZING, id)
        end
    end
    return nothing
end

"""
    remove_worker_from_pool(id)

Remove worker `id` from `GLOBAL_WORKER_POOL`. Called when a worker
deregisters (e.g. walltime expiry or crash). Any stale entry left in the pool's
channel is filtered out at `take!` time.
"""
function remove_worker_from_pool(id)
    lock(POOL_LOCK) do
        delete!(GLOBAL_WORKER_POOL.workers, id)
        delete!(INITIALIZING, id)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Job teardown
#
# Workers are submitted as individual scheduler allocations (see `launch`). If
# the main process exits while some are still pending or running, those
# allocations would be orphaned. Every scheduler `launch` registers an `atexit`
# hook (`cancel_worker_jobs`) that cancels all of this session's jobs by their
# shared job name.
# ----------------------------------------------------------------------------

# Ensures the teardown `atexit` hook is registered at most once per session.
const ATEXIT_HOOK_REGISTERED = Ref(false)

# Run `cmd`, discarding its output.
_run_quiet(cmd) = run(pipeline(cmd; stdout = devnull, stderr = devnull))

# Ids of this session's PBS jobs, matched by the shared job name via `qselect`.
_pbs_worker_job_ids(jobname) = filter(
    !isempty,
    readlines(pipeline(`qselect -N $jobname`; stderr = devnull)),
)

"""
    cancel_worker_jobs(jobname = worker_jobname())

Cancel every scheduler job submitted for workers in this session with `scancel`
(Slurm) or `qdel` (PBS). This tears down both connected workers (by cancelling
their allocation) and any still-pending jobs.

Jobs submitted by [`add_workers`](@ref) share the job name
`worker_jobname`, so they are cancelled together. Safe to call when no
matching jobs exist.

Registered as an `atexit` hook whenever workers are launched onto a scheduler, so
that jobs are not orphaned when the main process exits. It may also be called
directly to tear down workers early.

!!! note
    This intentionally does *not* call `rmprocs`. `add_workers` runs `addprocs`
    on a background task that holds Distributed's global worker lock until every
    submitted job has connected (or been cancelled); `rmprocs` needs that same
    lock, so calling it here would deadlock whenever a job is still pending.
    Cancelling the scheduler jobs releases those workers directly.
"""
function cancel_worker_jobs(jobname = worker_jobname())
    try
        if is_slurm_available()
            _run_quiet(`scancel --name $jobname`)
        elseif is_pbs_available()
            ids = _pbs_worker_job_ids(jobname)
            isempty(ids) || _run_quiet(`qdel $ids`)
        end
    catch e
        @warn "Failed to cancel worker jobs named $jobname" exception = e
    end
    return nothing
end

# Register `cancel_worker_jobs` as an `atexit` hook exactly once, so jobs
# submitted in this session are cleaned up if the main process exits before the
# caller tears them down. Guarded by `POOL_LOCK` against a double registration.
function ensure_worker_atexit_hook!()
    lock(POOL_LOCK) do
        if !ATEXIT_HOOK_REGISTERED[]
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
worker_cookie_arg() = `--worker=$(worker_cookie())`

"""
    SlurmManager(ntasks=get(ENV, "SLURM_NTASKS", 1))

The ClusterManager for Slurm clusters, taking in the number of tasks to request
with `srun`.

To execute the `srun` command, run `addprocs(SlurmManager(ntasks))`.

Keyword arguments can be passed to `srun`: `addprocs(SlurmManager(ntasks),
gpus_per_task=1)`.

By default the workers will inherit the running Julia environment.

To run a calibration, call `calibrate(WorkerBackend(), ...)`.

To run functions on a worker, call `remotecall(func, worker_id, args...)`.
"""
struct SlurmManager <: ClusterManager
    ntasks::Integer

    function SlurmManager(ntasks = parse(Int, get(ENV, "SLURM_NTASKS", "1")))
        new(ntasks)
    end
end

# This function needs to exist
function Distributed.manage(
    manager::SlurmManager,
    id::Integer,
    config::WorkerConfig,
    op::Symbol,
)
    op == :register && initialize_worker(id)
    op == :deregister && remove_worker_from_pool(id)
    return nothing
end

# Main SlurmManager function, adapted from the unmaintained ClusterManagers.jl
function Distributed.launch(
    sm::SlurmManager,
    params::Dict,
    instances_arr::Array,
    c::Condition,
)
    # Ensure submitted jobs are cancelled if the main process exits.
    ensure_worker_atexit_hook!()
    params = add_default_worker_params(params)
    exehome = params[:dir]
    exename = params[:exename]
    exeflags = params[:exeflags]
    env = Dict{String, String}(params[:env])
    propagate_env_vars!(env)

    worker_args = parse_slurm_worker_params(params)
    # Get job file location from parameter dictionary
    job_directory = setup_job_directory(exehome, params)

    jobname = worker_jobname()
    submission_time = (trunc(Int, Base.time() * 10))
    output_base =
        get(params, :o, get(params, :output, ".$jobname-$submission_time"))

    ntasks = sm.ntasks
    # Submit each worker as an individual single-task allocation so they can be
    # scheduled independently
    pids = []
    output_files = String[]
    for i in 1:ntasks
        output_path = "$output_base-$i.out"
        srun_cmd = `srun -J $jobname -n 1 -D $exehome $worker_args -o $output_path -- $exename $exeflags $(worker_cookie_arg())`
        @info "Starting SLURM job $jobname [$i/$ntasks]: $srun_cmd"
        push!(pids, open(addenv(srun_cmd, env)))
        push!(output_files, output_path)
    end

    poll_files_for_worker_startup(output_files, pids, instances_arr, c)
end

"""
    parse_slurm_worker_params(params::Dict)

Parse params into string arguments for the worker launch command.

Uses all keys that are not in `Distributed.default_addprocs_params()`.
"""
function parse_slurm_worker_params(params::Dict)
    stdkeys = keys(Distributed.default_addprocs_params())
    worker_params =
        filter(x -> (!(x[1] in stdkeys) && x[1] != :job_file_loc), params)
    worker_args = []

    for (k, v) in worker_params
        if string(k) == "o" || string(k) == "output"
            continue
        end
        if length(string(k)) == 1
            push!(worker_args, "-$k")
            if length(v) > 0
                push!(worker_args, v)
            end
        else
            k2 = replace(string(k), "_" => "-")
            if length(v) > 0
                push!(worker_args, "--$k2=$v")
            else
                push!(worker_args, "--$k2")
            end
        end
    end
    return worker_args
end

worker_jobname() = "julia-$(getpid())"

function setup_job_directory(exehome::String, params::Dict)
    job_directory = joinpath(exehome, get(params, :job_file_loc, "."))
    !isdir(job_directory) && mkdir(job_directory)
    return job_directory
end

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

# Poll one output file per individually-submitted job, pushing each worker's
# `WorkerConfig` as it appears. Each job runs a single task, so each file
# produces exactly one worker.
#
# Tolerant of partial success: a job whose launch process errors, or that never
# starts within the polling window, is logged and skipped so that workers which
# did start remain usable. Throws only if no workers start at all.
function poll_files_for_worker_startup(output_files, pids, instances_arr, c)
    @assert length(output_files) == length(pids)
    ntasks = length(output_files)
    t_start = time()
    # This regex will match the worker's socket, ex: julia_worker:9015#169.254.3.1
    julia_worker_regex = r"([\w]+):([\d]+)#(\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3})"
    retry_delays = ExponentialBackOff(720, 1.0, 30.0, 1.5, 0.1)
    t_waited = 0
    registered = Set{Int}()   # indices of jobs whose worker has registered
    failed = Set{Int}()       # indices of jobs whose launch process errored

    for retry_delay in [0.0, retry_delays...]
        t_waited = round(Int, time() - t_start)
        for i in 1:ntasks
            (i in registered || i in failed) && continue
            pid = pids[i]
            if process_exited(pid) && pid.exitcode != 0
                @warn "Worker launch process for job $i/$ntasks exited with code $(pid.exitcode); skipping. Check the job scheduler output."
                push!(failed, i)
                continue
            end
            job_output_file = output_files[i]
            (isfile(job_output_file) && filesize(job_output_file) > 0) ||
                continue
            open(job_output_file) do f
                for line in eachline(f)
                    re_match = match(julia_worker_regex, line)
                    if !isnothing(re_match)
                        config = worker_config(re_match, pid)
                        push!(registered, i)
                        push!(instances_arr, config)
                        @info "Worker ready after $(t_waited)s on host $(config.host), port $(config.port) (job $i/$ntasks)"
                        notify(c)
                        break
                    end
                end
            end
        end
        # Stop once every job is accounted for (started or failed)
        (length(registered) + length(failed) == ntasks) && break
        # Sleep to limit resource usage while waiting for jobs to start
        sleep(retry_delay)
    end

    nregistered = length(registered)
    if nregistered < ntasks
        not_ready = sort(collect(setdiff(Set(1:ntasks), registered)))
        @warn "After $t_waited s, $nregistered/$ntasks workers started. Jobs not ready: $not_ready. Continuing with available workers."
    end
    if nregistered == 0
        throw(
            ErrorException(
                "No workers started after $t_waited s. Check the job scheduler output.",
            ),
        )
    end
    return nothing
end

function worker_config(worker_launch_details, pid)
    config = WorkerConfig()
    config.port = parse(Int, worker_launch_details[2])
    config.host = strip(worker_launch_details[3])
    config.userdata = pid
    return config
end

# TODO: Add examples of usage for SlurmManager and PBSManager in the docstrings
# Things like `addprocs(SlurmManager(2), t = "00:10:00",ngpus=4)`, then `remotecall` or `calibrate`
"""
    PBSManager(ntasks)

The ClusterManager for PBS/Torque clusters, taking in the number of tasks to
request with `qsub`.

To execute the `qsub` command, run `addprocs(PBSManager(ntasks))`. Unlike the
[`SlurmManager`](@ref), this will not nest scheduled jobs, but will acquire new
resources.

Keyword arguments can be passed to `qsub`: `addprocs(PBSManager(ntasks),
nodes=2)`

By default, the workers will inherit the running Julia environment.

To run a calibration, call `calibrate(WorkerBackend(), ...)`

To run functions on a worker, call `remotecall(func, worker_id, args...)`
"""
struct PBSManager <: ClusterManager
    ntasks::Integer
end

function Distributed.manage(
    manager::PBSManager,
    id::Integer,
    config::WorkerConfig,
    op::Symbol,
)
    op == :register && initialize_worker(id)
    op == :deregister && remove_worker_from_pool(id)
    return nothing
end

function Distributed.launch(
    pm::PBSManager,
    params::Dict,
    instances_arr::Array,
    c::Condition,
)
    # Ensure submitted jobs are cancelled if the main process exits.
    ensure_worker_atexit_hook!()
    params = add_default_worker_params(params)
    exehome = params[:dir]
    exename = params[:exename]
    exeflags = params[:exeflags]
    exeflags = exeflags == `` ? `--project=$(project_dir())` : exeflags
    env = Dict{String, String}(params[:env])
    propagate_env_vars!(env)

    worker_args = parse_pbs_worker_params(params)
    job_directory = setup_job_directory(exehome, params)
    jobname = worker_jobname()
    submission_time = (trunc(Int, Base.time() * 10))
    output_base = get(params, :o, ".$jobname-$submission_time")

    ntasks = pm.ntasks
    # Submit each worker as an individual single-resource allocation (no `-J`
    # job array) so they can be scheduled and join the pool independently.
    #= qsub options:
        -V: inherit environment variables
        -N: job name
        -j oe: Send the output and error streams to the same file
        -o: output file =#
    pids = []
    output_files = String[]
    for i in 1:ntasks
        output_path = "$output_base-$i.out"
        qsub_cmd = `qsub -V -N $jobname -j oe $worker_args -o $output_path -- $exename $exeflags $(worker_cookie_arg())`
        @info "Starting PBS job $jobname [$i/$ntasks]: $qsub_cmd"
        push!(pids, open(addenv(qsub_cmd, env)))
        push!(output_files, output_path)
    end

    poll_files_for_worker_startup(output_files, pids, instances_arr, c)
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
    excepted_keys = (:job_file_loc,)
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

Loads `Logging` and sets the global logger to log to `worker_\$worker_id.log`.
This function should be called from the worker process.
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
    return map_remotecall_fetch(workers) do worker
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
    backend_worker_kwargs(get_backend())...,
)
default_cpu_kwargs(::PBSManager) = (;
    l_select = "ncpus=1",
    l_walltime = format_pbs_time(DEFAULT_WALLTIME),
    backend_worker_kwargs(get_backend())...,
)

default_gpu_kwargs(::SlurmManager) = (;
    gpus_per_task = 1,
    cpus_per_task = 4,
    time = format_slurm_time(DEFAULT_WALLTIME),
    backend_worker_kwargs(get_backend())...,
)
default_gpu_kwargs(::PBSManager) = (;
    l_select = "ngpus=1:ncpus=4",
    l_walltime = format_pbs_time(DEFAULT_WALLTIME),
    backend_worker_kwargs(get_backend())...,
)

backend_worker_kwargs(::Type{DerechoBackend}) = (; q = "main", A = "UCIT0011")
backend_worker_kwargs(::Type{GCPBackend}) = (; partition = "a3")
backend_worker_kwargs(::Type{<:AbstractBackend}) = (;)

function get_manager(cluster = :auto, nworkers = 1)
    if cluster == :slurm || (cluster == :auto && is_slurm_available())
        SlurmManager(nworkers)
    elseif cluster == :pbs || (cluster == :auto && is_pbs_available())
        PBSManager(nworkers)
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
        kwargs...
    )

Add `nworkers` worker processes to the current Julia session, automatically
detecting and configuring for the available computing environment.

This does not wait for the workers to connect. Each worker is submitted as an
individual allocation and adds itself to `GLOBAL_WORKER_POOL` once it
has started and loaded its code, so a calibration can begin with an empty pool
and pick up workers as they join.

The returned `Task` runs the (blocking) submission; `wait` on it to block until
all submissions have been processed. Submitted jobs are cancelled automatically
when the process exits (via an `atexit` hook); call [`cancel_worker_jobs`](@ref)
to tear them down earlier.

Use [`@worker_setup`](@ref) (instead of `@everywhere`) to load model code so
that workers joining later are initialized correctly.

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
- `kwargs`: Other kwargs can be passed directly through to `addprocs`.
"""
function add_workers(
    nworkers;
    device = :gpu,
    cluster = :auto,
    time = DEFAULT_WALLTIME,
    kwargs...,
)
    return errormonitor(
        Threads.@spawn _add_workers(nworkers; device, cluster, time, kwargs...)
    )
end

function _add_workers(nworkers; device, cluster, time, kwargs...)
    if cluster == :local || (cluster == :auto && !is_cluster_environment())
        @info "Using local processing mode, adding $nworkers worker$(nworkers == 1 ? "" : "s")"

        ids = addprocs(nworkers; kwargs...)

        @sync for id in ids
            @async initialize_worker(id)
        end

        return ids
    end

    manager = get_manager(cluster, nworkers)
    @info "Using $(nameof(typeof(manager))) to add $nworkers workers"

    default_kwargs =
        device == :gpu ? default_gpu_kwargs(manager) :
        device == :cpu ? default_cpu_kwargs(manager) :
        throw(ArgumentError("device must be :gpu or :cpu, got $(repr(device))"))

    normalized_kwargs = process_time_parameter(manager, time, kwargs)
    merged_kwargs = merge(default_kwargs, normalized_kwargs)

    return addprocs(manager; merged_kwargs...)
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
    # For other manager types, just pass through the kwargs unchanged
    return kwargs
end

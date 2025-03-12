using Distributed
using Logging

export SlurmManager, PBSManager, set_worker_loggers

get_worker_pool() = workers() == [1] ? WorkerPool() : default_worker_pool()

function run_worker_iteration(
    iter,
    ensemble_size,
    output_dir;
    failure_rate = DEFAULT_FAILURE_RATE,
)
    nfailures = 0
    worker_pool = get_worker_pool()
    all_known_workers = get_worker_pool()

    work_to_do = map(1:ensemble_size) do m
        (w) -> begin
            @info "Running particle $m on worker $w"
            remotecall_wait(forward_model, w, iter, m)
        end
    end

    @sync while !isempty(work_to_do)
        # Add new workers to worker_pool
        all_workers = get_worker_pool()
        new_workers = setdiff(all_workers.workers, all_known_workers.workers)
        foreach(x -> push!(worker_pool, x), new_workers)
        all_known_workers = all_workers
        if !isempty(worker_pool.workers)
            worker = take!(worker_pool)
            run_fwd_model = pop!(work_to_do)
            @async try
                run_fwd_model(worker)
            catch e
                @warn "Error running on worker $worker" exception = e
                nfailures += 1
            finally
                push!(worker_pool, worker)
            end
        else
            println("no workers available")
            sleep(10) # Wait for workers to become available
        end
    end

    iter_failure_rate = nfailures / ensemble_size
    if iter_failure_rate > failure_rate
        error(
            "Ensemble for iter $iter had a $(iter_failure_rate * 100)% failure rate",
        )
    end
end

worker_cookie() = begin
    Distributed.init_multi()
    cluster_cookie()
end
worker_cookie_arg() = `--worker=$(worker_cookie())`

"""
    SlurmManager(; ntasks=get(ENV, "SLURM_NTASKS", 1), expr=:())

A ClusterManager implementation for Slurm job scheduling systems. Workers inherit the current Julia environment by default.

# Arguments
- `ntasks::Integer`: Number of tasks to allocate via `srun` (defaults to SLURM_NTASKS environment variable or 1)
- `expr::Expr`: Expression to evaluate on each worker at initialization

# Usage
```julia
# Add workers using default settings
addprocs(SlurmManager(ntasks=4))

# Pass additional arguments to `srun`
addprocs(SlurmManager(ntasks=4), gpus_per_task=1)

# Related functions
- `calibrate(WorkerBackend, ...)`: Perform calibration using workers
- `remotecall(func, worker_id, args...)`: Execute functions on specific workers
"""
struct SlurmManager <: ClusterManager
    ntasks::Integer
    expr::Expr

    function SlurmManager(
        ntasks = parse(Int, get(ENV, "SLURM_NTASKS", "1"));
        expr = :(),
    )
        new(ntasks, expr)
    end
end

function Distributed.manage(
    manager::SlurmManager,
    id::Integer,
    config::WorkerConfig,
    op::Symbol,
)
    if op == :register
        set_worker_logger(id)
        evaluate_initial_expression(id, manager.expr)
    end
end

function evaluate_initial_expression(id, expr)
    try
        Distributed.remotecall_eval(Main, id, expr)
    catch e
        @error "Initial worker expression errored:" exception = e
    end
end

function set_worker_logger(id)
    Distributed.remotecall_eval(Main, id, :(using ClimaCalibrate, Logging))
    remotecall(id) do
        ClimaCalibrate.set_worker_logger()
    end
end

# Main SlurmManager function, adapted from the unmaintained ClusterManagers.jl
function Distributed.launch(
    sm::SlurmManager,
    params::Dict,
    instances_arr::Array,
    c::Condition,
)
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
    default_output = ".$jobname-$submission_time.out"
    output_path = get(params, :o, get(params, :output, default_output))

    ntasks = sm.ntasks
    srun_cmd = `srun -J $jobname -n $ntasks -D $exehome $worker_args -o $output_path -- $exename $exeflags $(worker_cookie_arg())`
    @info "Starting SLURM job $jobname: $srun_cmd"
    pid = open(addenv(srun_cmd, env))

    poll_file_for_worker_startup(output_path, ntasks, pid, instances_arr, c)
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

# Poll a single file for multiple workers
function poll_file_for_worker_startup(
    job_output_file::String,
    ntasks::Int,
    pid,
    instances_arr,
    c,
)
    t_start = time()
    # This regex will match the worker's socket, ex: julia_worker:9015#169.254.3.1
    julia_worker_regex = r"([\w]+):([\d]+)#(\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3})"
    retry_delays = ExponentialBackOff(720, 1.0, 30.0, 1.5, 0.1)
    t_waited = nothing
    registered_workers = Set{String}()

    for retry_delay in [0.0, retry_delays...]
        if process_exited(pid) && pid.exitcode != 0
            error(
                """Worker launch process exited with code $(pid.exitcode).
          Please check the terminal for error messages from the job scheduler.""",
            )
        end
        t_waited = round(Int, time() - t_start)
        # Wait for output log to be created and populated, then parse
        if isfile(job_output_file)
            if filesize(job_output_file) > 0
                open(job_output_file) do f
                    for line in eachline(f)
                        re_match = match(julia_worker_regex, line)
                        if !isnothing(re_match) && !(line in registered_workers)
                            config = worker_config(re_match, pid)
                            push!(registered_workers, line)
                            push!(instances_arr, config)
                            @info "Worker ready after $(t_waited)s on host $(config.host), port $(config.port)"
                            notify(c)
                        end
                    end
                end
            end
            length(registered_workers) == ntasks && break
        else
            @info "Worker launch (after $t_waited s): No output file \"$job_output_file\" yet"
        end
        # Sleep for some time to limit resource usage while waiting for the job to start
        sleep(retry_delay)
    end

    if length(registered_workers) != ntasks
        throw(
            ErrorException(
                "Timeout after $t_waited s while waiting for worker(s) to get ready.",
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
    PBSManager(ntasks; expr=:())

A ClusterManager implementation for PBS/Torque job scheduling systems.

# Arguments
- `ntasks::Integer`: Number of tasks to request via `qsub`
- `expr::Expr`: Expression to evaluate on each worker at initialization

# Usage
```julia
# Add workers using default settings
addprocs(PBSManager(4))

# Pass additional arguments to `qsub`
addprocs(PBSManager(4), nodes=2)

# Use resource list options with l_ prefix
addprocs(PBSManager(4), l_walltime="00:10:00", l_select="2:ncpus=4:mem=8gb")

# Specify initialization expression
addprocs(PBSManager(4, expr=:(using MyPackage)))
```

Unlike the [`SlurmManager`](@ref), this will not nest scheduled jobs but will acquire new resources.
Workers inherit the current Julia environment by default.

# Resource Specification
- Parameters with `l_` prefix are passed to qsub's `-l` option (e.g., `l_walltime="00:10:00"` becomes `-l walltime=00:10:00`)
- Standard PBS/Torque options can be passed directly as keyword arguments

# Related Functions
- `calibrate(WorkerBackend, ...)`: Perform worker calibration
- `remotecall(func, worker_id, args...)`: Execute functions on specific workers

See also: [`addprocs`](@ref), [`Distributed`](@ref), [`SlurmManager`](@ref)
"""
struct PBSManager <: ClusterManager
    ntasks::Integer
    expr::Expr

    function PBSManager(ntasks; expr = :())
        new(ntasks, expr)
    end
end

function Distributed.manage(
    manager::PBSManager,
    id::Integer,
    config::WorkerConfig,
    op::Symbol,
)
    if op == :register
        working_dir = pwd()
        remotecall(cd, id, working_dir)
        set_worker_logger(id)
        evaluate_initial_expression(id, manager.expr)
    end
end

function Distributed.launch(
    pm::PBSManager,
    params::Dict,
    instances_arr::Array,
    c::Condition,
)
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

    ntasks = pm.ntasks
    default_output = ".$jobname-$submission_time.out"
    job_array_option = ntasks > 1 ? `-J 1-$ntasks` : ``
    output_path = get(params, :o, default_output)
    #= qsub options:
        -V: inherit environment variables
        -N: job name
        -j oe: Send the output and error streams to the same file
        -J 1-ntasks: Job array
        -o: output file =#
    qsub_cmd = `qsub -V -N $jobname -j oe $job_array_option $worker_args -o $output_path -- $exename $exeflags $(worker_cookie_arg())`
    @info "Starting PBS job $jobname: $qsub_cmd"
    pid = open(addenv(qsub_cmd, env))

    poll_file_for_worker_startup(output_path, ntasks, pid, instances_arr, c)
end

"""
    parse_pbs_worker_params(params::Dict)

Parse params into string arguments for the worker launch command.

Uses all keys that are not in `Distributed.default_addprocs_params()`.
Keys that start with `l_` will be treated as `-l` arguments to `qsub`. For example, l_walltime = "00:10:00" is transformed into `-l walltime=00:10:00`.
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
            ClimaCalibrate.set_worker_logger()
        end
    end
end

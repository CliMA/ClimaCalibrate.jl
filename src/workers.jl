using Distributed
import EnsembleKalmanProcesses as EKP
export SlurmManager, default_worker_pool

default_worker_pool() = WorkerPool(workers())

function run_worker_iteration(
    iter,
    ensemble_size,
    output_dir;
    worker_pool,
    failure_rate,
)
    # Create a channel to collect results
    results = Channel{Any}(ensemble_size)
    nfailures = 0
    @sync begin
        for m in 1:(ensemble_size)
            @async begin
                worker = take!(worker_pool)
                @info "Running particle $m on worker $worker"
                try
                    remotecall_wait(forward_model, worker, iter, m)
                catch e
                    @warn "Error running member $m" exception = e
                    nfailures += 1
                finally
                    # Always return worker to pool
                    put!(worker_pool, worker)
                end
            end
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
worker_arg() = `--worker=$(worker_cookie())`

"""
    SlurmManager(ntasks=get(ENV, "SLURM_NTASKS", 1))

The ClusterManager for Slurm clusters, taking in the number of tasks to request with `srun`.
To execute the `srun` command, run `addprocs(SlurmManager(ntasks))`
Keyword arguments can be passed to `srun`: `addprocs(SlurmManager(ntasks), gpus_per_task=1)`
By default the workers will inherit the running Julia environment.
To run a calibration, call `calibrate(WorkerBackend, ...)`
To run functions on a worker, call `remotecall(func, worker_id, args...)`
"""
struct SlurmManager <: ClusterManager
    ntasks::Integer

    function SlurmManager(
        ntasks::Integer = parse(Int, get(ENV, "SLURM_NTASKS", "1")),
    )
        new(ntasks)
    end
end

function Distributed.manage(
    manager::SlurmManager,
    id::Integer,
    config::WorkerConfig,
    op::Symbol,
)
    # This function needs to exist, but so far we don't do anything
end

# Main SlurmManager function, mostly copied from the unmaintained ClusterManagers.jl
# Original code: https://github.com/JuliaParallel/ClusterManagers.jl
# TODO: Log per member
function Distributed.launch(
    sm::SlurmManager,
    params::Dict,
    instances_arr::Array,
    c::Condition,
)
    default_params = Distributed.default_addprocs_params()
    params = merge(default_params, Dict{Symbol, Any}(params))
    exehome = params[:dir]
    exename = params[:exename]
    exeflags = params[:exeflags]
    env = Dict{String, String}(params[:env])

    # Taken from Distributed.LocalManager
    propagate_env_vars!(env)

    worker_args = parse_worker_params(params)
    # Get job file location from parameter dictionary
    job_directory = setup_job_directory(exehome, params)

    output_path_func = configure_output_args!(job_directory, worker_args)

    ntasks = sm.ntasks
    jobname = worker_jobname()
    srun_cmd = `srun -J $jobname -n $ntasks -D $exehome $(worker_args) $exename $exeflags $(worker_arg())`

    @info "Starting SLURM job $jobname: $srun_cmd"
    srun_pid = open(addenv(srun_cmd, env))

    # Wait for workers to start
    t_start = time()
    for i in 0:(ntasks - 1)
        output_file = output_path_func(lpad(i, 4, "0"))
        worker_config = poll_worker_startup(output_file, i, t_start, srun_pid)
        # Add configs to `instances_arr` for internal Distributed use
        push!(instances_arr, worker_config)
        notify(c)
    end
end

"""
    parse_worker_params(params::Dict)

Parse params into string arguments for the worker launch command.

Uses all keys that are not in `Distributed.default_addprocs_params()`.
"""
function parse_worker_params(params::Dict)
    stdkeys = keys(Distributed.default_addprocs_params())
    worker_params =
        filter(x -> (!(x[1] in stdkeys) && x[1] != :job_file_loc), params)
    worker_args = []

    for (k, v) in worker_params
        if length(string(k)) == 1
            push!(worker_args, "-$k")
            if length(v) > 0
                push!(worker_args, v)
            end
        else
            k2 = replace(string(k), "_" => "-")
            if length(v) > 0
                push!(worker_args, "--$k2=$v)")
            else
                push!(worker_args, "--$k2")
            end
        end
    end
    return worker_args
end


worker_jobname() = "julia-$(getpid())"

function configure_output_args!(job_directory, worker_args)
    default_template = ".$(worker_jobname())-$(trunc(Int, Base.time() * 10))"
    default_output(x) = joinpath(job_directory, "$default_template-$x.out")

    if any(arg -> occursin("-o", arg) || occursin("--output", arg), worker_args)
        # if has_output_name, ensure there is only one output arg
        locs = findall(
            x -> startswith(x, "-o") || startswith(x, "--output"),
            worker_args,
        )
        length(locs) > 1 &&
            error("Slurm Error: Multiple output files specified: $worker_args")
        job_output_file = worker_args[locs[1] + 1]
        return i -> job_output_file
    else
        # Slurm interpolates %4t to the task ID padded with up to four zeros
        push!(worker_args, "-o", default_output("%4t"))
        return default_output
    end
end

function setup_job_directory(exehome::String, params::Dict)
    job_directory = joinpath(exehome, get(params, :job_file_loc, "."))
    !isdir(job_directory) && mkdir(job_directory)
    return job_directory
end

function propagate_env_vars!(env)
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

function poll_worker_startup(
    job_output_file::String,
    worker_index::Int,
    t_start::Float64,
    pid,
)
    # This Regex will match the worker's socket and IP address
    # Example: julia_worker:9015#169.254.3.1
    julia_worker_regex = r"([\w]+):([\d]+)#(\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3})"
    could_not_connect_regex = r"could not connect"
    exiting_regex = r"exiting."
    worker_launch_details = nothing
    worker_errors = String[]
    retry_delays = ExponentialBackOff(10, 1.0, 512.0, 2.0, 0.1)
    t_waited = nothing
    for retry_delay in push!(collect(retry_delays), 0)
        t_waited = round(Int, time() - t_start)

        # Wait for output log to be created and populated, then parse
        if isfile(job_output_file)
            if filesize(job_output_file) > 0
                open(job_output_file) do f
                    # Due to error and warning messages, we need to check
                    # for a regex match on each line
                    for line in eachline(f)
                        re_match = match(julia_worker_regex, line)
                        if !isnothing(re_match)
                            worker_launch_details = re_match
                            break # We have found the match
                        end
                        for expr in [could_not_connect_regex, exiting_regex]
                            if !isnothing(match(expr, line))
                                worker_launch_details = nothing
                                push!(worker_errors, line)
                            end
                        end
                    end
                end
            end
            if !isempty(worker_errors) || !isnothing(worker_launch_details)
                break   # break if error or specification found
            else
                @info "Worker $worker_index (after $t_waited s): Output file found, but no connection details yet"
            end
        else
            @info "Worker $worker_index (after $t_waited s): No output file \"$job_output_file\" yet"
        end

        # Sleep for some time to limit resource usage while waiting for the job to start
        sleep(retry_delay)
    end

    if !isempty(worker_errors)
        throw(
            ErrorException(
                "Worker $worker_index failed after $t_waited s: $(join(worker_errors, " "))",
            ),
        )
    elseif isnothing(worker_launch_details)
        throw(
            ErrorException(
                "Timeout after $t_waited s while waiting for worker $worker_index to get ready.",
            ),
        )
    end
    config = WorkerConfig()
    config.port = parse(Int, worker_launch_details[2])
    config.host = strip(worker_launch_details[3])
    config.userdata = pid
    # Keep a reference to the proc, so it's properly closed once
    # the last worker exits.
    @info "Worker $worker_index ready after $t_waited s on host $(config.host), port $(config.port)"
    return config
end

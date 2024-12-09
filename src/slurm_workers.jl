using Distributed
import EnsembleKalmanProcesses as EKP
export worker_calibrate, add_slurm_workers

function run_iteration(iter, ensemble_size, output_dir; worker_pool = default_worker_pool(), failure_rate = 0.5)
    # Create a channel to collect results
    results = Channel{Any}(ensemble_size)
    nfailures = 0
    @sync begin
        for m in 1:(ensemble_size)
            @async begin
                worker = take!(worker_pool)
                @info "Running particle $m on worker $worker"
                try
                    remotecall( forward_model, worker, m, iter)
                catch e
                    @error "Error running member $m" exception = e
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
        error("Ensemble for iter $iter had a $(iter_failure_rate * 100)% failure rate")
    end
end

function worker_calibrate(config; failure_rate = 0.5, worker_pool = default_worker_pool(), ekp_kwargs...)
    (; ensemble_size, n_iterations, observations, noise, prior, output_dir) = config
    return worker_calibrate(ensemble_size, n_iterations, observations, noise, prior, output_dir; failure_rate, worker_pool, ekp_kwargs...)
end

function worker_calibrate(ensemble_size, n_iterations, observations, noise, prior, output_dir; failure_rate = 0.5, worker_pool = default_worker_pool(), ekp_kwargs...)
    initialize(
        ensemble_size,
        observations,
        noise,
        prior,
        output_dir;
        rng_seed = 1234,
        ekp_kwargs...,
    )
    for iter in 0:(n_iterations)
        (; time) = @timed run_iteration(iter, config; worker_pool, failure_rate)
        @info "Iteration $iter time: $time"
        # Process results
        G_ensemble = observation_map(iter)
        save_G_ensemble(output_dir, iter, G_ensemble)
        update_ensemble(output_dir, iter, prior)
        iter_path = path_to_iteration(output_dir, iter)
    end
    return JLD2.load_object(joinpath(path_to_iteration(output_dir, n_iterations)), "eki_file.jld2")
end

function worker_calibrate(ekp::EKP.EnsembleKalmanProcess, ensemble_size,n_iterations, observations, noise, prior, output_dir; failure_rate = 0.5, worker_pool = default_worker_pool(), ekp_kwargs...)
    initialize(
        ekp, prior, output_dir
        ;
        rng_seed = 1234,
    )
    for iter in 0:(n_iterations)
        (; time) = @timed run_iteration(iter, ensemble_size, output_dir; worker_pool, failure_rate)
        @info "Iteration $iter time: $time"
        # Process results
        G_ensemble = observation_map(iter)
        save_G_ensemble(output_dir, iter, G_ensemble)
        update_ensemble(output_dir, iter, prior)
        iter_path = path_to_iteration(output_dir, iter)
    end
    return JLD2.load_object(path_to_iteration(output_dir, n_iterations))
end

worker_cookie() = begin Distributed.init_multi(); cluster_cookie() end
worker_arg() = `--worker=$(worker_cookie())`

struct SlurmManager <: ClusterManager
    ntasks::Integer
end

function Distributed.manage(manager::SlurmManager, id::Integer, config::WorkerConfig,
    op::Symbol)
# This function needs to exist, but so far we don't do anything
end

function Distributed.launch(sm::SlurmManager,params::Dict, instances_arr::Array,
    c::Condition)
    default_params = Distributed.default_addprocs_params()
    params = merge(default_params, Dict{Symbol, Any}(params))
    exehome = params[:dir]
    exename = params[:exename]
    exeflags = params[:exeflags]

    stdkeys = keys(Distributed.default_addprocs_params())

    slurm_params = filter(x->(!(x[1] in stdkeys) && x[1] != :job_file_loc), params)

    srunargs = []

    for k in keys(slurm_params)
        if length(string(k)) == 1
            push!(srunargs, "-$k")
            val = p[k]
            if length(val) > 0
                push!(srunargs, "$(p[k])")
            end
        else
            k2 = replace(string(k), "_"=>"-")
            val = p[k]
            if length(val) > 0
                push!(srunargs, "--$(k2)=$(p[k])")
            else
                push!(srunargs, "--$(k2)")
            end
        end
    end
    # Get job file location from parameter dictionary.
    job_file_loc = joinpath(exehome, get(params, :job_file_loc, "."))

    # Make directory if not already made.
    if !isdir(job_file_loc)
        mkdir(job_file_loc)
    end

    # Check for given output file name
    jobname = "julia-$(getpid())"
    default_template = ".$jobname-$(trunc(Int, Base.time() * 10))"
    default_output(x) = "$default_template-$x.out"

    # Set output name
    has_output_name = ("-o" in srunargs) | ("--output" in srunargs)
    job_output_file = if has_output_name
        # if has_output_name, ensure there is only one output arg
        loc = findfirst(x-> x == "-o" || x == "--output", srunargs)
        job_output = srunargs[loc+1]
        # Remove output argument to reappend
        filter!(x -> x != "-o" && x != "--output", srunargs)
        filter!(x -> !occursin(r"^-[oe]", x), srunargs)
        job_output
    else
        # Slurm interpolates %4t to the task ID padded with up to four zeros
        default_output("%4t")
    end
    push!(srunargs, "-o", job_output_file)
    ntasks = sm.ntasks
    srun_cmd = `srun -J $jobname -n $ntasks -D $exehome $(srunargs) $exename $exeflags $(worker_arg())`

    @info "Starting SLURM job $jobname: $srun_cmd"
    srun_proc = open(srun_cmd)

    slurm_spec_regex = r"([\w]+):([\d]+)#(\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3})"
    could_not_connect_regex = r"could not connect"
    exiting_regex = r"exiting."

    # Wait for workers to start
    t_start = time()
    t_waited = round(Int, time() - t_start)
    retry_delays = ExponentialBackOff(10, 1.0, 512.0, 2.0, 0.1)
    for i in 0:ntasks - 1
        slurm_spec_match::Union{RegexMatch,Nothing} = nothing
        worker_errors = String[]
        if !has_output_name
            job_output_file = default_output("0000")
        end
        for retry_delay in push!(collect(retry_delays), 0)
            t_waited = round(Int, time() - t_start)

            # Wait for output log to be created and populated, then parse

            if isfile(job_output_file)
                if filesize(job_output_file) > 0
                    open(job_output_file) do f
                        # Due to error and warning messages, the specification
                        # may not appear on the file's first line
                        for line in eachline(f)
                            re_match = match(slurm_spec_regex, line)
                            if !isnothing(re_match)
                                slurm_spec_match = re_match
                            end
                            for expr in [could_not_connect_regex, exiting_regex]
                                if !isnothing(match(expr, line))
                                    slurm_spec_match = nothing
                                    push!(worker_errors, line)
                                end
                            end
                        end
                    end
                end
                if !isempty(worker_errors) || !isnothing(slurm_spec_match)
                    break   # break if error or specification found
                else
                    @info "Worker $i (after $t_waited s): Output file found, but no connection details yet"
                end
            else
                @info "Worker $i (after $t_waited s): No output file \"$job_output_file\" yet"
            end

            # Sleep for some time to limit resource usage while waiting for the job to start
            sleep(retry_delay)
        end

        if !isempty(worker_errors)
            throw(SlurmException("Worker $i failed after $t_waited s: $(join(worker_errors, " "))"))
        elseif isnothing(slurm_spec_match)
            throw(SlurmException("Timeout after $t_waited s while waiting for worker $i to get ready."))
        end

        config = WorkerConfig()
        config.port = parse(Int, slurm_spec_match[2])
        config.host = strip(slurm_spec_match[3])
        @info "Worker $i ready after $t_waited s on host $(config.host), port $(config.port)"
        # Keep a reference to the proc, so it's properly closed once
        # the last worker exits.
        config.userdata = srun_proc
        push!(instances_arr, config)
        notify(c)
    end
end

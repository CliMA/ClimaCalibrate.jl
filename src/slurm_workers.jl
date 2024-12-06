using Distributed, ClusterManagers
import EnsembleKalmanProcesses as EKP

#=
 `srun -J julia-4014777 -n 10 -D /home/nefrathe/clima/ClimaCalibrate.jl 
 --cpus-per-task=1 
 -t 00:20:00 -o /home/nefrathe/clima/ClimaCalibrate.jl/./julia-4014777-17333586001-%4t.out 
    /clima/software/julia/julia-1.11.0/bin/julia 
    --project=/home/nefrathe/clima/ClimaCalibrate.jl/Project.toml 
    --worker=U3knisg2TJufcrbJ`
=#


# srun_proc = open(srun_cmd)

# worker_cookie() = begin Distributed.init_multi(); cluster_cookie() end
# worker_arg() = `--worker=$(worker_cookie())`

# srun_cmd = `srun -J $jobname -n $np -D $exehome $(srunargs) $exename $exeflags $(worker_arg())`

default_worker_pool() = WorkerPool(workers())

function run_iteration(iter, ensemble_size, output_dir; worker_pool = default_worker_pool(), failure_rate = 0.5)
    # Create a channel to collect results
    results = Channel{Any}(ensemble_size)
    @sync begin
        for m in 1:(ensemble_size)
            @async begin
                # Get a worker from the pool
                worker = take!(worker_pool)
                try
                    model_config = set_up_forward_model(m, iter, config)
                    result = remotecall_fetch(
                        run_forward_model,
                        worker,
                        model_config,
                    )
                    put!(results, (m, result))
                catch e
                    @error "Error running member $m" exception = e
                    put!(results, (m, e))
                finally
                    # Always return worker to pool
                    put!(worker_pool, worker)
                end
            end
        end
    end

    # Collect all results
    ensemble_results = Dict{Int, Any}()
    for _ in 1:(ensemble_size)
        m, result = take!(results)
        if result isa Exception
            @error "Member $m failed" error = result
        else
            ensemble_results[m] = result
        end
    end
    results = values(ensemble_results)
    iter_failure_rate = sum(isa.(results, Exception)) / ensemble_size
    if iter_failure_rate > failure_rate
        error("Ensemble for iter $iter had a $(iter_failure_rate * 100)% failure rate")
    end
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


function worker_calibrate(config; worker_pool = default_worker_pool())
    (; ensemble_size, observations, noise, prior, output_dir) = config
    return worker_calibrate(ensemble_size, observations, noise, prior, output_dir; worker_pool)
end

function slurm_worker_pool(nprocs::Int; slurm_kwargs...)
    return WorkerPool(addprocs(
        SlurmManager(nprocs);
        t = "01:00:00", cpus_per_task = 1,
        exeflags = "--project=$(Base.active_project())",
        slurm_kwargs...,
    ))
end

# gpus_per_task=1
worker_pool = default_worker_pool()

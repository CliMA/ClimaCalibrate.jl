using Distributed

import EnsembleKalmanProcesses as EKP

export get_backend, calibrate, model_run

export JuliaBackend, WorkerBackend
export HPCBackend, ClimaGPUBackend, DerechoBackend, CaltechHPCBackend

abstract type AbstractBackend end

struct JuliaBackend <: AbstractBackend end

abstract type HPCBackend <: AbstractBackend end
abstract type SlurmBackend <: HPCBackend end

struct CaltechHPCBackend <: SlurmBackend end
struct ClimaGPUBackend <: SlurmBackend end

struct DerechoBackend <: HPCBackend end

struct WorkerBackend <: AbstractBackend end

"""
    get_backend()

Get ideal backend for deploying forward model runs. 
Each backend is found via `gethostname()`. Defaults to JuliaBackend if none is found.
"""
function get_backend()
    HOSTNAMES = [
        (r"^clima.gps.caltech.edu$", ClimaGPUBackend),
        (r"^login[1-4].cm.cluster$", CaltechHPCBackend),
        (r"^hpc-(\d\d)-(\d\d).cm.cluster$", CaltechHPCBackend),
        (r"derecho([1-8])$", DerechoBackend),
        (r"dec(\d\d\d\d)$", DerechoBackend), # This should be more specific
    ]

    for (pattern, backend) in HOSTNAMES
        !isnothing(match(pattern, gethostname())) && return backend
    end

    return JuliaBackend
end

"""
    module_load_string(backend)

Return a string that loads the correct modules for a given backend when executed via bash.
"""
function module_load_string(::Type{CaltechHPCBackend})
    return """export MODULEPATH="/groups/esm/modules:\$MODULEPATH"
    module purge
    module load climacommon/2024_10_09"""
end

function module_load_string(::Type{ClimaGPUBackend})
    return """module purge
    module load julia/1.11.0 cuda/julia-pref openmpi/4.1.5-mpitrampoline"""
end

function module_load_string(::Type{DerechoBackend})
    return """export MODULEPATH="/glade/campaign/univ/ucit0011/ClimaModules-Derecho:\$MODULEPATH" 
    module purge
    module load climacommon
    module list
    """
end

calibrate(
    config::ExperimentConfig;
    model_interface = nothing,
    hpc_kwargs = Dict(),
    ekp_kwargs...,
) = calibrate(get_backend(), config; model_interface, hpc_kwargs, ekp_kwargs...)

function calibrate(
    ensemble_size::Int,
    n_iterations::Int,
    observations,
    noise,
    prior,
    output_dir;
    model_interface = nothing,
    hpc_kwargs = Dict(),
    ekp_kwargs...,
)
    return calibrate(
        get_backend(),
        ensemble_size,
        n_iterations,
        observations,
        noise,
        prior,
        output_dir;
        model_interface,
        hpc_kwargs,
        ekp_kwargs...,
    )
end

function calibrate(
    ::Type{JuliaBackend},
    config::ExperimentConfig;
    ekp_kwargs...,
)
    (; n_iterations, output_dir, ensemble_size) = config
    ekp = initialize(config; ekp_kwargs...)
    on_error(e::InterruptException) = rethrow(e)
    on_error(e) =
        @error "Single ensemble member has errored. See stacktrace" exception =
            (e, catch_backtrace())
    for i in 0:(n_iterations - 1)
        @info "Running iteration $i"
        pmap(1:ensemble_size; retry_delays = 0, on_error) do m
            forward_model(i, m)
            @info "Completed member $m"
        end
        G_ensemble = observation_map(i)
        save_G_ensemble(config, i, G_ensemble)
        terminate = update_ensemble(config, i)
        !isnothing(terminate) && break
        iter_path = path_to_iteration(output_dir, i + 1)
        ekp = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    end
    return ekp
end

const DEFAULT_FAILURE_RATE = 0.5

"""
    calibrate(backend, ensemble_size, n_iterations, observations, noise, prior, output_dir; ekp_kwargs...)
    calibrate(backend, ekp::EnsembleKalmanProcess, ensemble_size, n_iterations, prior, output_dir)
    calibrate(backend, config::ExperimentConfig; ekp_kwargs...)

Run a full calibration on the given backend.

If the EKP struct is not given, it will be constructed upon initialization. 
The experiment configuration (ensemble size, prior, observations, etc) can be 
wrapped in an ExperimentConfig or passed in as arguments to the function.

Available Backends: WorkerBackend, CaltechHPCBackend, ClimaGPUBackend, DerechoBackend, JuliaBackend

Derecho, ClimaGPU, and CaltechHPC backends are designed to run on a specific high-performance computing cluster.
WorkerBackend uses Distributed.jl to run the forward model on workers.

## Keyword Arguments for HPC backends
- `model_interface: Path to the model interface file.
- `hpc_kwargs`: Dictionary of resource arguments for HPC clusters, passed to the job scheduler.
- `verbose::Bool`: Enable verbose logging.
- Any keyword arguments for the EnsembleKalmanProcess constructor, such as `scheduler`
"""
function calibrate(
    b::Type{WorkerBackend},
    config::ExperimentConfig;
    failure_rate = DEFAULT_FAILURE_RATE,
    worker_pool = default_worker_pool(),
    ekp_kwargs...,
)
    (; ensemble_size, n_iterations, observations, noise, prior, output_dir) =
        config
    return calibrate(
        b,
        ensemble_size,
        n_iterations,
        observations,
        noise,
        prior,
        output_dir;
        failure_rate,
        worker_pool,
        ekp_kwargs...,
    )
end

function calibrate(
    b::Type{WorkerBackend},
    ensemble_size::Int,
    n_iterations::Int,
    observations,
    noise,
    prior,
    output_dir;
    failure_rate = DEFAULT_FAILURE_RATE,
    worker_pool = default_worker_pool(),
    ekp_kwargs...,
)
    eki = ekp_constructor(ensemble_size, prior, observations, noise)
    return calibrate(
        b,
        eki,
        ensemble_size,
        n_iterations,
        prior,
        output_dir;
        worker_pool,
        ekp_kwargs...,
    )
end

function calibrate(
    b::Type{WorkerBackend},
    ekp::EKP.EnsembleKalmanProcess,
    ensemble_size,
    n_iterations,
    prior,
    output_dir;
    failure_rate = DEFAULT_FAILURE_RATE,
    worker_pool = default_worker_pool(),
)
    initialize(ekp, prior, output_dir)
    for iter in 0:n_iterations
        (; time) = @timed run_worker_iteration(
            iter,
            ensemble_size,
            output_dir;
            worker_pool,
            failure_rate,
        )
        @info "Iteration $iter time: $time"
        # Process results
        G_ensemble = observation_map(iter)
        save_G_ensemble(output_dir, iter, G_ensemble)
        update_ensemble(output_dir, iter, prior)
        iter_path = path_to_iteration(output_dir, iter)
    end
    return JLD2.load_object(
        joinpath(path_to_iteration(output_dir, n_iterations), "eki_file.jld2"),
    )
end

function calibrate(
    b::Type{<:HPCBackend},
    config::ExperimentConfig;
    experiment_dir = project_dir(),
    model_interface = abspath(
        joinpath(experiment_dir, "..", "..", "model_interface.jl"),
    ),
    verbose = false,
    hpc_kwargs = Dict(),
    ekp_kwargs...,
)
    (; ensemble_size, n_iterations, observations, noise, prior, output_dir) =
        config
    return calibrate(
        b,
        ensemble_size,
        n_iterations,
        observations,
        noise,
        prior,
        output_dir;
        model_interface,
        verbose,
        hpc_kwargs,
        ekp_kwargs...,
    )
end

function calibrate(
    b::Type{<:HPCBackend},
    ensemble_size::Int,
    n_iterations::Int,
    observations,
    noise,
    prior,
    output_dir;
    experiment_dir = project_dir(),
    model_interface = abspath(
        joinpath(experiment_dir, "..", "..", "model_interface.jl"),
    ),
    verbose = false,
    hpc_kwargs,
    ekp_kwargs...,
)
    ekp = ekp_constructor(ensemble_size, prior, observations, noise)
    return calibrate(
        b,
        ekp,
        ensemble_size,
        n_iterations,
        prior,
        output_dir;
        experiment_dir,
        model_interface,
        verbose,
        hpc_kwargs,
        ekp_kwargs...,
    )
end

function calibrate(
    b::Type{<:HPCBackend},
    ekp::EKP.EnsembleKalmanProcess,
    ensemble_size,
    n_iterations,
    prior,
    output_dir;
    experiment_dir = project_dir(),
    model_interface = abspath(
        joinpath(experiment_dir, "..", "..", "model_interface.jl"),
    ),
    verbose = false,
    hpc_kwargs,
    ekp_kwargs...,
)
    @info "Initializing calibration" n_iterations ensemble_size output_dir

    initialize(ekp, prior, output_dir)
    module_load_str = module_load_string(b)
    for i in 0:(n_iterations - 1)
        @info "Iteration $i"
        jobids = map(1:ensemble_size) do member
            @info "Running ensemble member $member"
            model_run(
                b,
                i,
                member,
                output_dir,
                experiment_dir,
                model_interface,
                module_load_str;
                hpc_kwargs,
            )
        end

        wait_for_jobs(
            jobids,
            output_dir,
            i,
            experiment_dir,
            model_interface,
            module_load_str;
            hpc_kwargs,
            verbose,
            reruns = 0,
        )
        @info "Completed iteration $i, updating ensemble"
        G_ensemble = observation_map(i)
        save_G_ensemble(output_dir, i, G_ensemble)
        terminate = update_ensemble(output_dir, i, prior)
        !isnothing(terminate) && break
        iter_path = path_to_iteration(output_dir, i + 1)
        ekp = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    end
    return ekp
end

# Dispatch on backend type to unify `calibrate` for all HPCBackends
# Scheduler interfaces should not depend on backend struct
"""
    model_run(backend, iter, member, output_dir, experiment_dir; model_interface, verbose, hpc_kwargs)

Construct and execute a command to run a single forward model on a given job scheduler.

Dispatches on `backend` to run [`slurm_model_run`](@ref) or [`pbs_model_run`](@ref).

Arguments:
- iter: Iteration number
- member: Member number
- output_dir: Calibration experiment output directory
- project_dir: Directory containing the experiment's Project.toml
- model_interface: Model interface file
- module_load_str: Commands which load the necessary modules
- hpc_kwargs: Dictionary containing the resources for the job. Easily generated using [`kwargs`](@ref).
"""
model_run(
    ::Type{<:SlurmBackend},
    iter,
    member,
    output_dir,
    project_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
) = slurm_model_run(
    iter,
    member,
    output_dir,
    project_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
)
model_run(
    ::Type{DerechoBackend},
    iter,
    member,
    output_dir,
    project_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
) = pbs_model_run(
    iter,
    member,
    output_dir,
    project_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
)

using Distributed

import EnsembleKalmanProcesses as EKP
import Dates

export get_backend, calibrate, model_run

export JuliaBackend, WorkerBackend
export HPCBackend,
    ClimaGPUBackend, DerechoBackend, CaltechHPCBackend, GCPBackend

abstract type AbstractBackend end

const DEFAULT_FAILURE_RATE = 0.5

"""
    JuliaBackend

The simplest backend, used to run a calibration in Julia without any parallelization.
"""
struct JuliaBackend <: AbstractBackend end

"""
    HPCBackend <: AbstractBackend

All concrete types of `HPCBackend` share the same keyword arguments for the
constructors.

## Keyword Arguments for HPC backends
- `hpc_kwargs::Dict{Symbol, String}`: Dictionary of arguments passed to the job
  scheduler (e.g., Slurm or PBS). You may find the function [`kwargs`](@ref)
  helpful to construct `hpc_kwargs`.
- `verbose::Bool`: Enable verbose logging output. The default is `false`.
- `experiment_dir::String`: Directory containing the experiment's Project.toml
  file. The default is the current project directory.
- `model_interface::String`: Absolute path to the model interface file that
  defines how to run the forward model. The default is
  `abspath(joinpath(project_dir(), "..", "..", "model_interface.jl"))`.
"""
abstract type HPCBackend <: AbstractBackend end
abstract type SlurmBackend <: HPCBackend end

"""
    CaltechHPCBackend

Used for Caltech's [high-performance computing cluster](https://www.hpc.caltech.edu/).

See [`HPCBackend`](@ref) for the keyword arguments to construct a
`CaltechHPCBackend`.
"""
Base.@kwdef struct CaltechHPCBackend <: SlurmBackend
    experiment_dir::String = project_dir()
    model_interface::String =
        abspath(joinpath(project_dir(), "..", "..", "model_interface.jl"))
    hpc_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
    verbose::Bool = false
end

"""
    ClimaGPUBackend

Used for CliMA's private GPU server.

See [`HPCBackend`](@ref) for the keyword arguments to construct a
`ClimaGPUBackend`.
"""
Base.@kwdef struct ClimaGPUBackend <: SlurmBackend
    experiment_dir::String = project_dir()
    model_interface::String =
        abspath(joinpath(project_dir(), "..", "..", "model_interface.jl"))
    hpc_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
    verbose::Bool = false
end

"""
    GCPBackend

Used for CliMA's private GPU server.

See [`HPCBackend`](@ref) for the keyword arguments to construct a
`GCPBackend`.
"""
Base.@kwdef struct GCPBackend <: SlurmBackend
    experiment_dir::String = project_dir()
    model_interface::String =
        abspath(joinpath(project_dir(), "..", "..", "model_interface.jl"))
    hpc_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
    verbose::Bool = false
end

"""
    DerechoBackend

Used for NSF NCAR's [Derecho supercomputing system](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/).

See [`HPCBackend`](@ref) for the keyword arguments to construct a
`DerechoBackend`.
"""
Base.@kwdef struct DerechoBackend <: HPCBackend
    experiment_dir::String = project_dir()
    model_interface::String =
        abspath(joinpath(project_dir(), "..", "..", "model_interface.jl"))
    hpc_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
    verbose::Bool = false
end

"""
    WorkerBackend

Used to run calibrations on Distributed.jl's workers.
For use on a Slurm cluster, see [`SlurmManager`](@ref).

## Keyword Arguments for `WorkerBackend`
- `failure_rate::Float64 `: The threshold for the percentage of workers that can
  fail before an iteration is stopped. The default is
  $DEFAULT_FAILURE_RATE.
- `worker_pool`: A worker pool created from the workers available.
"""
Base.@kwdef struct WorkerBackend{WORKERPOOL <: WorkerPool} <: AbstractBackend
    failure_rate::Float64 = DEFAULT_FAILURE_RATE
    worker_pool::WORKERPOOL = default_worker_pool()
end

function get_backend_kwargs(b::HPCBackend)
    (; experiment_dir, model_interface, hpc_kwargs, verbose) = b
    return (; experiment_dir, model_interface, hpc_kwargs, verbose)
end

function get_backend_kwargs(::JuliaBackend)
    return (;)
end

function get_backend_kwargs(b::WorkerBackend)
    (; failure_rate, worker_pool) = b
    return (; failure_rate, worker_pool)
end

"""
    get_backend()

Get ideal backend for deploying forward model runs.
Each backend is found via `gethostname()`. Defaults to `JuliaBackend` if none is
found.
"""
function get_backend()
    # TODO: Add WorkerBackend as default if there are multiple workers
    HOSTNAMES = [
        (r"^clima.gps.caltech.edu$", ClimaGPUBackend),
        (r"^login[1-4].cm.cluster$", CaltechHPCBackend),
        (r"^hpc-(\d\d)-(\d\d).cm.cluster$", CaltechHPCBackend),
        (r"^hpc\d+-slurm-login-\d+$", GCPBackend),
        (r"^hpc\d+-a\d+nodeset-\d+$", GCPBackend),
        (r"derecho([1-8])$", DerechoBackend),
        (r"deg(\d\d\d\d)$", DerechoBackend), # This should be more specific
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
function module_load_string(::CaltechHPCBackend)
    return """export MODULEPATH="/resnick/groups/esm/modules:\$MODULEPATH"
    module purge
    module load climacommon/2024_10_09"""
end

function module_load_string(::ClimaGPUBackend)
    return """module purge
    module load julia/1.11.0 cuda/julia-pref openmpi/4.1.5-mpitrampoline"""
end

function module_load_string(::DerechoBackend)
    return """export MODULEPATH="/glade/campaign/univ/ucit0011/ClimaModules-Derecho:\$MODULEPATH" 
    module purge
    module load climacommon
    module list
    """
end

function module_load_string(::GCPBackend)
    return """export OPAL_PREFIX="/sw/openmpi-5.0.5"
    export PATH="/sw/openmpi-5.0.5/bin:\$PATH"
    export LD_LIBRARY_PATH="/sw/openmpi-5.0.5/lib:\$LD_LIBRARY_PATH"
    export UCX_MEMTYPE_CACHE=y  # UCX Memory optimization which toggles whether UCX library intercepts cu*alloc* calls
    """
end

"""
    calibrate(
        ensemble_size::Int,
        n_iterations::Int,
        observations,
        noise,
        prior,
        output_dir;
        backend_kwargs::NamedTuple,
        ekp_kwargs...,
    )

Run a calibration using a backend constructed from `backend_kwargs` and a
`EKP.EnsembleKalmanProcess` constructed from `ekp_kwargs`.

See the backend's documentation for the available keyword arguments.
"""
function calibrate(
    ensemble_size::Int,
    n_iterations::Int,
    observations,
    noise,
    prior,
    output_dir;
    backend_kwargs::NamedTuple,
    ekp_kwargs...,
)
    backend = get_backend()
    return calibrate(
        backend(; backend_kwargs...),
        ensemble_size,
        n_iterations,
        observations,
        noise,
        prior,
        output_dir;
        ekp_kwargs...,
    )
end

function calibrate(
    ::JuliaBackend,
    ekp::EKP.EnsembleKalmanProcess,
    n_iterations,
    prior,
    output_dir,
)
    ekp = initialize(ekp, prior, output_dir)
    ensemble_size = EKP.get_N_ens(ekp)

    on_error(e::InterruptException) = rethrow(e)
    on_error(e) =
        @error "Single ensemble member has errored. See stacktrace" exception =
            (e, catch_backtrace())

    first_iter = last_completed_iteration(output_dir) + 1
    for iter in first_iter:(n_iterations - 1)
        errors = 0
        @info "Running iteration $iter"
        foreach(1:ensemble_size) do m
            try
                forward_model(iter, m)
                @info "Completed member $m"
            catch e
                errors += 1
                on_error(e)
            end
        end
        if errors == ensemble_size
            error("Full ensemble has failed, aborting calibration.")
        end
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update!(ekp, output_dir, iter, prior)
        !isnothing(terminate) && break
    end
    return ekp
end

"""
    calibrate(backend, ekp::EnsembleKalmanProcess, ensemble_size, n_iterations, prior, output_dir)
    calibrate(backend, ensemble_size, n_iterations, observations, noise, prior, output_dir; ekp_kwargs...)

Run a full calibration on the given backend.

If the EKP struct is not given, it will be constructed upon initialization.
While EKP keyword arguments are passed through to the EKP constructor, if using
many keywords it is recommended to construct the EKP object and pass it into
`calibrate`.

Available Backends: [`WorkerBackend`](@ref), [`CaltechHPCBackend`](@ref),
[`ClimaGPUBackend`](@ref), [`DerechoBackend`](@ref), [`JuliaBackend`](@ref).

Derecho, ClimaGPU, and CaltechHPC backends are designed to run on a specific
high-performance computing cluster. WorkerBackend uses Distributed.jl to run the
forward model on workers.
"""
function calibrate(
    b::AbstractBackend,
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    output_dir;
    ekp_kwargs...,
)
    ekp = ekp_constructor(
        ensemble_size,
        prior,
        observations,
        noise;
        ekp_kwargs...,
    )

    # Dispatch on backend
    return calibrate(b, ekp, n_iterations, prior, output_dir)
end

function calibrate(
    b::WorkerBackend,
    ekp::EKP.EnsembleKalmanProcess,
    n_iterations,
    prior,
    output_dir,
)
    (; failure_rate, worker_pool) = get_backend_kwargs(b)
    ekp = initialize(ekp, prior, output_dir)
    ensemble_size = EKP.get_N_ens(ekp)

    first_iter = last_completed_iteration(output_dir) + 1
    for iter in first_iter:(n_iterations - 1)
        @info "Running Iteration $iter"
        (; time) = @timed run_worker_iteration(
            iter,
            ensemble_size,
            output_dir;
            worker_pool,
            failure_rate,
        )
        formatted_time =
            Dates.canonicalize(Dates.Millisecond(round(time * 1000)))
        if isempty(Dates.periods(formatted_time))
            @info "Iteration $iter time: 0 second"
        else
            @info "Iteration $iter time: $formatted_time"
        end
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update!(ekp, output_dir, iter, prior)
        !isnothing(terminate) && break
    end
    return ekp
end

function calibrate(
    b::HPCBackend,
    ekp::EKP.EnsembleKalmanProcess,
    n_iterations,
    prior,
    output_dir;
    exeflags = "",
)
    ensemble_size = EKP.get_N_ens(ekp)
    @info "Initializing calibration" n_iterations ensemble_size output_dir
    ekp = initialize(ekp, prior, output_dir)
    module_load_str = module_load_string(b)
    first_iter = last_completed_iteration(output_dir) + 1

    for iter in first_iter:(n_iterations - 1)
        run_hpc_iteration(
            b,
            iter,
            ensemble_size,
            output_dir,
            module_load_str;
            exeflags,
        )
        @info "Completed iteration $iter, updating ensemble"
        ekp = load_ekp_struct(output_dir, iter)
        terminate = observation_map_and_update!(ekp, output_dir, iter, prior)
        !isnothing(terminate) && break
    end

    return ekp
end

function run_hpc_iteration(
    b::HPCBackend,
    iter,
    ensemble_size,
    output_dir,
    module_load_str;
    exeflags = "",
)
    (; hpc_kwargs, verbose, experiment_dir, model_interface) =
        get_backend_kwargs(b)
    @info "Iteration $iter"
    job_ids = map(1:ensemble_size) do member
        model_run(
            b,
            iter,
            member,
            output_dir,
            experiment_dir,
            module_load_str;
            exeflags,
        )
    end
    job_ids = filter(!isnothing, job_ids)
    if !isempty(job_ids)
        job_id_type = typeof(first(job_ids))
        job_ids = job_id_type[id for id in job_ids]
        wait_for_jobs(
            job_ids,
            output_dir,
            iter,
            experiment_dir,
            model_interface,
            module_load_str;
            hpc_kwargs,
            verbose,
            reruns = 0,
        )
    end
end

# Dispatch on backend type to unify `calibrate` for all HPCBackends
# This keeps the scheduler interface independent from the backends
"""
    model_run(backend, iter, member, output_dir, project_dir, module_load_str; exeflags)

Construct and execute a command to run a single forward model on a given job scheduler.

Uses the given `backend` to run [`slurm_model_run`](@ref) or [`pbs_model_run`](@ref).

Arguments:
- `iter`: Iteration number
- `member`: Member number
- `output_dir`: Calibration experiment output directory
- `project_dir`: Directory containing the experiment's Project.toml
- `module_load_str`: Commands which load the necessary modules
"""
function model_run(
    backend::HPCBackend,
    iter,
    member,
    output_dir,
    project_dir,
    module_load_str;
    exeflags = "",
)
    if model_completed(output_dir, iter, member)
        @info "Skipping completed member $member (found checkpoint)"
        return
    elseif model_started(output_dir, iter, member)
        @info "Resuming member $member (incomplete run detected)"
    else
        @info "Running member $member"
    end

    write_model_started(output_dir, iter, member)

    (; hpc_kwargs, model_interface) = get_backend_kwargs(backend)

    job_id = if backend isa SlurmBackend
        slurm_model_run(
            iter,
            member,
            output_dir,
            project_dir,
            model_interface,
            module_load_str;
            hpc_kwargs,
            exeflags,
        )
    elseif backend isa DerechoBackend
        pbs_model_run(
            iter,
            member,
            output_dir,
            project_dir,
            model_interface,
            module_load_str;
            hpc_kwargs,
            exeflags,
        )
    else
        error("Unsupported backend type: $backend")
    end
    return job_id
end

backend_worker_kwargs(::Type{DerechoBackend}) = (; q = "main", A = "UCIT0011")

backend_worker_kwargs(::Type{GCPBackend}) = (; partition = "a3")

backend_worker_kwargs(::Type{<:AbstractBackend}) = (;)

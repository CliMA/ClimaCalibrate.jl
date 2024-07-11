export get_backend, calibrate, model_run

abstract type AbstractBackend end

struct JuliaBackend <: AbstractBackend end

abstract type HPCBackend <: AbstractBackend end
abstract type SlurmBackend <: HPCBackend end

struct CaltechHPCBackend <: SlurmBackend end
struct ClimaGPUBackend <: SlurmBackend end

struct DerechoBackend <: HPCBackend end

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
    module load climacommon/2024_05_27"""
end

function module_load_string(::Type{ClimaGPUBackend})
    return """module purge
    module load julia/1.10.0 cuda/julia-pref openmpi/4.1.5-mpitrampoline"""
end

function module_load_string(::Type{DerechoBackend})
    return """export MODULEPATH="/glade/campaign/univ/ucit0011/ClimaModules-Derecho:\$MODULEPATH" 
    module purge
    module load climacommon
    module list
    """
end

calibrate(config::ExperimentConfig; ekp_kwargs...) =
    calibrate(get_backend(), config; ekp_kwargs...)

calibrate(experiment_dir::AbstractString; ekp_kwargs...) =
    calibrate(get_backend(), ExperimentConfig(experiment_dir); ekp_kwargs...)

calibrate(
    b::Type{JuliaBackend},
    experiment_dir::AbstractString;
    ekp_kwargs...,
) = calibrate(b, ExperimentConfig(experiment_dir); ekp_kwargs...)

function calibrate(
    ::Type{JuliaBackend},
    config::ExperimentConfig;
    ekp_kwargs...,
)
    (; n_iterations, ensemble_size) = config
    eki = initialize(config; ekp_kwargs...)
    for i in 0:(n_iterations - 1)
        @info "Running iteration $i"
        for m in 1:ensemble_size
            run_forward_model(set_up_forward_model(m, i, config))
            @info "Completed member $m"
        end
        G_ensemble = observation_map(i)
        save_G_ensemble(config, i, G_ensemble)
        eki = update_ensemble(config, i)
    end
    return eki
end

"""
    calibrate(::Type{AbstractBackend}, config::ExperimentConfig; kwargs...)
    calibrate(::Type{AbstractBackend}, experiment_dir; kwargs...)

Run a full calibration, scheduling the forward model runs on Caltech's HPC cluster.

Takes either an ExperimentConfig or an experiment folder.

Available Backends: CaltechHPCBackend, ClimaGPUBackend, DerechoBackend, JuliaBackend


# Keyword Arguments
- `experiment_dir: Directory containing experiment configurations.
- `model_interface: Path to the model interface file.
- `hpc_kwargs`: Dictionary of resource arguments, passed to the job scheduler.
- `verbose::Bool`: Enable verbose logging.

# Usage
Open julia: `julia --project=experiments/surface_fluxes_perfect_model`
```julia
using ClimaCalibrate

experiment_dir = joinpath(pkgdir(ClimaCalibrate), "experiments", "surface_fluxes_perfect_model")
model_interface = joinpath(experiment_dir, "model_interface.jl")

# Generate observational data and load interface
include(joinpath(experiment_dir, "generate_data.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))
include(model_interface)

hpc_kwargs = kwargs(time = 3)
backend = get_backend()
eki = calibrate(backend, experiment_dir; model_interface, hpc_kwargs);
```
"""
function calibrate(
    b::Type{<:HPCBackend},
    experiment_dir::AbstractString;
    hpc_kwargs,
    ekp_kwargs...,
)
    calibrate(b, ExperimentConfig(experiment_dir); hpc_kwargs, ekp_kwargs...)
end

function calibrate(
    b::Type{<:HPCBackend},
    config::ExperimentConfig;
    experiment_dir = dirname(Base.active_project()),
    model_interface = abspath(
        joinpath(experiment_dir, "..", "..", "model_interface.jl"),
    ),
    verbose = false,
    reruns = 1,
    hpc_kwargs,
    ekp_kwargs...,
)
    # ExperimentConfig is created from a YAML file within the experiment_dir
    (; n_iterations, output_dir, ensemble_size) = config
    @info "Initializing calibration" n_iterations ensemble_size output_dir

    eki = initialize(config; ekp_kwargs...)
    module_load_str = module_load_string(b)
    for iter in 0:(n_iterations - 1)
        @info "Iteration $iter"
        jobids = map(1:ensemble_size) do member
            @info "Running ensemble member $member"
            model_run(
                b,
                iter,
                member,
                output_dir,
                experiment_dir,
                model_interface,
                module_load_str;
                hpc_kwargs,
            )
        end

        statuses = wait_for_jobs(
            jobids,
            output_dir,
            iter,
            experiment_dir,
            model_interface,
            module_load_str;
            hpc_kwargs,
            verbose,
            reruns,
        )
        @info "Completed iteration $iter, updating ensemble"
        G_ensemble = observation_map(iter)
        save_G_ensemble(config, iter, G_ensemble)
        eki = update_ensemble(config, iter)
    end
    return eki
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
- experiment_dir: Directory containing the experiment's Project.toml
- model_interface: File containing the model interface
- module_load_str: Commands which load the necessary modules
- hpc_kwargs: Dictionary containing the resources for the job. Easily generated using [`kwargs`](@ref).
"""
model_run(
    b::Type{<:SlurmBackend},
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
) = slurm_model_run(
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
)
model_run(
    b::Type{DerechoBackend},
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
) = pbs_model_run(
    iter,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
)

abstract type AbstractBackend end

struct JuliaBackend <: AbstractBackend end
abstract type SlurmBackend <: AbstractBackend end
struct CaltechHPCBackend <: SlurmBackend end
struct ClimaGPUBackend <: SlurmBackend end

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
    ]

    for (pattern, backend) in HOSTNAMES
        !isnothing(match(pattern, gethostname())) && return backend
    end

    return JuliaBackend
end

"""
    module_load_string(T) where {T<:Type{SlurmBackend}}

Return a string that loads the correct modules for a given backend when executed via bash.
"""
function module_load_string(::Type{CaltechHPCBackend})
    return """export MODULEPATH=/groups/esm/modules:\$MODULEPATH
    module purge
    module load climacommon/2024_05_27"""
end

function module_load_string(::Type{ClimaGPUBackend})
    return """module purge
    module load julia/1.10.0 cuda/julia-pref openmpi/4.1.5-mpitrampoline"""
end

"""
    calibrate(::Type{JuliaBackend}, config::ExperimentConfig)
    calibrate(::Type{JuliaBackend}, experiment_dir::AbstractString)

Run a calibration in Julia.

Takes an ExperimentConfig or an experiment folder.
If no backend is passed, one is chosen via `get_backend`. 
This function is intended for use in a larger workflow, assuming that all needed 
model interface and observation map functions are set up for the calibration.

# Example
Run: `julia --project=experiments/surface_fluxes_perfect_model`
```julia
import ClimaCalibrate

# Generate observational data and load interface
experiment_dir = dirname(Base.active_project())
include(joinpath(experiment_dir, "generate_data.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))
include(joinpath(experiment_dir, "model_interface.jl"))

# Initialize and run the calibration
eki = ClimaCalibrate.calibrate(experiment_dir)
```
"""
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
    initialize(config; ekp_kwargs...)
    (; n_iterations, ensemble_size) = config
    eki = nothing
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
    calibrate(::Type{SlurmBackend}, config::ExperimentConfig; kwargs...)
    calibrate(::Type{SlurmBackend}, experiment_dir; kwargs...)

Run a full calibration, scheduling the forward model runs on Caltech's HPC cluster.

Takes either an ExperimentConfig or an experiment folder.

# Keyword Arguments
- `experiment_dir: Directory containing experiment configurations.
- `model_interface: Path to the model interface file.
- `slurm_kwargs`: Dictionary of slurm arguments, passed through to `sbatch`.
- `verbose::Bool`: Enable verbose output for debugging.

# Usage
Open julia: `julia --project=experiments/surface_fluxes_perfect_model`
```julia
import ClimaCalibrate: CaltechHPCBackend, calibrate

experiment_dir = dirname(Base.active_project())
model_interface = joinpath(experiment_dir, "model_interface.jl")

# Generate observational data and load interface
include(joinpath(experiment_dir, "generate_data.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))
include(model_interface)

slurm_kwargs = kwargs(time = 3)
eki = calibrate(CaltechHPCBackend, experiment_dir; model_interface, slurm_kwargs);
```
"""
function calibrate(
    b::Type{<:SlurmBackend},
    experiment_dir::AbstractString;
    slurm_kwargs,
    ekp_kwargs...,
)
    calibrate(b, ExperimentConfig(experiment_dir); slurm_kwargs, ekp_kwargs...)
end

function calibrate(
    b::Type{<:SlurmBackend},
    config::ExperimentConfig;
    experiment_dir = dirname(Base.active_project()),
    model_interface = abspath(
        joinpath(experiment_dir, "..", "..", "model_interface.jl"),
    ),
    verbose = false,
    slurm_kwargs = Dict(:time_limit => 45, :ntasks => 1),
    ekp_kwargs...,
)
    # ExperimentConfig is created from a YAML file within the experiment_dir
    (; n_iterations, output_dir, ensemble_size) = config
    @info "Initializing calibration" n_iterations ensemble_size output_dir
    initialize(config; ekp_kwargs...)

    eki = nothing
    module_load_str = module_load_string(b)
    for iter in 0:(n_iterations - 1)
        @info "Iteration $iter"
        jobids = map(1:ensemble_size) do member
            @info "Running ensemble member $member"
            sbatch_model_run(
                iter,
                member,
                output_dir,
                experiment_dir,
                model_interface,
                module_load_str;
                slurm_kwargs,
            )
        end

        statuses = wait_for_jobs(
            jobids,
            output_dir,
            iter,
            experiment_dir,
            model_interface,
            module_load_str;
            slurm_kwargs,
            verbose,
        )
        report_iteration_status(statuses, output_dir, iter)
        @info "Completed iteration $iter, updating ensemble"
        G_ensemble = observation_map(iter)
        save_G_ensemble(config, iter, G_ensemble)
        eki = update_ensemble(config, iter)
    end
    return eki
end

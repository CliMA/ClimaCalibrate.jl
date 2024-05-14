abstract type AbstractBackend end

struct JuliaBackend <: AbstractBackend end
struct CaltechHPC <: AbstractBackend end

function get_backend()
    backend = JuliaBackend
    if isfile("/etc/redhat-release") &&
       occursin("Red Hat", read("/etc/redhat-release", String))
        backend = CaltechHPC
    end
    return backend
end

"""
    calibrate(config::ExperimentConfig)
    calibrate(experiment_dir::AbstractString)

Run a calibration in Julia. Takes an ExperimentConfig or an experiment folder.

This function is intended for use in a larger workflow, assuming that all related 
model interfaces and data generation scripts are properly aligned with the configuration.

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
calibrate(config::ExperimentConfig; kwargs...) =
    calibrate(get_backend(), config; kwargs...)

calibrate(experiment_dir::AbstractString) =
    calibrate(get_backend(), ExperimentConfig(experiment_dir))

calibrate(b::Type{JuliaBackend}, experiment_dir::AbstractString) =
    calibrate(b, ExperimentConfig(experiment_dir))

function calibrate(::Type{JuliaBackend}, config::ExperimentConfig)
    initialize(config)
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
    calibrate(::Type{CaltechHPC}, config::ExperimentConfig; kwargs...)
    calibrate(::Type{CaltechHPC}, experiment_dir; kwargs...)

Runs a full calibration, scheduling the forward model runs on Caltech's HPC cluster.

Takes either an ExperimentConfig for an experiment folder.

# Keyword Arguments
- `experiment_dir::AbstractString`: Directory containing experiment configurations.
- `model_interface::AbstractString`: Path to the model interface file.
- `time_limit::AbstractString`: Time limit for Slurm jobs.
- `ntasks::Int`: Number of tasks to run in parallel.
- `cpus_per_task::Int`: Number of CPUs per Slurm task.
- `gpus_per_task::Int`: Number of GPUs per Slurm task.
- `partition::AbstractString`: Slurm partition to use.
- `verbose::Bool`: Enable verbose output for debugging.

# Usage
Open julia: `julia --project=experiments/surface_fluxes_perfect_model`
```julia
import ClimaCalibrate: CaltechHPC, calibrate

experiment_dir = dirname(Base.active_project())
model_interface = joinpath(experiment_dir, "model_interface.jl")

# Generate observational data and load interface
include(joinpath(experiment_dir, "generate_data.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))
include(model_interface)

slurm_kwargs = kwargs(time = 3)
eki = calibrate(CaltechHPC, experiment_dir; model_interface, slurm_kwargs);
```
"""
function calibrate(
    b::Type{CaltechHPC},
    experiment_dir::AbstractString;
    kwargs...,
)
    calibrate(b, ExperimentConfig(experiment_dir); kwargs...)
end

function calibrate(
    ::Type{CaltechHPC},
    config::ExperimentConfig;
    experiment_dir = dirname(Base.active_project()),
    model_interface = abspath(
        joinpath(experiment_dir, "..", "..", "model_interface.jl"),
    ),
    verbose = false,
    slurm_kwargs = Dict(:time_limit => 45),
)
    # ExperimentConfig is created from a YAML file within the experiment_dir
    (; n_iterations, output_dir, ensemble_size) = config
    @info "Initializing calibration" n_iterations ensemble_size output_dir
    initialize(config)

    eki = nothing
    for iter in 0:(n_iterations - 1)
        @info "Iteration $iter"
        jobids = map(1:ensemble_size) do member
            @info "Running ensemble member $member"
            sbatch_model_run(
                iter,
                member,
                output_dir,
                experiment_dir,
                model_interface;
                slurm_kwargs,
            )
        end

        statuses = wait_for_jobs(
            jobids,
            output_dir,
            iter,
            experiment_dir,
            model_interface;
            verbose,
            slurm_kwargs,
        )
        report_iteration_status(statuses, output_dir, iter)
        @info "Completed iteration $iter, updating ensemble"
        G_ensemble = observation_map(iter)
        save_G_ensemble(config, iter, G_ensemble)
        eki = update_ensemble(config, iter)
    end
    return eki
end

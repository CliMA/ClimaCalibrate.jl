
abstract type AbstractBackend end

struct JuliaBackend <: AbstractBackend end

"""
    calibrate(::AbstractBackend, configuration::ExperimentConfig; kwargs...)
    calibrate(::AbstractBackend, experiment_dir::AbstractString; kwargs...)
    calibrate(configuration::ExperimentConfig)

Conducts a full calibration experiment using the Ensemble Kalman Process (EKP). 
This function initializes the calibration, runs the forward model across all 
ensemble members for each iteration, and updates the ensemble based on observations.

# Arguments
- `backend::AbstractBackend`: Backend to run the calibration on. If not provided, default is `JuliaBackend`
- `configuration::ExperimentConfig`: Configuration object containing all necessary settings for the calibration experiment

# Usage
This function is intended to be used in a larger workflow where the 
`ExperimentConfig` is set up with the necessary experiment parameters. 
It assumes that all related model interfaces and data generation scripts
are properly aligned with the configuration.

# Example
Run: `julia --project=experiments/surface_fluxes_perfect_model`
```julia
import CalibrateAtmos

# Generate observational data and load interface
experiment_path = dirname(Base.active_project())
include(joinpath(experiment_path, "generate_data.jl"))
include(joinpath(experiment_path, "observation_map.jl"))
include(joinpath(experiment_path, "model_interface.jl"))

# Initialize and run the calibration
eki = CalibrateAtmos.calibrate(experiment_path)
"""
calibrate(config::ExperimentConfig) = calibrate(JuliaBackend(), config)

calibrate(experiment_path::AbstractString) =
    calibrate(JuliaBackend(), ExperimentConfig(experiment_path))

function calibrate(::JuliaBackend, configuration::ExperimentConfig)
    initialize(configuration)

    (; n_iterations, id, ensemble_size) = configuration

    eki = nothing
    physical_model = get_forward_model(Val(Symbol(id)))
    for i in 0:(n_iterations - 1)
        @info "Running iteration $i"
        for m in 1:ensemble_size
            run_forward_model(
                physical_model,
                get_config(physical_model, m, i, configuration),
            )
            @info "Completed member $m"
        end

        G_ensemble = observation_map(Val(Symbol(id)), i)
        save_G_ensemble(configuration, i, G_ensemble)
        eki = update_ensemble(configuration, i)
    end
    return eki
end

struct CaltechHPC <: AbstractBackend end

"""
    calibrate(::CaltechHPC; experiment_dir; kwargs...)
    calibrate(::CaltechHPC; config::ExperimentConfig; kwargs...)

Runs a full calibration using the Ensemble Kalman Process (EKP), scheduling the forward model runs on a Slurm cluster.

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
import CalibrateAtmos: CaltechHPC, calibrate

experiment_dir = dirname(Base.active_project())
model_interface = joinpath(experiment_dir, "model_interface.jl")

# Generate observational data and load interface
include(joinpath(experiment_dir, "generate_data.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))
include(model_interface)

eki = calibrate(CaltechHPC(), experiment_dir; time_limit = "3", model_interface);
```
"""
function calibrate(b::CaltechHPC, experiment_dir::AbstractString; kwargs...)
    calibrate(b, ExperimentConfig(experiment_dir); kwargs...)
end

function calibrate(
    ::CaltechHPC,
    config::ExperimentConfig;
    experiment_dir = dirname(Base.active_project()),
    model_interface = abspath(
        joinpath(experiment_dir, "..", "..", "model_interface.jl"),
    ),
    time_limit = "1:00:00",
    ntasks = 1,
    cpus_per_task = 1,
    gpus_per_task = 0,
    partition = gpus_per_task > 0 ? "gpu" : "expansion",
    verbose = false,
)
    # ExperimentConfig is created from a YAML file within the experiment_dir
    (; n_iterations, output_dir, ensemble_size) = config

    @info "Initializing calibration" n_iterations ensemble_size output_dir
    initialize(config)

    eki = nothing
    for iter in 0:(n_iterations - 1)
        @info "Iteration $iter"
        jobids = map(1:ensemble_size) do j
            srun_model(;
                output_dir,
                member = j,
                iter,
                time_limit,
                ntasks,
                partition,
                cpus_per_task,
                gpus_per_task,
                experiment_dir,
                model_interface,
                verbose,
            )
        end
        statuses = wait_for_jobs(jobids, output_dir, iter, verbose)
        report_ensemble_exit_status(statuses, output_dir, iter)
        @info "Completed iteration $iter, updating ensemble"
        G_ensemble = observation_map(Val(Symbol(config.id)), iter)
        save_G_ensemble(config, iter, G_ensemble)
        eki = update_ensemble(config, iter)
    end
    return eki
end

"""
    log_member_error(output_dir, iteration, member, verbose = false)

Log a warning message when an error occurs in a specific ensemble member during a model run in a Slurm environment. 
If verbose, includes the ensemble member's output.
"""
function log_member_error(output_dir, iteration, member, verbose = false)
    member_log = joinpath(
        path_to_ensemble_member(output_dir, iteration, member),
        "model_log.txt",
    )
    warn_str = "Ensemble member $member raised an error. See model log at $member_log for stacktrace"
    if verbose
        stacktrace = replace(readchomp(member_log), "\\n" => "\n")
        warn_str = warn_str * ": \n$stacktrace"
    end
    @warn warn_str
end

"""
    srun_model(;
        output_dir,
        iter,
        member,
        time_limit,
        ntasks,
        partition,
        cpus_per_task,
        gpus_per_task,
        experiment_dir,
        model_interface,
        verbose,
    )

Construct and execute a command to run a model simulation on a Slurm cluster for a single ensemble member.
"""
function srun_model(;
    output_dir,
    iter,
    member,
    time_limit,
    ntasks,
    partition,
    cpus_per_task,
    gpus_per_task,
    experiment_dir,
    model_interface,
    verbose,
)
    member_log = joinpath(
        path_to_ensemble_member(output_dir, iter, member),
        "model_log.txt",
    )

    sbatch_contents = """
#!/bin/bash
#SBATCH --job-name=run_$(iter)_$(member)
#SBATCH --time=$time_limit
#SBATCH --ntasks=$ntasks
#SBATCH --partition=$partition
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --gpus-per-task=$gpus_per_task
#SBATCH --output=$member_log

export MODULEPATH=/groups/esm/modules:\$MODULEPATH
module purge
module load climacommon/2024_04_05

srun --output=$member_log --open-mode=append julia --project=$experiment_dir -e '
import CalibrateAtmos as CAL
iteration = $iter; member = $member
model_interface = "$model_interface"; include(model_interface)

experiment_dir = "$experiment_dir"
experiment_config = CAL.ExperimentConfig(experiment_dir)
experiment_id = experiment_config.id
physical_model = CAL.get_forward_model(Val(Symbol(experiment_id)))
CAL.run_forward_model(physical_model, CAL.get_config(physical_model, member, iteration, experiment_dir))
@info "Forward Model Run Completed" experiment_id physical_model iteration member'
"""

    sbatch_filepath, io = mktemp(output_dir)
    write(io, sbatch_contents)
    close(io)

    @info "Running ensemble member $member"
    return submit_job(sbatch_filepath)
end

function wait_for_jobs(jobids, output_dir, iter, verbose)
    try
        all_done = false
        statuses = map(job_status, jobids)
        failed_members = []
        completed_members = []
        while !all_done
            statuses = map(job_status, jobids)

            for (m, status) in enumerate(statuses)
                if status == "FAILED" && !(m in failed_members)
                    log_member_error(output_dir, iter, m, verbose)
                    push!(failed_members, m)
                elseif status == "COMPLETED" && !(m in completed_members)
                    @info "Ensemble member $m complete"
                    push!(completed_members, m)
                end
            end
            all_done = all(job_completed, statuses)
            sleep(5)
        end
        return statuses
    catch e
        kill_all_jobs(jobids)
        e isa InterruptException || rethrow()
        return map(job_status, jobids)
    end
end

function report_ensemble_exit_status(statuses, output_dir, iter)
    all(job_completed.(statuses)) || error("Some jobs are not complete")
    if all(job_failed, statuses)
        error(
            "Full ensemble for iteration $iter has failed. See model logs in $(path_to_iteration(output_dir, iter)) for details.",
        )
    elseif any(job_failed, statuses)
        @warn "Failed ensemble members: $(findall(job_failed, statuses))"
    end
end

function submit_job(sbatch_filepath; debug = false)
    jobid = readchomp(`sbatch --parsable $sbatch_filepath`)
    debug || rm(sbatch_filepath)
    return string(jobid)
end

job_running(status) = status == "RUNNING"
job_success(status) = status == "COMPLETED"
job_failed(status) = status == "FAILED"
job_completed(status) = job_failed(status) || job_success(status)

"""
    job_status(jobid)

Parse the slurm jobid's state and return one of three status strings: "COMPLETED", "FAILED", or "RUNNING"
"""
function job_status(jobid)
    failure_statuses = ("FAILED", "CANCELLED+", "CANCELLED")
    output = readchomp(`sacct -j $jobid --format=State --noheader`)
    statuses = strip.(split(output, "\n"))
    if all(s -> s == "COMPLETED", statuses)
        return "COMPLETED"
    elseif any(s -> s in failure_statuses, statuses)
        return "FAILED"
    else
        return "RUNNING"
    end
end

"""
    kill_all_jobs(jobids)

Takes a list of slurm job IDs and runs `scancel` on them.
"""
function kill_all_jobs(jobids)
    for jobid in jobids
        try
            kill_slurm_job(jobid)
            println("Killed slurm job $jobid")
        catch e
            println("Failed to kill slurm job $jobid: ", e)
        end
    end
end

kill_slurm_job(jobid) = run(`scancel $jobid`)

function format_slurm_time(minutes::Int)
    # Calculate the number of days, hours, and minutes
    days = minutes / (60 * 24)
    hours = (minutes / 60) % 24
    remaining_minutes = minutes % 60

    # Format the string according to Slurm's time format
    if days > 0
        return string(
            days,
            "-",
            lpad(hours, 2, '0'),
            ":",
            lpad(remaining_minutes, 2, '0'),
            ":00",
        )
    else
        return string(
            lpad(hours, 2, '0'),
            ":",
            lpad(remaining_minutes, 2, '0'),
            ":00",
        )
    end
end

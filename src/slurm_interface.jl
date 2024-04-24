
"""
    slurm_calibration(; kwargs...)

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
This function is designed for use with a Slurm-managed cluster, ensuring that computational tasks are distributed efficiently.
"""
function slurm_calibration(;
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
    config = ExperimentConfig(experiment_dir)
    (; n_iterations, output_dir, ensemble_size, prior) = config

    @info "Initializing calibration" n_iterations ensemble_size output_dir
    initialize(config)

    eki = nothing
    for iter in 0:(n_iterations - 1)
        @info "Iteration $iter"
        procs = asyncmap(1:ensemble_size; ntasks = ensemble_size) do j
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

        handle_ensemble_procs(procs, iter, output_dir, verbose)
        @info "Completed iteration $iter, updating ensemble"
        G_ensemble = observation_map(Val(Symbol(config.id)), iter)
        save_G_ensemble(config, iter, G_ensemble)
        eki = update_ensemble(config, iter)
    end
    return eki
end

"""
    handle_ensemble_procs(procs, iteration, output_dir, verbose)

Helper function for `slurm_calibration`.
Handles the ensemble of processes running the forward model via Slurm.
"""
function handle_ensemble_procs(procs, iteration, output_dir, verbose)
    # Initial `try` handles InterruptException
    try
        asyncmap(enumerate(procs)) do (member, p)
            try
                wait(p)
                if p.exitcode != 0
                    log_member_error(output_dir, iteration, member, verbose)
                end
            catch e
                log_member_error(output_dir, iteration, member, verbose)
            end
        end
    catch e
        foreach(kill, procs)
        e isa InterruptException || rethrow()
    end
    # Wait for member error logs to be printed
    sleep(0.5)
    exit_codes = map(x -> getproperty(x, :exitcode), procs)
    if !any(x -> x == 0, exit_codes)
        error(
            "Full ensemble for iteration $iteration has failed. See model logs in $(path_to_iteration(output_dir, iteration)) for details.",
        )
    elseif !all(x -> x == 0, exit_codes)
        @warn "Failed ensemble members: $(findall(x -> x == 0, exit_codes))"
    end
end

"""
    log_member_error(output_dir, iteration, member, verbose = false)

Logs a warning message when an error occurs in a specific ensemble member during a model run in a Slurm environment. 
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

Constructs and executes a command to run a model simulation on a Slurm cluster for a single ensemble member.
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
    fwd_model_cmd = `
        srun 
        --job-name=run_$(iter)_$(member) \
        --time=$time_limit \
        --ntasks=$ntasks \
        --partition=$partition \
        --cpus-per-task=$cpus_per_task \
        --gpus-per-task=$gpus_per_task \
        julia --project=$experiment_dir -e "
    import CalibrateAtmos as CAL
    iteration = $iter; member = $member
    model_interface = \"$model_interface\"; include(model_interface)

    experiment_dir = \"$experiment_dir\"
    experiment_config = CAL.ExperimentConfig(experiment_dir)
    experiment_id = experiment_config.id
    physical_model = CAL.get_forward_model(Val(Symbol(experiment_id)))
    CAL.run_forward_model(physical_model, CAL.get_config(physical_model, member, iteration, experiment_dir))
    @info \"Forward Model Run Completed\" experiment_id physical_model iteration member"
    `
    @info "Running ensemble member $member"
    return open(
        pipeline(
            detach(fwd_model_cmd);
            stdout = member_log,
            stderr = member_log,
        ),
    )
end

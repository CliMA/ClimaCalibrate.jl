
"""
    slurm_calibration(;
        experiment_dir = dirname(Base.active_project()),
        model_interface = joinpath(experiment_dir, "..", "..", "model_interface.jl"),
        time_limit = "1:00:00",
        ntasks = 1,
        cpus_per_task = 1,
        gpus_per_task = 0,
        verbose = false,
    )

Runs a full calibration, scheduling forward model runs on the slurm cluster using `srun_model`.
This function makes heavy assumptions and requires some setup.
 - The correct project must be selected, and the observation map has already been `include`d
 - The session is not running in an existing slurm job, and is running on the Resnick central cluster.

Input arguments:
 - `experiment_dir`: The directory storing relevant experiment information. (Default: dirname(Base.active_project()))
 - `model_interface`: Model interface file to be included during the model run. (Default: joinpath(experiment_dir, "..", "..", "model_interface.jl"))
 - `time_limit`: Slurm time limit
 - `ntasks`: Slurm ntasks
 - `cpus_per_task`: Slurm CPUs per task
 - `gpus_per_task`: Slurm GPUs per task
 - `partition`: Slurm partition. (Default: gpus_per_task > 0 ? "gpu" : "expansion")
 - `verbose`: Turn on verbose model logging
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

    # Experiment config is created from a YAML file within the experiment_dir
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
    Handles the ensemble of processes running the forward model via slurm.
"""
function handle_ensemble_procs(procs, iteration, output_dir, verbose)
    # Initial try handles InterruptException
    try
        asyncmap(enumerate(procs)) do (member, p)
            member_log = joinpath(
                path_to_ensemble_member(output_dir, iteration, member),
                "model_log.txt",
            )
            try
                wait(p)
                if p.exitcode != 0
                    warn_on_member_error(member, member_log, verbose)
                end
            catch e
                warn_on_member_error(member, member_log, verbose)
            end
        end
    catch e
        foreach(kill, procs)
        e isa InterruptException || rethrow()
    end
    # Wait for processes to be killed
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

function warn_on_member_error(member, member_log, verbose = false)
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

Runs a single forward model ensemble member. Constructs the `srun` command, then 
runs it in a separate process.
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

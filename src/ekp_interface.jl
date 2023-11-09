import TOML, YAML
import JLD2
import Random
using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface

"""
    path_to_iteration(output_dir, iteration)
Returns the path to the iteration folder within `output_dir` for the given iteration number.
"""
path_to_iteration(output_dir, iteration) =
    joinpath(output_dir, join(["iteration", lpad(iteration, 3, "0")], "_"))

"""
    initialize(
        experiment_id;
        config = YAML.load_file("experiments/\$experiment_id/ekp_config.yml"),
        Γ = JLD2.load(config["truth_noise"]),
        y = JLD2.load(config["truth_data"]),
        rng_seed = 1234,
    )
Initializes the EKP object and the model ensemble.

Takes in
 - `experiment_id`: the name of the experiment, which corresponds to the name of the subfolder in `experiments/`
 - `config`: a dictionary of configuration values
"""
function initialize(
    experiment_id;
    config = YAML.load_file("experiments/$experiment_id/ekp_config.yml"),
    Γ = JLD2.load_object(config["truth_noise"]),
    y = JLD2.load_object(config["truth_data"]),
    rng_seed = 1234,
)
    Random.seed!(rng_seed)
    rng_ekp = Random.MersenneTwister(rng_seed)

    output_dir = config["output_dir"]
    prior_path = config["prior_path"]
    param_names = config["parameter_names"]
    ensemble_size = config["ensemble_size"]
    # Save in EKI object in iteration_000 folder
    eki_path = joinpath(output_dir, "iteration_000", "eki_file.jld2")

    param_dict = TOML.parsefile(prior_path)
    prior_vec = [get_parameter_distribution(param_dict, n) for n in param_names]
    prior = combine_distributions(prior_vec)

    initial_ensemble =
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size)
    eki = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        y,
        Γ,
        EKP.Inversion();
        rng = rng_ekp,
    )

    save_parameter_ensemble(
        EKP.get_u_final(eki), # constraints applied when saving
        prior,
        param_dict,
        output_dir,
        "parameters.toml",
        0,  # Initial iteration = 0
    )
    JLD2.save_object(eki_path, eki)
    return eki
end

"""
    update_ensemble(
        experiment_id,
        iteration;
        config = YAML.load_file("experiments/\$experiment_id/ekp_config.yml"),
    )
Updates the EKI object and saves parameters for the next iteration.
Assumes that the observation map has been run and saved in the current iteration folder.
"""
function update_ensemble(
    experiment_id,
    iteration;
    config = YAML.load_file("experiments/$experiment_id/ekp_config.yml"),
)
    output_dir = config["output_dir"]
    names = config["parameter_names"]
    # Load EKI object from iteration folder
    iter_path = path_to_iteration(output_dir, iteration)
    eki_path = joinpath(iter_path, "eki_file.jld2")
    eki = JLD2.load_object(eki_path)

    # Load data from the ensemble
    G_ens = JLD2.load_object(joinpath(iter_path, "observation_map.jld2"))

    # Update
    EKP.update_ensemble!(eki, G_ens)
    iteration += 1

    # Update and save parameters for next iteration
    prior_path = config["prior_path"]
    param_dict = TOML.parsefile(prior_path)
    prior_vec = [get_parameter_distribution(param_dict, n) for n in names]
    prior = combine_distributions(prior_vec)
    save_parameter_ensemble(
        EKP.get_u_final(eki),  # constraints applied when saving
        prior,
        param_dict,
        output_dir,
        "parameters.toml",
        iteration,
    )

    # Save EKI object for next iteration
    iter_path = path_to_iteration(output_dir, iteration)
    eki_path = joinpath(iter_path, "eki_file.jld2")
    JLD2.save_object(eki_path, eki)
end

#!/bin/sh
#SBATCH --time=00:30:00

experiment_id=$1
i=$2
echo "Running update step after iteration $i"

julia --color=no --project=experiments/$experiment_id -e '
    import YAML, JLD2
    import CalibrateAtmos

    experiment_id = "'$experiment_id'"
    i = '$i'
    include("experiments/'$experiment_id'/model_interface.jl")
    G_ensemble = CalibrateAtmos.observation_map(Val(:'$experiment_id'), i)
    config = YAML.load_file(joinpath("experiments", experiment_id, "ekp_config.yml"))
    output_dir = config["output_dir"]
    iter_path = CalibrateAtmos.path_to_iteration(output_dir, i)
    JLD2.save_object(joinpath(iter_path, "observation_map.jld2"), G_ensemble)
    CalibrateAtmos.update_ensemble(experiment_id, i)
'
echo "Update step for iteration $i complete"

#!/bin/sh
#SBATCH --time=00:30:00

experiment_id=$1
i=$2

echo "Iteration $i"

julia --project=experiments -e '
    import YAML, JLD2
    import CalibrateAtmos

    experiment_id = "'$experiment_id'"
    i = '$i'
    include("experiments/$experiment_id/observation_map.jl")
    G_ensemble = observation_map(i)
    config = YAML.load_file(joinpath("experiments", experiment_id, "ekp_config.yml"))
    output_dir = config["output_dir"]
    iter_path = CalibrateAtmos.path_to_iteration(output_dir, i)
    JLD2.save_object(joinpath(iter_path, "observation_map.jld2"), G_ensemble)
    CalibrateAtmos.update_ensemble(experiment_id, i)
'

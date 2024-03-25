#!/bin/sh
#SBATCH --time=00:30:00

experiment_id=$1
i=$2
echo "Running update step after iteration $i"

julia --color=no --project=experiments/$experiment_id -e '
    import CalibrateAtmos as CAL

    experiment_id = "'$experiment_id'"
    i = '$i'
    include("experiments/'$experiment_id'/model_interface.jl")
    G_ensemble = CAL.observation_map(Val(Symbol(experiment_id)), i)
    CAL.save_G_ensemble(experiment_id, i, G_ensemble)
    CAL.update_ensemble(experiment_id, i)
'
echo "Update step for iteration $i complete"

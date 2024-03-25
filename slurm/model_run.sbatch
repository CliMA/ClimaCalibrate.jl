#!/bin/bash

# Extract command-line arguments
experiment_id=$1
i=$2

# Find output directory
format_i=$(printf "iteration_%03d" "$i")
member=$(printf "member_%03d" "$SLURM_ARRAY_TASK_ID")
output=output/$experiment_id/$format_i/$member/model_log.txt

# Run the forward model
srun --output=$output julia --color=no --project=experiments/$experiment_id -e '
    import CalibrateAtmos as CAL
    
    experiment_id = "'$experiment_id'"
    i = '$i'
    member = '$SLURM_ARRAY_TASK_ID'

    include("experiments/$experiment_id/model_interface.jl")

    physical_model = CAL.get_forward_model(Val(Symbol(experiment_id)))
    config = CAL.get_config(physical_model, member, i, experiment_id)
    CAL.run_forward_model(physical_model, config)
'

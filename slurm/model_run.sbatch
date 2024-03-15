#!/bin/bash

# Extract command-line arguments
experiment_id=$1
iteration=$2

# Find output directory
format_i=$(printf "iteration_%03d" "$iteration")
member=$(printf "member_%03d" "$SLURM_ARRAY_TASK_ID")
output=output/$experiment_id/$format_i/$member/model_log.out

# Run the forward model
srun --output=$output julia --color=no --project=experiments/$experiment_id -e "
    import CalibrateAtmos
    include(\"experiments/$experiment_id/model_interface.jl\")

    physical_model = CalibrateAtmos.get_forward_model(Val(:$experiment_id))
    config = CalibrateAtmos.get_config(physical_model, $SLURM_ARRAY_TASK_ID, $iteration, \"$experiment_id\")
    CalibrateAtmos.run_forward_model(physical_model, config)
"

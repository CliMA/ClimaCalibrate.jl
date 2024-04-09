#!/bin/bash
set -euo pipefail

source slurm/parse_commandline.sh
if [ ! -d "$output" ] ; then
    mkdir -p "$output"
fi

# Initialize the project and setup calibration
init_id=$(sbatch --parsable \
                 --output="$logfile" \
                 --partition="$partition" \
                 --export=generate_data="$generate_data" \
                 slurm/initialize.sbatch "$experiment_id")
echo -e "Initialization job_id: $init_id\n"

# Loop over iterations
dependency="afterok:$init_id"
for i in $(seq 0 $((n_iterations - 1)))
do
    echo "Scheduling iteration $i"
    ensemble_array_id=$(
        sbatch --dependency="$dependency" --kill-on-invalid-dep=yes --parsable \
                --job=model-"$i" \
                --output=/dev/null \
                --array=1-"$ensemble_size" \
                --time="$slurm_time" \
                --ntasks="$slurm_ntasks" \
                --partition="$partition" \
                --cpus-per-task="$slurm_cpus_per_task" \
                --gpus-per-task="$slurm_gpus_per_task" \
                slurm/model_run.sbatch "$experiment_id" "$i")

    dependency=afterany:$ensemble_array_id
    echo "Iteration $i job id: $ensemble_array_id"

    update_id=$(
        sbatch --dependency="$dependency" --kill-on-invalid-dep=yes --parsable \
               --job=update-"$i" \
               --output="$logfile" \
               --open-mode=append \
               --partition="$partition" \
               slurm/update.sbatch "$experiment_id" "$i")

    dependency=afterok:$update_id
    echo -e "Update $i job id: $update_id\n"
done

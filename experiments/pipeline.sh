#!/bin/bash
# Configure the environment
export MODULEPATH=/groups/esm/modules:$MODULEPATH
module load climacommon/2024_02_27

# Parse command line
experiment_id=${1?Error: no experiment ID given}
tasks_per_model_run=${2?Error: no tasks per model run given}

# Get ensemble size, number of iterations, and output dir from EKP config file
ensemble_size=$(grep "ensemble_size:" experiments/$experiment_id/ekp_config.yml | awk '{print $2}')
n_iterations=$(grep "n_iterations:" experiments/$experiment_id/ekp_config.yml | awk '{print $2}')
output=$(grep "output_dir:" experiments/$experiment_id/ekp_config.yml | awk '{print $2}')

mkdir $output

echo "Running experiment $experiment_id with $tasks_per_model_run tasks per model run"
init_id=$(sbatch --parsable \
                 --output=$output/log.out \
                 --open-mode=append \
                 --partition=expansion \
                 experiments/initialize.sbatch $experiment_id)
echo "Initialization job_id: $init_id"
echo ""

# Loop over iterations
dependency="afterok:$init_id"
for i in $(seq 0 $((n_iterations - 1)))
do
    echo "Scheduling iteration $i"
    format_i=$(printf "iteration_%03d" "$i")

    ensemble_array_id=$(
        sbatch --dependency=$dependency --kill-on-invalid-dep=yes --parsable \
               --job=model-$i \
               --output=/dev/null \
               --ntasks=$tasks_per_model_run \
               --array=1-$ensemble_size \
               --partition=expansion \
               experiments/model_run.sbatch $experiment_id $i)

    dependency=afterany:$ensemble_array_id
    echo "Iteration $i job id: $ensemble_array_id"

    update_id=$(
        sbatch --dependency=$dependency --kill-on-invalid-dep=yes --parsable \
               --job=update-$i \
               --output=$output/log.out \
               --open-mode=append \
               --partition=expansion \
               experiments/update.sbatch $experiment_id $i)

    dependency=afterany:$update_id
    echo "Update $i job id: $update_id"
    echo ""
done

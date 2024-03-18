#!/bin/bash
export MODULEPATH=/groups/esm/modules:$MODULEPATH
module load climacommon/2024_03_18

source slurm/parse_commandline.sh
if [ ! -d $output ] ; then
    mkdir -p $output
fi

# Initialize the project and setup calibration
init_id=$(sbatch --parsable \
                 --output=$logfile \
                 --partition=$partition \
                 slurm/initialize.sbatch $experiment_id)
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
                --array=1-$ensemble_size \
                --time=$slurm_time \
                --ntasks=$slurm_ntasks \
                --partition=$partition \
                --cpus-per-task=$slurm_cpus_per_task \
                --gpus-per-task=$slurm_gpus_per_task \
                slurm/model_run.sbatch $experiment_id $i
    )

    dependency=afterany:$ensemble_array_id
    echo "Iteration $i job id: $ensemble_array_id"

    update_id=$(
        sbatch --dependency=$dependency --kill-on-invalid-dep=yes --parsable \
               --job=update-$i \
               --output=$logfile \
               --open-mode=append \
               --partition=$partition \
               slurm/update.sbatch $experiment_id $i)

    dependency=afterany:$update_id
    echo "Update $i job id: $update_id"
    echo ""
done

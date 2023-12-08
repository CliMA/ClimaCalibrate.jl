#!/bin/bash

# Configure for Resnick HPCC
module load julia/1.9.4 cuda/12.2 ucx/1.14.1_cuda-12.2 openmpi/4.1.5_cuda-12.2 nsight-systems/2023.3.1
export OPENBLAS_NUM_THREADS=1
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export JULIA_MAX_NUM_PRECOMPILE_FILES=100
export JULIA_CPU_TARGET="broadwell;skylake"
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_HAS_CUDA="true"
export MPITRAMPOLINE_LIB=/groups/esm/software/MPIwrapper/ompi4.1.5_cuda-12.2/lib64/libmpiwrapper.so
export MPITRAMPOLINE_MPIEXEC=/groups/esm/software/MPIwrapper/ompi4.1.5_cuda-12.2/bin/mpiwrapperexec

# Parse command line
experiment_id=${1?Error: no experiment ID given}
tasks_per_model_run=${2?Error: no tasks per model run given}

echo "Running experiment $experiment_id with $tasks_per_model_run tasks per model run."
echo 'Initializing ensemble for calibration.'
init_id=$(sbatch --parsable \
                 --output=init_$experiment_id.out \
                 experiments/initialize.sbatch $experiment_id)

# Get ensemble size and the number of iterations from configuration file
ensemble_size=$(grep "ensemble_size:" experiments/$experiment_id/ekp_config.yml | awk '{print $2}')
n_iterations=$(grep "n_iterations:" experiments/$experiment_id/ekp_config.yml | awk '{print $2}')

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
               experiments/model_run.sbatch $experiment_id $i)

    dependency=afterany:$ensemble_array_id
    echo "Iteration $i job id: $ensemble_array_id"

    update_id=$(
        sbatch --dependency=$dependency --kill-on-invalid-dep=yes --parsable \
               --job=update-$i \
               --output=output/$experiment_id/$format_i/update_log.out \
               experiments/update.sbatch $experiment_id $i)

    dependency=afterany:$update_id
    echo "Update $i job id: $update_id"
done

# Configure for Resnick HPCC
module load julia/1.10.0 cuda/12.2 ucx/1.14.1_cuda-12.2 openmpi/4.1.5_cuda-12.2 nsight-systems/2023.3.1
export OPENBLAS_NUM_THREADS=1
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export JULIA_MAX_NUM_PRECOMPILE_FILES=100
export JULIA_CPU_TARGET="broadwell;skylake"
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_HAS_CUDA="true"
export MPITRAMPOLINE_LIB=/groups/esm/software/MPIwrapper/ompi4.1.5_cuda-12.2/lib64/libmpiwrapper.so
export MPITRAMPOLINE_MPIEXEC=/groups/esm/software/MPIwrapper/ompi4.1.5_cuda-12.2/bin/mpiwrapperexec
#!/bin/bash
#PBS -A UCIT0011
#PBS -N my_test_job
#PBS -q preempt
#PBS -m n
#PBS -l walltime=00:20:00
#PBS -l select=4:ncpus=64:mem=480GB:ngpus=4

# Use scratch for temporary files to avoid space limits in /tmp
export TMPDIR=${SCRATCH}/temp
mkdir -p ${TMPDIR}

# If you are using zsh as default shell
source /glade/u/apps/derecho/23.09/spack/opt/spack/lmod/8.7.24/gcc/7.5.0/c645/lmod/lmod/init/zsh
export MODULEPATH="/glade/campaign/univ/ucit0011/ClimaModules-Derecho:$MODULEPATH"
module purge
module load climacommon/2024_05_27


$MPITRAMPOLINE_MPIEXEC -n 16 -ppn 4 set_gpu_rank julia --color=yes --project=experiments/surface_fluxes_perfect_model -e '
import ClimaCalibrate as CAL
iteration = 1; member = 1
model_interface = "model_interface.jl"; include(model_interface)

experiment_dir = "exp/dir"
CAL.run_forward_model(CAL.set_up_forward_model(member, iteration, experiment_dir))
'
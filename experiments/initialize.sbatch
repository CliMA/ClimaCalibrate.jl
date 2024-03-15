#!/bin/sh
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job init_calibration

experiment_id=$1

echo "Initializing calibration for experiment: $experiment_id"
julia --color=no --project=experiments/$experiment_id -e 'using Pkg; Pkg.instantiate(;verbose=true)'

julia --color=no --project=experiments/$experiment_id -e '
    import CalibrateAtmos
    CalibrateAtmos.initialize("'$experiment_id'")
'

echo "Calibration initialized"

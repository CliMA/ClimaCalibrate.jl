#!/bin/sh
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job init_calibration

experiment_id=$1

echo "Initializing calibration for experiment $experiment_id"
julia --project=experiments -e 'using Pkg; Pkg.instantiate(;verbose=true)'
julia --project=experiments -e "
import CalibrateAtmos
using Distributions
using EnsembleKalmanProcesses.ParameterDistributions
CalibrateAtmos.initialize(\"$experiment_id\")"
echo "Calibration initialized."
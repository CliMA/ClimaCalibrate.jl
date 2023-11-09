#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --output="experiments/sphere_held_suarez_rhoe_equilmoist/truth_simulation/model_log.out"

module load julia/1.9.3 cuda/12.2 ucx/1.14.1_cuda-12.2 openmpi/4.1.5_cuda-12.2 hdf5/1.12.2-ompi415
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export JULIA_NUM_PRECOMPILE_TASKS=8
export JULIA_CPU_TARGET='broadwell;skylake'

echo "Generating truth observations."

srun  --open-mode=append julia --project=experiments -e '
using ClimaComms
ClimaComms.init(ClimaComms.context())
import ClimaAtmos as CA
import YAML
using NetCDF
import JLD2

experiment_dir = joinpath("experiments", "sphere_held_suarez_rhoe_equilmoist")
output_dir = joinpath(experiment_dir, "truth_simulation")

config_dict = YAML.load_file(joinpath(experiment_dir, "atmos_config.yml"))
config_dict["t_end"] = "3000days"
config_dict["output_dir"] = output_dir

config = CA.AtmosConfig(config_dict)
integrator = CA.get_integrator(config)
CA.solve_atmos!(integrator)

ta = ncread(joinpath(output_dir, "ta_60.0d_average.nc"), "ta")
include(joinpath(experiment_dir, "observation_map.jl"))
(; observation, variance) = process_member_data(ta; output_variance = true)
@show observation
@show variance
JLD2.save_object(joinpath(experiment_dir, "obs_mean.jld2"), observation)
JLD2.save_object(joinpath(experiment_dir, "obs_noise_cov.jld2"), variance)
'

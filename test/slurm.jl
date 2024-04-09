# Script for running the SurfaceFluxes calibration on slurm. Used in buildkite testing
# To run: julia --project=experiments/surface_fluxes_perfect_model test/slurm.jl

import CalibrateAtmos
using Test
import EnsembleKalmanProcesses: get_ϕ_mean_final, get_g_mean_final

experiment_dir = dirname(Base.active_project());

# Generate observational data and include observational map 
include(joinpath(experiment_dir, "generate_data.jl"))
include(joinpath(experiment_dir, "observation_map.jl"));

model_interface = joinpath(experiment_dir, "model_interface.jl");
eki = CalibrateAtmos.slurm_calibration(; time_limit = "3", model_interface);

include(joinpath(experiment_dir, "postprocessing.jl"))

@testset "End to end test using file config (surface fluxes perfect model)" begin
    parameter_values = get_ϕ_mean_final(prior, eki)
    test_parameter_values = [4.8636494875560246, 5.18733785098307]
    @test all(isapprox.(parameter_values, test_parameter_values; rtol = 1e-3))

    forward_model_output = get_g_mean_final(eki)
    test_model_output = [0.04756260327994823]
    @test all(isapprox.(forward_model_output, test_model_output; rtol = 1e-3))
end
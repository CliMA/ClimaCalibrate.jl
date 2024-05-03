# Tests for SurfaceFluxes example calibration on slurm, used in buildkite testing
# To run, open the REPL: julia --project=experiments/surface_fluxes_perfect_model
# And include this file

import CalibrateAtmos:
    get_backend, CaltechHPC, JuliaBackend, calibrate, get_prior
using Test
import EnsembleKalmanProcesses: get_ϕ_mean_final, get_g_mean_final

experiment_dir = dirname(Base.active_project())
model_interface = joinpath(experiment_dir, "model_interface.jl")

# Generate observational data and include observational map 
include(joinpath(experiment_dir, "generate_data.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))
include(model_interface)

prior = get_prior(joinpath(experiment_dir, "prior.toml"))

function test_sf_calibration_output(eki, prior)
    @testset "End to end test using file config (surface fluxes perfect model)" begin
        parameter_values = get_ϕ_mean_final(prior, eki)
        test_parameter_values = [4.8636494875560246, 5.18733785098307]
        @test all(
            isapprox.(parameter_values, test_parameter_values; rtol = 1e-3),
        )

        forward_model_output = get_g_mean_final(eki)
        test_model_output = [0.04756260327994823]
        @test all(
            isapprox.(forward_model_output, test_model_output; rtol = 1e-3),
        )
    end
end

# Test Caltech HPC backend
backend = get_backend()
@test backend == CaltechHPC

eki = calibrate(
    backend,
    experiment_dir;
    time_limit = 5,
    model_interface,
    verbose = true,
)
test_sf_calibration_output(eki, prior)

# Pure Julia Backend
eki = calibrate(JuliaBackend, experiment_dir)
test_sf_calibration_output(eki, prior)

include(joinpath(experiment_dir, "postprocessing.jl"))

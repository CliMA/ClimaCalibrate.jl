# Tests for SurfaceFluxes example calibration on slurm, used in buildkite testing
# To run, open the REPL: julia --project=experiments/surface_fluxes_perfect_model test/slurm_backend_e2e.jl

using Pkg;
pkg"instantiate";
import ClimaCalibrate:
    get_backend, HPCBackend, JuliaBackend, calibrate, get_prior, kwargs
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

@assert get_backend() <: HPCBackend
hpc_kwargs = kwargs(time = 5, ncpus = 1, ncpus_per_task = 1)
eki = calibrate(experiment_dir; model_interface, hpc_kwargs, verbose = true)
test_sf_calibration_output(eki, prior)

# Pure Julia calibration, this should run anywhere
eki = calibrate(JuliaBackend, experiment_dir)
test_sf_calibration_output(eki, prior)

include(joinpath(experiment_dir, "postprocessing.jl"))

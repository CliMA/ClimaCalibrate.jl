import ClimaCalibrate

using EnsembleKalmanProcesses.ParameterDistributions
using Test

# Tests for ensuring ClimaCalibrate has protected interfaces. The API tested below must be defined for each model,
# otherwise ClimaCalibrate will throw an error.

@testset "Model Interface stubs" begin

    @testset "set_up_forward_model" begin
        prior_path = joinpath(
            pkgdir(ClimaCalibrate),
            "experiments",
            "surface_fluxes_perfect_model",
            "prior.toml",
        )
        experiment_dir = ClimaCalibrate.ExperimentConfig(
            1,
            1,
            [1],
            [1],
            ClimaCalibrate.get_prior(prior_path),
            "output",
            false,
        )
        @test_throws ErrorException("set_up_forward_model not implemented") ClimaCalibrate.set_up_forward_model(
            1,
            1,
            experiment_dir,
        )
    end

    @testset "run_forward_model" begin
        @test_throws ErrorException("run_forward_model not implemented") ClimaCalibrate.run_forward_model(
            nothing,
        )
    end

    @testset "observation_map" begin
        @test_throws ErrorException("observation_map not implemented") ClimaCalibrate.observation_map(
            1,
        )
    end
end

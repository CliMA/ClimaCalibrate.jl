import ClimaCalibrate

using EnsembleKalmanProcesses.ParameterDistributions
using Test

# Tests for ensuring ClimaCalibrate has protected interfaces. The API tested below must be defined for each model,
# otherwise ClimaCalibrate will throw an error.

struct TestPhysicalModel <: ClimaCalibrate.AbstractPhysicalModel end

@testset "Model Interface stubs" begin

    @testset "get_config" begin
        @test_throws ErrorException(
            "get_config not implemented for TestPhysicalModel()",
        ) ClimaCalibrate.get_config(TestPhysicalModel(), 1, 1, Dict{Any, Any}())
    end

    @testset "run_forward_model" begin
        @test_throws ErrorException(
            "run_forward_model not implemented for TestPhysicalModel()",
        ) ClimaCalibrate.run_forward_model(
            TestPhysicalModel(),
            Dict{Any, Any}(),
        )
    end

    @testset "get_forward_model" begin
        @test_throws ErrorException(
            "get_forward_model not implemented for Val{:test}()",
        ) ClimaCalibrate.get_forward_model(Val(:test))
    end

    @testset "observation_map" begin
        @test_throws ErrorException(
            "observation_map not implemented for experiment Val{:test}() at iteration 1",
        ) ClimaCalibrate.observation_map(Val(:test), 1)
    end

    @testset "calibrate func" begin
        experiment_config = ClimaCalibrate.ExperimentConfig(
            "test",
            1,
            1,
            [20.0],
            [0.01;;],
            constrained_gaussian("test_param", 10, 5, 0, Inf),
            joinpath("test", "e2e_test_output"),
            false,
        )
        @test_throws ErrorException ClimaCalibrate.calibrate(experiment_config)
    end

end

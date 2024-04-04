import CalibrateAtmos

using EnsembleKalmanProcesses.ParameterDistributions
using Test

# Tests for ensuring CalibrateAtmos has protected interfaces. The API tested below must be defined for each model,
# otherwise CalibrateAtmos will throw an error.

struct TestPhysicalModel <: CalibrateAtmos.AbstractPhysicalModel end

@testset "Model Interface stubs" begin

    @testset "get_config" begin
        @test_throws ErrorException(
            "get_config not implemented for TestPhysicalModel()",
        ) CalibrateAtmos.get_config(TestPhysicalModel(), 1, 1, Dict{Any, Any}())
    end

    @testset "run_forward_model" begin
        @test_throws ErrorException(
            "run_forward_model not implemented for TestPhysicalModel()",
        ) CalibrateAtmos.run_forward_model(
            TestPhysicalModel(),
            Dict{Any, Any}(),
        )
    end

    @testset "get_forward_model" begin
        @test_throws ErrorException(
            "get_forward_model not implemented for Val{:test}()",
        ) CalibrateAtmos.get_forward_model(Val(:test))
    end

    @testset "observation_map" begin
        @test_throws ErrorException(
            "observation_map not implemented for experiment Val{:test}() at iteration 1",
        ) CalibrateAtmos.observation_map(Val(:test), 1)
    end

    @testset "calibrate func" begin
        experiment_config = CalibrateAtmos.ExperimentConfig(
            "test",
            1,
            1,
            [20.0],
            [0.01;;],
            constrained_gaussian("test_param", 10, 5, 0, Inf),
            joinpath("test", "e2e_test_output"),
            false,
            false,
        )
        @test_throws ErrorException CalibrateAtmos.calibrate(experiment_config)
    end

end

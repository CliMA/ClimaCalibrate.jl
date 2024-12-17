import ClimaCalibrate

using EnsembleKalmanProcesses.ParameterDistributions
using Test

# Tests for ensuring ClimaCalibrate has protected interfaces. The API tested below must be defined for each model,
# otherwise ClimaCalibrate will throw an error.

@testset "Model Interface stubs" begin
    @test_throws ErrorException("forward_model not implemented") ClimaCalibrate.forward_model(
        1,
        1,
    )
    @test_throws ErrorException("observation_map not implemented") ClimaCalibrate.observation_map(
        1,
    )
end

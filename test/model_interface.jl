import ClimaCalibrate

using EnsembleKalmanProcesses.ParameterDistributions
using Test

# Tests for ensuring ClimaCalibrate has protected interfaces. The API tested
# below must be defined for each model, otherwise ClimaCalibrate will throw an
# error. However, `analyze_iteration` and `postprocess_g_ensemble` are
# optional to implement.

@testset "Model Interface stubs" begin
    ctx = ClimaCalibrate.Context.CalibrationContext(
        0,
        nothing,
        "idk",
        10,
        nothing,
        nothing,
        nothing,
    )
    @test_throws ErrorException("forward_model not implemented") ClimaCalibrate.forward_model(
        ctx,
    )
    @test_throws ErrorException("observation_map not implemented") ClimaCalibrate.observation_map(
        ctx,
    )
    @test isnothing(ClimaCalibrate.analyze_iteration(ctx, 2))
    @test 2 == ClimaCalibrate.postprocess_g_ensemble(ctx, 2)
end

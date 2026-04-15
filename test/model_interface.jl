import ClimaCalibrate

using EnsembleKalmanProcesses.ParameterDistributions
using Test

# Tests for ensuring ClimaCalibrate has protected interfaces. The API tested
# below must be defined for each model, otherwise ClimaCalibrate will throw an
# error. However, `analyze_iteration` and `postprocess_g_ensemble` are
# optional to implement.

# TODO: Better name than this? Maybe CalibrationProblem or CalibrationInterface
# or ModelInterface?
struct DummyContext <: ClimaCalibrate.AbstractCalibrationContext end

@testset "Model Interface stubs" begin
    ctx = DummyContext()
    @test_throws ErrorException("forward_model not implemented for DummyContext") ClimaCalibrate.forward_model(
        ctx,
        1,
        1,
    )
    @test_throws ErrorException(
        "observation_map not implemented for DummyContext",
    ) ClimaCalibrate.observation_map(ctx, 1)
    @test isnothing(ClimaCalibrate.analyze_iteration(ctx, 1, 1, 1, 1, 1))
    @test 2 == ClimaCalibrate.postprocess_g_ensemble(ctx, 1, 2, 3, 4, 5)
end

# TODO: Update these tests

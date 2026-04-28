import ClimaCalibrate

using EnsembleKalmanProcesses.ParameterDistributions
using Test

# Tests for ensuring ClimaCalibrate has protected interfaces. The API tested
# below must be defined for each model, otherwise ClimaCalibrate will throw an
# error. However, `analyze_iteration` and `postprocess_g_ensemble` are
# optional to implement.

struct DummyModelInterface <: ClimaCalibrate.AbstractModelInterface end

@testset "Model Interface stubs" begin
    interface = DummyModelInterface()
    @test_throws ErrorException(
        "forward_model not implemented for DummyModelInterface",
    ) ClimaCalibrate.forward_model(interface, 1, 1)
    @test_throws ErrorException(
        "observation_map not implemented for DummyModelInterface",
    ) ClimaCalibrate.observation_map(interface, 1)
    @test isnothing(ClimaCalibrate.analyze_iteration(interface, 1, 1, 1, 1, 1))
    @test 2 == ClimaCalibrate.postprocess_g_ensemble(interface, 1, 2, 3, 4, 5)
    @test_throws ErrorException(
        "model_interface_filepath not implemented for DummyModelInterface",
    ) ClimaCalibrate.model_interface_filepath(DummyModelInterface())
    @test ClimaCalibrate.experiment_dir(DummyModelInterface()) ==
          ClimaCalibrate.project_dir()
    @test ClimaCalibrate.exeflags(DummyModelInterface()) == ""
end

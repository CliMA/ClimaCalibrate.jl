import CalibrateAtmos
import EnsembleKalmanProcesses: get_u_mean_final, get_g_mean_final
using Test

experiment_id = "surface_fluxes_perfect_model"
experiment_path = joinpath(pkgdir(CalibrateAtmos), "experiments", experiment_id)
include(joinpath(experiment_path, "model_interface.jl"))
include(joinpath(experiment_path, "generate_truth.jl"))

eki = CalibrateAtmos.calibrate(experiment_id)

@testset "Test pure Julia calibration (surface fluxes perfect model)" begin
    parameter_values = get_u_mean_final(eki)
    test_parameter_values = [0.930140658789275, 0.08095811150600453]
    @test all(isapprox.(parameter_values, test_parameter_values; rtol = 1e-3))

    forward_model_output = get_g_mean_final(eki)
    test_model_output = [0.04734060615301132]
    @test all(isapprox.(parameter_values, test_parameter_values; rtol = 1e-3))
end

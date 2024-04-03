import CalibrateAtmos
import EnsembleKalmanProcesses: get_ϕ_mean_final, get_g_mean_final
using Test

experiment_id = "surface_fluxes_perfect_model"
experiment_path = joinpath(pkgdir(CalibrateAtmos), "experiments", experiment_id)
include(joinpath(experiment_path, "model_interface.jl"))
include(joinpath(experiment_path, "generate_data.jl"))

prior = CalibrateAtmos.get_prior(joinpath(experiment_path, "prior.toml"))
eki = CalibrateAtmos.calibrate(experiment_id)

@testset "Test pure Julia calibration (surface fluxes perfect model)" begin
    parameter_values = get_ϕ_mean_final(prior, eki)
    test_parameter_values = [4.8684152849621976, 5.2022848059758875]
    @test all(isapprox.(parameter_values, test_parameter_values; rtol = 1e-3))

    forward_model_output = get_g_mean_final(eki)
    test_model_output = [0.04734060615301132]
    @test all(isapprox.(forward_model_output, test_model_output; rtol = 1e-3))
end

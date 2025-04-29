using Test

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaParams as CP

import ClimaCalibrate as CAL
import JLD2

# Experiment Info
output_file = "model_output.jld2"
prior = constrained_gaussian("test_param", 10, 5, 0, Inf)
n_iterations = 1
ensemble_size = 20
observations = [20.0]
noise = [0.01;;]
output_dir = mktempdir()

# Model interface
# This "model" just samples parameters and returns them, we are checking that the 
# results are reproducible.
function CAL.forward_model(iteration, member)
    member_path = CAL.path_to_ensemble_member(output_dir, iteration, member)
    param_path = CAL.parameter_path(output_dir, iteration, member)
    toml_dict = CP.create_toml_dict(Float64; override_file = param_path)
    (; test_param) = CP.get_parameter_values(toml_dict, "test_param")
    JLD2.save_object(joinpath(member_path, output_file), test_param)
end

function CAL.observation_map(iteration)
    dims = 1
    G_ensemble = Array{Float64}(undef, dims..., ensemble_size)
    for m in 1:ensemble_size
        member_path =
            TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
        output = JLD2.load_object(joinpath(member_path, output_file))
        G_ensemble[:, m] .= output
    end
    return G_ensemble
end

function CAL.analyze_iteration(ekp, g_ensemble, prior, output_dir, iteration)
    @info "Analyzing iteration"
    @info "Iteration $iteration"
    @info "Current mean constrained parameter: $(EKP.get_ϕ_mean_final(prior, ekp))"
    @info "g_ensemble: $g_ensemble"
    @info "output_dir: $output_dir"
    return nothing
end

ekp = CAL.calibrate(
    CAL.JuliaBackend,
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    output_dir,
)

@testset "Test end-to-end calibration" begin
    parameter_values =
        [EKP.get_ϕ_mean(prior, ekp, it) for it in 1:(n_iterations + 1)]
    @test parameter_values[1][1] ≈ 8.507 rtol = 0.01
    @test parameter_values[end][1] ≈ 11.852161842745355 rtol = 0.01
end

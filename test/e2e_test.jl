using Test

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaParams as CP

import CalibrateAtmos:
    AbstractPhysicalModel,
    get_config,
    run_forward_model,
    get_forward_model,
    ExperimentConfig,
    calibrate,
    observation_map

import JLD2

# Experiment Info
id = "e2e_test"
output_file = "model_output.jld2"
prior = constrained_gaussian("test_param", 10, 5, 0, Inf)
n_iterations = 1
ensemble_size = 10
observations = [20.0]
noise = [0.01;;]
output_dir = joinpath("test",  "e2e_test_output")

experiment_config = ExperimentConfig(
    id;
    prior,
    n_iterations,
    ensemble_size,
    observations,
    noise,
    output_dir,
)

# Model interface
struct TestModel end

TestModel <: AbstractPhysicalModel

get_forward_model(::Val{:e2e_test}) = TestModel()

function get_config(
    physical_model::TestModel,
    member,
    iteration,
    experiment_id::AbstractString,
)
    model_config = Dict()
    output_dir = (experiment_config.output_dir)
    member_path = path_to_ensemble_member(output_dir, iteration, member)
    model_config["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    model_config["toml"] = parameter_path
    return model_config
end

function run_forward_model(::TestModel, config)
    toml_dict = CP.create_toml_dict(Float64; override_file = config["toml"])
    (; test_param) = CP.get_parameter_values(toml_dict, "test_param")
    output = test_param
    JLD2.save_object(joinpath(config["output_dir"], output_file), output)
end

# Observation map
function observation_map(::Val{:e2e_test}, iteration)

    (; ensemble_size) = experiment_config
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

# Test!
ekp = calibrate(experiment_config)

@testset "Test dummy end-to-end calibration" begin
    parameter_values = [EKP.get_ϕ_mean(prior, ekp, it) for it in 1:n_iterations+1]
    @test parameter_values[1][1] ≈ 9.779 rtol = 0.01
    @test parameter_values[end][1] ≈ 19.63 rtol = 0.01
end


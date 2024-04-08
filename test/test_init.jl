using Distributions
using EnsembleKalmanProcesses.ParameterDistributions
import CalibrateAtmos
import ClimaParams as CP
import LinearAlgebra: I
using Test

FT = Float64
output_dir = "test_init"
prior_path = joinpath("test_case_inputs", "prior.toml")
param_names = ["one", "two"]

prior = CalibrateAtmos.get_prior(prior_path)
noise = 0.1 * I
observations = zeros(Float64, 1)
n_iterations = 1
ensemble_size = 10

config = CalibrateAtmos.ExperimentConfig(
    "test",
    n_iterations,
    ensemble_size,
    observations,
    noise,
    prior,
    output_dir,
    false,
    false,
)

CalibrateAtmos.initialize(config)

override_file = joinpath(
    config.output_dir,
    "iteration_000",
    "member_001",
    "parameters.toml",
)
td = CP.create_toml_dict(FT; override_file)
params = CP.get_parameter_values(td, param_names)

@testset "Initialized parameter values" begin
    # This checks for random seed as well
    @test params.one == 1.8171573383720587
    @test params.two == 5.408386812503563
end

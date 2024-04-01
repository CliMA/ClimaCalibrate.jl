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

CalibrateAtmos.initialize(
    "test";
    prior,
    ensemble_size,
    output_dir,
    noise,
    observations,
    rng_seed = 4444,
)

override_file =
    joinpath(output_dir, "iteration_000", "member_001", "parameters.toml")
td = CP.create_toml_dict(FT; override_file)
params = CP.get_parameter_values(td, param_names)

@testset "Initialized parameter values" begin
    # This checks for random seed as well
    @test params.one == 2.7212695972038583
    @test params.two == 4.891199610995353
end

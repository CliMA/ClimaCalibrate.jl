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
config = Dict(
    "output_dir" => output_dir,
    "prior_path" => prior_path,
    "ensemble_size" => 1,
)
Γ = 0.1 * I
y = zeros(Float64, 1)

CalibrateAtmos.initialize("test"; config, Γ, y, rng_seed = 4444)

override_file =
    joinpath(output_dir, "iteration_000", "member_001", "parameters.toml")
td = CP.create_toml_dict(FT; override_file)
params = CP.get_parameter_values(td, param_names)

@testset "Initialized parameter values" begin
    # This checks for random seed as well
    @test params.one == 2.7212695972038583
    @test params.two == 4.891199610995353
end

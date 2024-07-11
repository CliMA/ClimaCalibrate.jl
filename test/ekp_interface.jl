using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
import ClimaCalibrate as CAL
import ClimaParams as CP
import LinearAlgebra: I
using Test

FT = Float64
output_dir = "test_init"
prior_path = joinpath(pkgdir(CAL), "test", "test_case_inputs", "prior.toml")
param_names = ["one", "two"]

prior = CAL.get_prior(prior_path)
noise = 0.1 * I
observations = zeros(Float64, 1)
n_iterations = 1
ensemble_size = 10

config = CAL.ExperimentConfig(
    n_iterations,
    ensemble_size,
    observations,
    noise,
    prior,
    output_dir,
)

eki = CAL.initialize(config)
eki_with_kwargs = CAL.initialize(
    config;
    scheduler = EKP.MutableScheduler(2),
    accelerator = EKP.NesterovAccelerator(),
)

@testset "Test passing kwargs to EKP struct" begin
    @test eki_with_kwargs.scheduler != eki.scheduler
    @test eki_with_kwargs.scheduler isa EKP.MutableScheduler

    @test eki_with_kwargs.accelerator != eki.accelerator
    @test eki_with_kwargs.accelerator isa EKP.NesterovAccelerator
end

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

@testset "Environment variables" begin
    @test_throws ErrorException(
        "Experiment dir not found in environment. Ensure that env variable \"CALIBRATION_EXPERIMENT_DIR\" is set.",
    ) CAL.env_experiment_dir()
    @test_throws ErrorException(
        "Iteration number not found in environment. Ensure that env variable \"CALIBRATION_ITERATION\" is set.",
    ) CAL.env_iteration()
    @test_throws ErrorException(
        "Member number not found in environment. Ensure that env variable \"CALIBRATION_MEMBER_NUMBER\" is set.",
    ) CAL.env_member_number()
    @test_throws ErrorException(
        "Model interface file not found in environment. Ensure that env variable \"CALIBRATION_MODEL_INTERFACE\" is set.",
    ) CAL.env_model_interface()

    test_ENV = Dict()
    test_ENV["CALIBRATION_EXPERIMENT_DIR"] = experiment_dir = "test"
    test_ENV["CALIBRATION_ITERATION"] = "0"
    iter_number = parse(Int, test_ENV["CALIBRATION_ITERATION"])
    test_ENV["CALIBRATION_MEMBER_NUMBER"] = "1"
    member_number = parse(Int, test_ENV["CALIBRATION_MEMBER_NUMBER"])
    test_ENV["CALIBRATION_MODEL_INTERFACE"] =
        model_interface = joinpath(pkgdir(CAL), "model_interface.jl")

    @test experiment_dir == CAL.env_experiment_dir(test_ENV)
    @test iter_number == CAL.env_iteration(test_ENV)
    @test member_number == CAL.env_member_number(test_ENV)
    @test model_interface == CAL.env_model_interface(test_ENV)
end

using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
import ClimaCalibrate as CAL
import ClimaParams as CP
import LinearAlgebra: I
using Test
import Random

rng_seed = 1234
Random.seed!(rng_seed)
rng_ekp = Random.MersenneTwister(rng_seed)

FT = Float64
output_dir = "test_init"
prior_path = joinpath(pkgdir(CAL), "test", "test_case_inputs", "prior.toml")
param_names = ["one", "two"]

prior = CAL.get_prior(prior_path)
noise = 0.1 * I
observations = zeros(Float64, 1)
n_iterations = 1
ensemble_size = 10

user_initial_ensemble =
    EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size)
user_constructed_eki = EKP.EnsembleKalmanProcess(
    user_initial_ensemble,
    observations,
    noise,
    EKP.Inversion(),
    EKP.default_options_dict(EKP.Inversion());
    rng = rng_ekp,
)

eki = CAL.initialize(
    ensemble_size,
    observations,
    noise,
    prior,
    output_dir;
    rng_seed,
)
eki_with_kwargs = CAL.initialize(
    ensemble_size,
    observations,
    noise,
    prior,
    output_dir;
    scheduler = EKP.MutableScheduler(2),
    accelerator = EKP.NesterovAccelerator(),
)

@testset "Test passing kwargs to EKP struct" begin
    @test eki_with_kwargs.scheduler != eki.scheduler
    @test eki_with_kwargs.scheduler isa EKP.MutableScheduler

    @test eki_with_kwargs.accelerator != eki.accelerator
    @test eki_with_kwargs.accelerator isa EKP.NesterovAccelerator
end

@testset "Test that a user-constructed EKP obj is same as initialized one" begin
    for prop in propertynames(eki)
        prop in [:u, :accelerator, :localizer] && continue
        @test getproperty(eki, prop) == getproperty(user_constructed_eki, prop)
    end
    @test eki.u[1].stored_data == user_constructed_eki.u[1].stored_data
end

override_file =
    joinpath(output_dir, "iteration_000", "member_001", "parameters.toml")
td = CP.create_toml_dict(FT; override_file)
params = CP.get_parameter_values(td, param_names)

@testset "Initialized parameter values" begin
    # This checks for random seed as well
    @test params.one == 3.416574531266089
    @test params.two == 4.614950047803855
end

@testset "Test passing an EKP struct into `initialize`" begin
    LHF_target = 4.0
    ensemble_size = 5
    N_iterations = 5
    Γ = 20.0 * EKP.I
    output_dir = joinpath("test", "custom_ekp")
    initial_ensemble =
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size)
    ensemble_kalman_process = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        LHF_target,
        Γ,
        EKP.Inversion(),
    )
    CAL.initialize(ensemble_kalman_process, prior, output_dir)
    override_file =
        joinpath(output_dir, "iteration_000", "member_001", "parameters.toml")
    td = CP.create_toml_dict(FT; override_file)
    params = CP.get_parameter_values(td, param_names)
    @test params.one == 3.1313341622997677
    @test params.two == 5.063035177034372
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

@testset "minibatcher_over_samples tests" begin
    # Regular case
    mb = CAL.minibatcher_over_samples(6, 2)
    @test mb.minibatches == [[1, 2], [3, 4], [5, 6]]

    # Non-divisible case: 7 samples with batch size 2 (should drop last sample)
    mb_partial = CAL.minibatcher_over_samples(7, 2)
    @test mb_partial.minibatches == [[1, 2], [3, 4], [5, 6]]  # 7th is dropped

    # Edge: batch size larger than n_samples
    mb_small = CAL.minibatcher_over_samples(3, 5)
    @test mb_small.minibatches == []

    # Edge: n_samples = 0
    @test_throws ArgumentError CAL.minibatcher_over_samples(0, 2)

    # Edge: batch_size = 0
    @test_throws ArgumentError CAL.minibatcher_over_samples(2, 0)

    # Using vector input
    samples = [1, 2, 3, 4, 5, 6]
    mb2 = CAL.minibatcher_over_samples(samples, 2)
    @test mb2.minibatches == [[1, 2], [3, 4], [5, 6]]
end

@testset "observation_series_from_samples tests" begin
    samples = [EKP.Observation([i], ones(1, 1), string(i)) for i in 1:6]
    # Regular case
    series = CAL.observation_series_from_samples(samples, 2)
    @test series.observations == samples
    @test series.minibatcher.minibatches == [[1, 2], [3, 4], [5, 6]]
    @test series.names == ["1", "2", "3", "4", "5", "6"]

    # fewer samples than batch size
    series2 = CAL.observation_series_from_samples(samples, 7)
    @test series2.minibatcher.minibatches == []  # No batches

    # empty sample list
    @test_throws ArgumentError CAL.observation_series_from_samples(
        EKP.Observation[],
        2,
    )

    # Test mismatched names
    bad_names = ["a", "b", "c"]
    @test_throws ArgumentError CAL.observation_series_from_samples(
        samples,
        2,
        bad_names,
    )
end

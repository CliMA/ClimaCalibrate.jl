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
eki_with_kwargs = EKP.EnsembleKalmanProcess(
    user_initial_ensemble,
    observations,
    noise,
    EKP.Inversion();
    scheduler = EKP.MutableScheduler(2),
    accelerator = EKP.NesterovAccelerator(),
    rng = rng_ekp,
)
CAL.save_eki_and_parameters(eki_with_kwargs, output_dir, 0, prior)

@testset "Test loading latest EKP struct" begin
    # Test loading from directory with no completed iterations
    empty_dir = joinpath(output_dir, "empty")
    mkpath(empty_dir)
    @test isnothing(CAL.load_latest_ekp(empty_dir))

    # Test loading from directory with completed iterations
    # We already have an EKP struct saved from earlier tests
    latest_ekp = CAL.load_latest_ekp(output_dir)
    @test !isnothing(latest_ekp)
    @test latest_ekp isa EKP.EnsembleKalmanProcess
    # Compare with the known EKP struct
    for prop in propertynames(latest_ekp)
        prop in [:u, :accelerator, :localizer] && continue
        @test getproperty(latest_ekp, prop) ==
              getproperty(eki_with_kwargs, prop)
    end
    @test latest_ekp.u[1].data == eki_with_kwargs.u[1].data
end

override_file =
    joinpath(output_dir, "iteration_000", "member_001", "parameters.toml")
td = CP.create_toml_dict(FT; override_file)
params = CP.get_parameter_values(td, param_names)

@testset "Initialized parameter values" begin
    # This checks for random seed as well
    @test params.one ≈ 3.416574531266089
    @test params.two ≈ 4.614950047803855
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
    @test params.one ≈ 3.1313341622997677
    @test params.two ≈ 5.063035177034372
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

@testset "G ensemble matrix" begin
    g_ens_mat = CAL.g_ens_matrix(user_constructed_eki)
    @test size(g_ens_mat) == (1, 10)
    @test g_ens_mat isa Matrix{Float64}
end

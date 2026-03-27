using Test

import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP

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
    Γ = ones(1, 1)
    y = [1.0]
    prior_u1 = EKP.constrained_gaussian("amplitude", 2, 1, 0, Inf)
    prior = EKP.combine_distributions([prior_u1])
    N_ensemble = 10
    initial_ensemble = EKP.construct_initial_ensemble(prior, N_ensemble)
    ekp = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, EKP.Inversion())
    g_ens_mat = CAL.g_ens_matrix(ekp)
    @test size(g_ens_mat) == (1, 10)
    @test g_ens_mat isa Matrix{Float64}
end

# TODO: Add tests for get_metadata_for_nth_iteration and
# get_observations_for_nth_iteration

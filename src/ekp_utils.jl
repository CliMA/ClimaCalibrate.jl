module EKPUtils

import LinearAlgebra
import Statistics
import EnsembleKalmanProcesses as EKP

export minibatcher_over_samples,
    observation_series_from_samples,
    g_ens_matrix,
    get_metadata_for_nth_iteration,
    get_observations_for_nth_iteration,
    residual

_fixed_minibatcher_indices(n_batches, batch_size) =
    [collect(((i - 1) * batch_size + 1):(i * batch_size)) for i in 1:n_batches]

"""
    minibatcher_over_samples(n_samples, batch_size)

Create a `FixedMinibatcher` that divides `n_samples` into batches of size `batch_size`.

If `n_samples` is not divisible by `batch_size`, the remaining samples will be dropped.
"""
function minibatcher_over_samples(n_samples::Int, batch_size::Int)
    n_samples <= 0 &&
        throw(ArgumentError("Number of samples ($n_samples) must be positive"))
    batch_size <= 0 &&
        throw(ArgumentError("Batch size ($batch_size) must be positive"))
    n_batches = div(n_samples, batch_size)
    remainder = n_samples % batch_size
    if remainder > 0
        @warn "Number of samples $n_samples not divisible by batch size $batch_size. The last $(remainder) samples will be dropped."
    end
    given_batches = _fixed_minibatcher_indices(n_batches, batch_size)
    return EKP.FixedMinibatcher(given_batches)
end

"""
    minibatcher_over_samples(samples, batch_size)

Create a `FixedMinibatcher` that divides a vector of samples into batches of size `batch_size`.

If the number of samples is not divisible by `batch_size`, the remaining samples will be dropped.
"""
function minibatcher_over_samples(samples::Vector, batch_size::Int)
    return minibatcher_over_samples(length(samples), batch_size)
end

"""
    observation_series_from_samples(samples, batch_size, names = nothing)

Create an `EKP.ObservationSeries` from a vector of `EKP.Observation` samples.

If the number of samples is not divisible by `batch_size`, the remaining samples will be dropped.
"""
function observation_series_from_samples(
    samples::Vector{<:EKP.Observation},
    batch_size,
    names = nothing,
)
    if !isnothing(names) && length(names) != length(samples)
        throw(
            ArgumentError(
                "Number of names ($(length(names))) must match number of samples ($(length(samples)))",
            ),
        )
    end
    minibatcher = minibatcher_over_samples(samples, batch_size)
    names = isnothing(names) ? string.(1:length(samples)) : names
    return EKP.ObservationSeries(samples, minibatcher, names)
end

"""
    g_ens_matrix(eki::EKP.EnsembleKalmanProcess{FT}) where {FT <: AbstractFloat}

Construct a G ensemble matrix of type `FT` with all `NaN`s for the current
iteration.
"""
function g_ens_matrix(
    eki::EKP.EnsembleKalmanProcess{FT},
) where {FT <: AbstractFloat}
    obs = EKP.get_obs(eki)
    single_obs_len = sum(length(obs))
    ensemble_size = EKP.get_N_ens(eki)
    g_ens_matrix = Array{FT}(undef, single_obs_len, ensemble_size)
    fill!(g_ens_matrix, NaN)
    return g_ens_matrix
end

"""
    get_metadata_for_nth_iteration(obs_series::EKP.ObservationSeries, N)

For the `N`th iteration, return a vector of the metadata of the observation(s)
being processed.
"""
function get_metadata_for_nth_iteration(obs_series::EKP.ObservationSeries, N)
    minibatch_obs = get_observations_for_nth_iteration(obs_series, N)
    metadata_vec = map(obs -> EKP.get_metadata(obs), minibatch_obs)
    return vcat(metadata_vec...)
end

"""
    get_observations_for_nth_iteration(obs_series::EKP.ObservationSeries, N)

For the `N`th iteration, return a vector of the observation(s) being processed.
"""
function get_observations_for_nth_iteration(
    obs_series::EKP.ObservationSeries,
    N,
)
    num_epochs = EKP.get_length_epoch(obs_series)
    # EKP.get_minibatch fails with N > num_epochs, so we use mod1 to go back to
    # the first epoch which seems consistent with what EKP does
    minibatch_indices = EKP.get_minibatch(obs_series, mod1(N, num_epochs))
    minibatch_obs = EKP.get_observations(obs_series)[minibatch_indices]
    return minibatch_obs
end

"""
    residual(
        ekp::EKP.EnsembleKalmanProcess;
        N = EKP.get_N_iterations(ekp),
        ignore_nan = true,
    )

Return the normalized residual `(mean(G) - obs) / σ` for the `N`th iteration,
where `σ` is the square root of the diagonal of the observation noise
covariance.

If `ignore_nan = true`, then the mean of the G ensemble at each index is
computed over the ensemble members that are not `NaN`.

!!! note "Sign convention" 
    The sign of the residual is flipped compared to
    [`ClimaCalibrate.analyze_residual`](@ref), which uses `obs - mean(G)`.

See the [Visualization](@ref) documentation for how to interpret the residual.
"""
function residual(
    ekp::EKP.EnsembleKalmanProcess;
    N = EKP.get_N_iterations(ekp),
    ignore_nan = true,
)
    obs_series = EKP.get_observation_series(ekp)
    obs = get_observations_for_nth_iteration(obs_series, N)
    stacked_obs = mapreduce(EKP.get_obs, vcat, obs)
    stacked_noise_var = mapreduce(vcat, obs) do o
        covs = EKP.get_obs_noise_cov(o, build = false)
        mapreduce(_diag, vcat, covs)
    end
    g_mean = if ignore_nan
        # Mean of an empty vector is NaN, so indices where all ensemble
        # members are NaN stay NaN
        map(row -> Statistics.mean(filter(!isnan, row)), eachrow(EKP.get_g(ekp, N)))
    else
        EKP.get_g_mean(ekp, N)
    end
    return (g_mean .- stacked_obs) ./ sqrt.(stacked_noise_var)
end

"""
    _diag(A)

Return the diagonal of the matrix `A` as a vector without materializing the
entire matrix.
"""
_diag(A::AbstractMatrix) = LinearAlgebra.diag(A)
# diag(U * Diagonal(S) * Vt)[i] is the sum of U[i, j] * S[j] * Vt[j, i] over j
_diag(A::LinearAlgebra.SVD) = vec(sum(A.U .* transpose(A.S .* A.Vt), dims = 2))
_diag(A::EKP.SVDplusD) = _diag(EKP.get_svd_cov(A)) .+ _diag(EKP.get_diag_cov(A))

end

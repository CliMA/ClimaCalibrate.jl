"""
    EKPUtils

A collection of utility functions for simplifying common operations in
EnsembleKalmanProcesses.jl.
"""
module EKPUtils

import EnsembleKalmanProcesses as EKP

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

Construct an uninitialized G ensemble matrix of type `FT` for the current
iteration.
"""
function g_ens_matrix(
    eki::EKP.EnsembleKalmanProcess{FT},
) where {FT <: AbstractFloat}
    obs = EKP.get_obs(eki)
    single_obs_len = sum(length(obs))
    ensemble_size = EKP.get_N_ens(eki)
    return Array{FT}(undef, single_obs_len, ensemble_size)
end

# TODO: Move get_observations_for_nth_iteration and
# get_metadata_for_nth_iteration to this module (it is a little bit more
# challenging because the tests use ObservationRecipe)

end

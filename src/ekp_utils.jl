"""
    ClimaCalibrate.EKPUtils

Helpers for working with EnsembleKalmanProcesses.jl objects that do not
depend on the rest of ClimaCalibrate.

Covers building minibatchers and `ObservationSeries` from a vector of samples
([`minibatcher_over_samples`](@ref), [`observation_series_from_samples`](@ref)),
allocating a G ensemble matrix of the right shape ([`g_ens_matrix`](@ref)), and
looking up which observations an iteration is being scored against
([`get_observations_for_nth_iteration`](@ref)).
"""
module EKPUtils

import EnsembleKalmanProcesses as EKP

export minibatcher_over_samples,
    observation_series_from_samples,
    g_ens_matrix,
    get_metadata_for_nth_iteration,
    get_observations_for_nth_iteration

_fixed_minibatcher_indices(n_batches, batch_size) =
    [collect(((i - 1) * batch_size + 1):(i * batch_size)) for i in 1:n_batches]

"""
    minibatcher_over_samples(n_samples, batch_size)

Create a `FixedMinibatcher` that divides `n_samples` into batches of size
`batch_size`.

If `n_samples` is not divisible by `batch_size`, the remaining samples are
dropped and a warning is emitted.

# Examples
```julia
minibatcher = ClimaCalibrate.minibatcher_over_samples(10, 5)
```

See also [`observation_series_from_samples`](@ref).
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

If the number of samples is not divisible by `batch_size`, the remaining samples
are dropped.

# Examples
```julia
obs_series = ClimaCalibrate.observation_series_from_samples(observations, 5)
```

See also [`minibatcher_over_samples`](@ref).
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

Construct a G ensemble matrix of type `FT`, filled with `NaN`, sized for the
current iteration's observation and ensemble.

Starting from `NaN` means a member whose forward model failed is left as `NaN`,
which is how EKP is told to ignore it.

# Examples
```julia
G_ensemble = ClimaCalibrate.g_ens_matrix(ekp)
```
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
    _repeats_each_epoch(minibatcher)

Return `true` if `minibatcher` gives every epoch the same minibatches, so that
the schedule of an iteration that has not run yet is known in advance.
"""
_repeats_each_epoch(minibatcher) = false

_repeats_each_epoch(minibatcher::EKP.FixedMinibatcher) =
    EKP.get_method(minibatcher) == "order"

"""
    get_observations_for_nth_iteration(obs_series::EKP.ObservationSeries, N)

For the `N`th iteration, return a vector of the observation(s) being processed.
"""
function get_observations_for_nth_iteration(
    obs_series::EKP.ObservationSeries,
    N,
)
    # A minibatcher that shuffles draws a different minibatch in each epoch, so
    # the schedule for iteration `N` is the one EKP stored when it ran that
    # iteration. EKP stores it up to the last iteration that has run
    len_epoch = EKP.get_length_epoch(obs_series)
    n_scheduled = length(EKP.get_minibatches(obs_series)) * len_epoch
    if N > n_scheduled
        _repeats_each_epoch(EKP.get_minibatcher(obs_series)) || error(
            "Iteration $N has no minibatch: this `ObservationSeries` has been \
            scheduled through iteration $n_scheduled, and its minibatcher \
            draws a new order for each epoch. Pass an iteration that has \
            already run.",
        )
        N = mod1(N, len_epoch)
    end
    minibatch_indices = EKP.get_minibatch(obs_series, N)
    minibatch_obs = EKP.get_observations(obs_series)[minibatch_indices]
    return minibatch_obs
end

end

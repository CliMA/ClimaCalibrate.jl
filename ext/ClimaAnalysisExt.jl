module ClimaAnalysisExt

import ClimaAnalysis
import ClimaAnalysis: OutputVar

import ClimaCalibrate.Pipeline as Pipeline
import ClimaCalibrate.Pipeline: AbstractSample, AbstractCovariance
import ClimaCalibrate.Pipeline: SeasonalSample, NoiseCovariance, SVDCovariance

import EnsembleKalmanProcesses as EKP

import Dates
import Statistics: mean, var
import NaNStatistics: nanmean, nanvar

import LinearAlgebra: Diagonal

include("climaanalysis_helper.jl")
include("utils.jl")

"""
    sample(samp::SeasonalSample, var::OutputVar, start_date, end_date)

Window the data between `start_date` and `end_date`, compute seasonal averages
of `var`, and flatten the data.

The dates `start_date` and `end_date` must be in `ClimaAnalysis.dates(var)`.

There is no error checking on the dates passed in. As such, the user should
check that the dates represent the start and end of the seasons.
"""
function Pipeline.sample(
    samp::SeasonalSample,
    var::OutputVar,
    start_date,
    end_date;
    metadata = false,
)
    start_date isa AbstractString && (start_date = Dates.DateTime(start_date))
    end_date isa AbstractString && (end_date = Dates.DateTime(end_date))

    # TODO: Check if the start and end dates make sense!
    # Round to the nearest start and end of the season and print a warning if it is not
    # This may be annoying if the seasons are not what we expected

    # Window in ClimaAnalysis round to the nearest number, so we check the dates exist
    (
        start_date in ClimaAnalysis.dates(var) &&
        end_date in ClimaAnalysis.dates(var)
    ) || error(
        "$start_date and $end_date are not found in var. The available dates are $(ClimaAnalysis.dates(var))",
    )
    var = ClimaAnalysis.window(var, "time", left = start_date, right = end_date)
    return Pipeline.sample(samp, var, metadata = metadata)
end

"""
    sample(samp::SeasonalSample, var::OutputVar)

Compute seasonal averages of `var`, and flatten the data.
"""
function Pipeline.sample(
    sample_config::SeasonalSample,
    var::OutputVar;
    metadata = false,
)
    var = _process_sample(sample_config, var)
    flat_var = ClimaAnalysis.flatten(
        var,
        ignore_nan = sample_config.ignore_nan_in_sample,
    )
    if metadata
        return (flat_var.data, flat_var.metadata)
    else
        return flat_var.data
    end
end

"""
    _process_sample(sample_config::SeasonalSample, var::OutputVar)

Make a sample by taking seasonal averages of `var`.
"""
function _process_sample(sample_config::SeasonalSample, var::OutputVar)
    return ClimaAnalysis.average_season_across_time(
        var,
        ignore_nan = sample_config.ignore_nan_in_average,
    )
end

"""
    covariance(covar_config::AbstractCovariance,
               sample_config::AbstractSample,
               var::OutputVar)

Compute the covariance from `var`.
"""
function Pipeline.covariance(
    covar_config::AbstractCovariance,
    sample_config::AbstractSample,
    var::OutputVar,
)
    # If no dates are supplied, then construct the full covariance matrix given `var`
    start_date = first(ClimaAnalysis.dates(var))
    end_date = last(ClimaAnalysis.dates(var))
    return Pipeline.covariance(
        covar_config,
        sample_config,
        var,
        start_date,
        end_date,
    )
end

"""
    covariance(covar_config::NoiseCovariance,
               sample_config::SeasonalSample,
               var::OutputVar,
               start_date,
               end_date)

Compute the noise covariance matrix of seasonal averages from `var` that is
appropriate for a sample of seasonal averages from seasons between `start_date`
and `end_date`.
"""
function Pipeline.covariance(
    covar_config::NoiseCovariance,
    sample_config::SeasonalSample,
    var::OutputVar,
    start_date,
    end_date,
)
    # Convert dates to Dates.DateTime if they are strings
    # Note that we do not check if these dates are in the time dimension of var
    # because we are dealing with seasons
    start_date isa AbstractString && (start_date = Dates.DateTime(start_date))
    end_date isa AbstractString && (end_date = Dates.DateTime(end_date))

    # Compute seasonal averages
    seasonal_average_across_time_var = _process_sample(sample_config, var)

    # Variance of the seasons
    group_by_season(time_dim) = split_by_season(
        time_dim,
        var.attributes["start_date"],
        seasons = ("MAM", "JJA", "SON", "DJF"),
    )
    reduce_by = covar_config.ignore_nan ? nanvar : var
    seasonal_variance_var = group_and_reduce_by(
        seasonal_average_across_time_var,
        "time",
        group_by_season,
        reduce_by,
    )

    # Add model error scale
    if !iszero(covar_config.model_error_scale)
        reduce_by = covar_config.ignore_nan ? nanmean : mean
        seaonal_average_var = group_and_reduce_by(
            seasonal_average_across_time_var,
            "time",
            group_by_season,
            reduce_by,
        )
        seasonal_variance_var.data += seaonal_average_var.data
    end

    # Construct full covariance matrix
    # First, find the seasons covered by the dates
    seasons = find_seasons(start_date, end_date)
    seasons_to_int = Dict("MAM" => 1, "JJA" => 2, "SON" => 3, "DJF" => 4)

    # Then, use group_and_reduce_by to duplicate time slices
    reduce_by_identity(A; dims) = A
    group_by_slice(dim) = [[dim[seasons_to_int[season]]] for season in seasons]
    full_covariance_var = group_and_reduce_by(
        seasonal_variance_var,
        "time",
        group_by_slice,
        reduce_by_identity,
    )

    # TODO: maybe add an option for ignoring nans when flattening?
    # This would be borrowed from sample_config
    diag_cov = ClimaAnalysis.flatten(full_covariance_var).data
    !iszero(covar_config.regularization) &&
        (diag_cov .+= covar_config.regularization)
    return Diagonal(diag_cov)
end

"""
    covariance(covar_config::SVDCovariance,
               sample_config::SeasonalSample,
               var::OutputVar,
               start_date,
               end_date)
# TODO: I don't know what this function is doing
Compute the SVD covariance matrix...
"""
function Pipeline.covariance(
    covar_config::SVDCovariance,
    sample_config::SeasonalSample,
    var::OutputVar,
    start_date,
    end_date,
)
    # TODO: Ask Nat about this function since I don't really get how this function work
    # TODO: Add nan handling somewhere
    # Right now, this is just copied and paste from Nat's PR

    # Reshape `var` into a matrix `var_mat` where each column is a time slice
    var_vec = map(times(var)) do t
        flatten(slice(var, time = t)).data
    end
    var_mat = hcat(var_vec...)

    gamma_low_rank = EKP.tsvd_cov_from_samples(var_mat)

    gamma_diag = (model_error_scale * flatten(average_time(var)).data) .^ 2
    if !iszero(regularization)
        gamma_diag += regularization * EKP.I
    end

    return EKP.SVDplusD(gamma_low_rank, Diagonal(gamma_diag))
end

"""
    observation(sample_config::AbstractSample
                covar_config::AbstractCovariance,
                var::OutputVar)

Return an `EnsembleKalmanProcesses.Observation` of sample determined by `sample_config`
and covariance matrix determined by `covar_config`.

# TODO: I don't know what these functions are called
Metadata is included in the observation and can be used to reconstruct vector into
`OutputVar`. See @ref
"""
function Pipeline.observation(
    sample_config::AbstractSample,
    covar_config::AbstractCovariance,
    var::OutputVar;
    name = nothing,
)
    return Pipeline.observation(
        sample_config,
        covar_config,
        var,
        first(dates(var)),
        last(dates(var)),
        name = name,
    )
end

"""
    observation(sample_config::AbstractSample,
                covar_config::AbstractCovariance,
                var::OutputVar,
                start_date,
                end_date;
                name = nothing)


"""
function Pipeline.observation(
    sample_config::AbstractSample,
    covar_config::AbstractCovariance,
    var::OutputVar,
    start_date,
    end_date;
    name = nothing,
)
    flattened_data, metadata = Pipeline.sample(
        sample_config,
        var,
        start_date,
        end_date,
        metadata = true,
    )
    covar = Pipeline.covariance(
        covar_config,
        sample_config,
        var,
        start_date,
        end_date,
    )
    isnothing(name) && (name = get(var.attributes, "short_name", ""))
    # TODO: Figure out what a combined observation look like for metadata
    return EKP.Observation(
        Dict(
            "samples" => flattened_data,
            "covariances" => covar,
            "names" => name,
            "metadata" => metadata,
        ),
    )
end

# TODO: Come up with better names

"""
    unflatten_sample_from_obs(obs, name)

Unflatten the sample in `obs` corresponding to `name` into a `OutputVar`.
"""
function Pipeline.unflatten_sample_from_obs(obs, name)
    obs_idx = _find_idx_from_name(obs, name)
    metadata = _find_metadata_from_name(obs, obs_idx)
    sample = EKP.get_samples(obs)[obs_idx]
    return ClimaAnalysis.unflatten(metadata, sample)
end

# TODO: For documentation for this function, add a warning that the
# result may not make sense depending on blah

"""
    unflatten_cov_from_obs(obs, name)

Unflatten the diagonal of the covariance matrix corresponding to `name` in `obs`
into a `OutputVar`.
"""
function Pipeline.unflatten_cov_from_obs(obs, name)
    obs_idx = _find_idx_from_name(obs, name)
    metadata = _find_metadata_from_name(obs, obs_idx)
    cov = EKP.get_covs(obs)[obs_idx]

    # TODO: Write this function :(
    # This function need to dispatch on different types for the covariance matrix
    diag = diag_from_cov(cov)
    return ClimaAnalysis.unflatten(metadata, diag)
end

# TODO: Ask Nat about the functionality of this function
# since I am not sure what context it will be used in

"""
    unflatten_vec_from_obs(obs, vec, name)

Unflatten `vec` into a `OutputVar` using the metadata from the observation with
the name `name` in `obs`.
"""
function Pipeline.unflatten_vec_from_obs(obs, vec, name)
    obs_idx = _find_idx_from_name(obs, name)
    metadata = _find_metadata_from_name(obs, obs_idx)
    return ClimaAnalysis.unflatten(metadata, vec)
end

"""
    _find_idx_from_name(obs, name)

Find the index of the observation corresponding to `name` in `obs`.
"""
function _find_idx_from_name(obs, name)
    # TODO: What if an observation doesn't have a name and you combine the observations?
    # TODO: What if an observation doesn't have metadata and you combine
    # the observations?
    idx_vec = findall(x -> x == name, obs.names)

    # Check name is unique
    if length(idx_vec) == 0
        error("There are no observation with the name $name")
    elseif length(idx_vec) > 1
        error(
            "There are multiple observations ($(length(idx_vec))) with that name",
        )
    end

    return first(idx_vec)
end

"""
"""
function _find_metadata_from_name(obs, idx)
    # The metadata is either a vector because it is formed by combining
    # observations using combine_observations or a ClimaAnalysis.Var.metadata
    # because it is a single observation
    metadata = if isnothing(EKP.get_metadata(obs))
        error("No metadata is found for this observation")
    elseif EKP.get_metadata(obs) isa ClimaAnalysis.Var.Metadata
        obs.metadata
    else
        # Assume metdata is an iterable
        obs.metadata[idx]
    end
    return metadata
end

end

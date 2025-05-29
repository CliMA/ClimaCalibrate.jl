module ClimaAnalysisExt

import ClimaAnalysis
import ClimaAnalysis: OutputVar

import ClimaCalibrate.Pipeline as Pipeline
import ClimaCalibrate.Pipeline: AbstractSample, AbstractCovariance
import ClimaCalibrate.Pipeline:
    SeasonalSample, NoiseCovariance, SVDPlusDCovariance

import EnsembleKalmanProcesses as EKP

import Dates
import Statistics: mean, var
import NaNStatistics: nanmean, nanvar

import LinearAlgebra: Diagonal

include("climaanalysis_helper.jl")
include("utils.jl")

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
    (start_date in dates(var) && end_date in dates(var)) || error(
        "$start_date and $end_date are not found in var. The available dates are $(dates(var))",
    )
    var = window(var, "time", left = start_date, right = end_date)
    return sample(samp, var, metadata = metadata)
end

# TODO: Write a test to make sure that this is idempotent (if the data is already seaonal averages for example)
function Pipeline.sample(
    samp_config::SeasonalSample,
    var::OutputVar;
    metadata = false,
)
    var = _process_sample(samp_config, var)
    flat_var = ClimaAnalysis.flatten(
        var,
        ignore_nan = samp_config.ignore_nan_in_sample,
    )
    if metadata
        return (flat_var.data, flat_var.metadata)
    else
        return flat_var.data
    end
end

function _process_sample(samp_config::SeasonalSample, var::OutputVar)
    return ClimaAnalysis.average_season_across_time(
        var,
        ignore_nan = samp_config.ignore_nan_in_average,
    )
end

function Pipeline.covariance(
    covar_config::AbstractCovariance,
    samp_config::AbstractSample,
    var::OutputVar,
)
    # If no dates are supplied, then construct the full covariance matrix given `var`
    start_date = first(ClimaAnalysis.dates(var))
    end_date = last(ClimaAnalysis.dates(var))
    return Pipeline.covariance(
        covar_config,
        samp_config,
        var,
        start_date,
        end_date,
    )
end

function Pipeline.covariance(
    covar_config::NoiseCovariance,
    samp_config::SeasonalSample,
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
    seasonal_average_across_time_var = _process_sample(samp_config, var)

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


    diag_cov = ClimaAnalysis.flatten(full_covariance_var).data
    !iszero(covar_config.regularization) &&
        (diag_cov .+= covar_config.regularization)
    return Diagonal(diag_cov)
end

function Pipeline.covariance(
    covar_config::SVDPlusDCovariance,
    samp_config::SeasonalSample,
    var::OutputVar,
    start_date,
    end_date,
)
    # TODO: Ask Nat about this function since I don't really get it
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



function Pipeline.observation(
    samp_config::AbstractSample,
    covar_config::AbstractCovariance,
    var::OutputVar;
    name = nothing,
)
    return Pipeline.observation(
        samp_config,
        covar_config,
        var,
        first(dates(var)),
        last(dates(var)),
        name = name,
    )
end

function Pipeline.observation(
    samp_config::AbstractSample,
    covar_config::AbstractCovariance,
    var::OutputVar,
    start_date,
    end_date;
    name = nothing,
)
    flattened_data, metadata =
        Pipeline.sample(samp_config, var, start_date, end_date, metadata = true)
    covar = Pipeline.covariance(
        covar_config,
        samp_config,
        var,
        start_date,
        end_date,
    )
    isnothing(name) && (name = get(var.attributes, "short_name", ""))
    # TODO: Figure out what a combined observation look like for metadata
    return EKP.Observation(flattened_data, covar, name, metadata)
end

function Pipeline.reconstruct_var_from_obs()
    error("Not yet implemented!")
end

# Reconstruct from g_ensemble, covariance

end

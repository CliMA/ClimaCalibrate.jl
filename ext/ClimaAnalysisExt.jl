module ClimaAnalysisExt

import ClimaAnalysis
import ClimaAnalysis: OutputVar

import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.ObservationRecipe: AbstractCovarianceEstimator
import ClimaCalibrate.ObservationRecipe:
    SeasonalDiagonalCovariance, SVDplusDCovariance

import EnsembleKalmanProcesses as EKP

import Dates
import Statistics
import Statistics: mean
import NaNStatistics: nanmean, nanvar

import LinearAlgebra: Diagonal, I

include("utils.jl")

"""
    covariance(covar_estimator::SeasonalDiagonalCovariance,
               vars::Union{OutputVar, Iterable{OutputVar}},
               start_date,
               end_date)

Compute the noise covariance matrix of seasonal averages from `var` that is
appropriate for a sample of seasonal averages across time for seasons between
`start_date` and `end_date`.
"""
function ObservationRecipe.covariance(
    covar_estimator::SeasonalDiagonalCovariance,
    vars,
    start_date,
    end_date,
)
    # Convert dates to Dates.DateTime if they are strings
    start_date = Dates.DateTime(start_date)
    end_date = Dates.DateTime(end_date)

    vars = _vars_to_iterable(vars)

    for var in vars
        _check_time_dim(var)
    end

    # Check if dates for start_date and end_date are in var
    _check_dates_in_var(vars, start_date, end_date)

    start_date <= end_date ||
        error("$start_date should not be later than $end_date")

    diagonals = map(vars) do var
        # Var is an OutputVar whose time slices are seasonal quantities
        seasonal_statistics_across_time_var = var

        # Throw warning if there are multiple time slices in a single season
        for season in ClimaAnalysis.Utils.split_by_season_across_time(
            ClimaAnalysis.dates(var),
        )
            if length(season) > 1
                short_name = get(var.attributes, "short_name", nothing)
                @warn(
                    "Detected multiple times for a season ($season) in the OutputVar with the short name $short_name"
                )
            end
        end

        # Variance of the seasons
        group_by_season(time_dim) = split_by_season_from_seconds(
            time_dim,
            var.attributes["start_date"],
            seasons = ("MAM", "JJA", "SON", "DJF"),
        )
        reduce_by = covar_estimator.ignore_nan ? nanvar : Statistics.var
        seasonal_variance_var = group_and_reduce_by(
            seasonal_statistics_across_time_var,
            "time",
            group_by_season,
            reduce_by,
        )

        # Add model error scale
        if !iszero(covar_estimator.model_error_scale)
            reduce_by = covar_estimator.ignore_nan ? nanmean : mean
            seasonal_average_var = group_and_reduce_by(
                seasonal_statistics_across_time_var,
                "time",
                group_by_season,
                reduce_by,
            )
            seasonal_variance_var.data .+= covar_estimator.float_type(
                (
                    covar_estimator.model_error_scale .*
                    seasonal_average_var.data
                ) .^ 2,
            )
        end

        # Construct full covariance matrix
        # First, find the seasons covered by the dates
        seasons = find_seasons(start_date, end_date)
        seasons_to_int =
            Dict("MAM" => 1, "JJA" => 2, "SON" => 3, "DJF" => 4)

        # Then, use group_and_reduce_by to duplicate time slices
        reduce_by_identity(A; dims) = A
        group_by_slice(dim) =
            [[dim[seasons_to_int[season]]] for season in seasons]
        full_covariance_var = group_and_reduce_by(
            seasonal_variance_var,
            "time",
            group_by_slice,
            reduce_by_identity,
        )

        # Add regularization
        diag_cov = ClimaAnalysis.flatten(full_covariance_var).data
        !iszero(covar_estimator.regularization) && (
            diag_cov .+= covar_estimator.float_type.(
                covar_estimator.regularization,
            )
        )
        diag_cov
    end
    return Diagonal(vcat(diagonals...))
end

"""
    covariance(covar_estimator::SVDplusDCovariance,
               vars::Union{OutputVar, Iterable{OutputVar}},
               start_date,
               end_date)

Compute the `EKP.SVDplusD` covariance matrix appropriate for a sample with times
between `start_date` and `end_date`.
"""
function ObservationRecipe.covariance(
    covar_estimator::SVDplusDCovariance,
    vars,
    start_date,
    end_date,
)
    vars = _vars_to_iterable(vars)

    # Convert dates to Dates.DateTime if they are strings
    start_date = Dates.DateTime(start_date)
    end_date = Dates.DateTime(end_date)

    start_date <= end_date ||
        error("$start_date should not be later than $end_date")

    # Check start and end dates provided are valid
    sample_dates = covar_estimator.sample_dates
    (start_date, end_date) in sample_dates ||
        error("$start_date and $end_date are not in $(sample_dates)")

    # Check all dates in sample_dates are valid
    for (sample_start_date, sample_end_date) in sample_dates
        _check_dates_in_var(vars, sample_start_date, sample_end_date)
    end

    # Form stacked samples
    stacked_samples = _stacked_samples(vars, sample_dates)

    # Check samples are all the same size
    all(x -> x == length(stacked_samples[1]), length.(stacked_samples)) ||
        error(
            "Length of all the samples are not the same. Try checking sample_dates and the dates of the OutputVars passed in",
        )

    # Compute SVD of covariance matrix
    stacked_sample_matrix =
        covar_estimator.float_type.(hcat(stacked_samples...))
    gamma_low_rank = EKP.tsvd_cov_from_samples(stacked_sample_matrix)

    # Add model error scale. This may not make sense if the samples do not
    # represent a single year. For example, if the stacked samples are seasonal
    # averages over two years, then this quantity is the mean of seasonal
    # averages spanned over two years, where the first DJF is the mean of every
    # other DJF and the second DJF is the mean of every other DJF.
    model_error_scale =
        (
            covar_estimator.model_error_scale .*
            mean(stacked_sample_matrix, dims = 2)
        ) .^ 2
    model_error_scale =
        covar_estimator.float_type.(Diagonal(vec(model_error_scale)))
    # Add regularization
    regularization =
        covar_estimator.float_type(covar_estimator.regularization) * I

    return EKP.SVDplusD(gamma_low_rank, model_error_scale + regularization)
end

"""
    observation(covar_estimator::AbstractCovarianceEstimator,
                vars,
                start_date,
                end_date;
                name = nothing)

Return an `EKP.Observation` with a sample between the dates `start_date` and
`end_date`, a covariance matrix defined by `covar_estimator`, `name` determined
from the short names of `vars`, and metadata.

!!! note
    Metadata is only added with EnsembleKalmanProcesses v2.4.3 or later.
"""
function ObservationRecipe.observation(
    covar_estimator::AbstractCovarianceEstimator,
    vars,
    start_date,
    end_date;
    name = nothing,
)
    # Convert dates to Dates.DateTime if they are strings
    start_date = Dates.DateTime(start_date)
    end_date = Dates.DateTime(end_date)

    start_date <= end_date ||
        error("$start_date should not be later than $end_date")

    vars = _vars_to_iterable(vars)

    # Check if all dates in vars are valid
    _check_dates_in_var(vars, start_date, end_date)

    # Get the flattened sample and metadata
    windowed_vars =
        ClimaAnalysis.window.(vars, "time", left = start_date, right = end_date)
    flat_vars = map(var -> ClimaAnalysis.flatten(var), windowed_vars)
    stacked_sample = vcat((flat_var.data for flat_var in flat_vars)...)
    metadata = [(flat_var.metadata for flat_var in flat_vars)...]
    covar = ObservationRecipe.covariance(
        covar_estimator,
        vars,
        start_date,
        end_date,
    )

    # Concatenate names with separator of semicolons
    isnothing(name) && (
        name = join(
            [get(var.attributes, "short_name", nothing) for var in vars],
            ";",
        )
    )
    return EKP.Observation(
        Dict(
            "samples" => stacked_sample,
            "covariances" => covar,
            "names" => name,
            "metadata" => metadata,
        ),
    )
end

"""
    _stacked_samples(vars::Iterable{OutputVar}, sample_dates)

Make a vector of stacked samples from multiple `OutputVar`s and an iterable of
2-iterable of dates.

Each sample corresponds to a single minibatch and the sample is stacked because
it may contains multiple variables.
"""
function _stacked_samples(vars, sample_dates)
    return map(sample_dates) do (sample_start_date, sample_end_date)
        vcat(
            (
                ClimaAnalysis.flatten(
                    ClimaAnalysis.window(
                        var,
                        "time",
                        left = sample_start_date,
                        right = sample_end_date,
                    ),
                ).data for var in vars
            )...,
        )
    end
end

"""
    group_and_reduce_by(var::OutputVar, group_by, reduce_by)

Group the dimension `dim_name` using `group_by` and apply the reduction
`reduce_by` along the dimension.

The first element in each group is used as the dimension of the resulting `OutputVar`.

Only the short name, long name, units, and start date are kept. All other attributes are
discarded in the process.

The function `group_by` takes in a vector of values for a dimension and returns a vector of
vectors of values of the dimension. The function `reduce_by` must be of the form `f(A;
dims)`, where `A` is a slice of `var.data` and `dims` is the index of `A` to reduce against
and return the reduction over the `dims` dimension.

Note `group_by` does not need to partition the values of the dimension. For example, the
`group_by` function can be `group_by(dim) = [[first(dim)]]`, which return the first slice of
the `OutputVar`.
"""
function group_and_reduce_by(var::OutputVar, dim_name, group_by, reduce_by)
    dim_name_in_var =
        ClimaAnalysis.Var.find_corresponding_dim_name_in_var(dim_name, var)
    dim_idx = var.dim2index[dim_name_in_var]

    # Group by
    dim_vals_groups = group_by(var.dims[dim_name_in_var])
    dim_indices_groups =
        indexin.(dim_vals_groups, Ref(var.dims[dim_name_in_var]))
    data_groups = map(dim_indices_groups) do dim_indices
        index_tuple = ntuple(
            idx -> idx == dim_idx ? dim_indices : Colon(),
            ndims(var.data),
        )
        view(var.data, index_tuple...)
    end

    # Reduce by and concat
    ret_data = cat(reduce_by.(data_groups, dims = dim_idx)..., dims = dim_idx)

    # Get the elements for constructing the new dimension
    ret_dim_indices = [first(dim_indices) for dim_indices in dim_indices_groups]

    # New dimension to return
    dim = var.dims[dim_name_in_var][ret_dim_indices]

    # Make new dimensions for OutputVar
    ret_dims = deepcopy(var.dims)
    ret_dims[dim_name_in_var] = dim

    # Keep short name, long name, units, and start_date and discard the rest
    keep_attribs = ("long_name", "short_name", "units", "start_date")
    ret_attribs = Dict(
        attrib => var.attributes[attrib] for
        attrib in keep_attribs if attrib in keys(var.attributes)
    )
    return ClimaAnalysis.remake(
        var,
        attributes = ret_attribs,
        data = ret_data,
        dims = ret_dims,
    )
end

end

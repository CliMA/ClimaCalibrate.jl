import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.ObservationRecipe: AbstractCovarianceEstimator
import ClimaCalibrate.ObservationRecipe:
    ScalarCovariance, SeasonalDiagonalCovariance, SVDplusDCovariance

"""
    covariance(covar_estimator::ScalarCovariance,
               vars::Union{OutputVar, Iterable{OutputVar}},
               start_date,
               end_date)

Compute the scalar covariance matrix.

Data from `vars` will not be used to compute the covariance matrix.
"""
function ObservationRecipe.covariance(
    covar_estimator::ScalarCovariance,
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

    start_date <= end_date || error("$start_date should earlier than $end_date")

    diagonals = map(vars) do var
        var = ClimaAnalysis.window(var, "time", left = start_date, right = end_date)
        flattened_data = ClimaAnalysis.flatten(var).data
        diag_cov = ones(eltype(flattened_data), size(flattened_data)...)
        diag_cov .*= covar_estimator.scalar

        if covar_estimator.use_latitude_weights
            flattened_lat_weights =
                ClimaAnalysis.flatten(
                    _lat_weights_var(
                        var,
                        min_cosd_lat = covar_estimator.min_cosd_lat,
                    ),
                ).data
            diag_cov .*= flattened_lat_weights
        end
        diag_cov
    end
    return Diagonal(vcat(diagonals...))
end

"""
    covariance(covar_estimator::SeasonalDiagonalCovariance,
               vars::Union{OutputVar, Iterable{OutputVar}},
               start_date,
               end_date)

Compute the noise covariance matrix of seasonal quantities from `var` that is
appropriate for a sample of seasonal quantities across time for seasons between
`start_date` and `end_date`.

The diagonal is computed from the variances of the seasonal quantities.
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

    start_date <= end_date || error("$start_date should earlier than $end_date")

    diagonals = map(vars) do var
        # Var is an OutputVar whose time slices are seasonal statistics
        seasonal_statistics_across_time_var = var

        # Throw an error if there are multiple time slices in a single season
        for season in ClimaAnalysis.Utils.split_by_season_across_time(
            ClimaAnalysis.dates(var),
        )
            if length(season) > 1
                short_name = get(var.attributes, "short_name", nothing)
                error(
                    "Detected multiple times for a season ($season) in the OutputVar with the short name $short_name. SeasonalDiagonalCovariance can only be used with OutputVars with a single time slice for each season",
                )
            end
        end

        # Variance of the seasons
        seasons = find_seasons(start_date, end_date)
        group_by_season(time_dim) = split_by_season_from_seconds(
            time_dim,
            var.attributes["start_date"],
            seasons = seasons,
        )
        reduce_by = covar_estimator.ignore_nan ? nanvar : Statistics.var

        # This can duplicate the same season which is what we want, because
        # a sample can be longer than a single year
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
            seasonal_variance_var.data .+=
                (
                    covar_estimator.model_error_scale .*
                    seasonal_average_var.data
                ) .^ 2
        end

        # Add regularization
        diag_cov = ClimaAnalysis.flatten(seasonal_variance_var).data
        !iszero(covar_estimator.regularization) &&
            (diag_cov .+= covar_estimator.regularization)

        # Add latitude weights
        if covar_estimator.use_latitude_weights
            flattened_lat_weights =
                ClimaAnalysis.flatten(
                    _lat_weights_var(
                        seasonal_variance_var,
                        min_cosd_lat = covar_estimator.min_cosd_lat,
                    ),
                ).data
            diag_cov .*= flattened_lat_weights
        end
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
        error("$start_date should be earlier than $end_date")

    # Check start and end dates provided are valid
    sample_date_ranges = covar_estimator.sample_date_ranges
    (start_date, end_date) in sample_date_ranges ||
        error("$start_date and $end_date are not in $(sample_date_ranges)")

    # Check all dates in sample_date_ranges are valid
    for (sample_start_date, sample_end_date) in sample_date_ranges
        _check_dates_in_var(vars, sample_start_date, sample_end_date)
    end

    # Form stacked samples
    stacked_samples = _stacked_samples(vars, sample_date_ranges)

    # Check samples are all the same size
    all(x -> x == length(first(stacked_samples)), length.(stacked_samples)) ||
        error(
            "Length of all the samples are not the same. Try checking sample_date_ranges and the dates of the OutputVars passed in. If there are `NaN`s in the data, check that the number of `NaN`s in each sample is the same.",
        )

    # Compute SVD of covariance matrix
    stacked_sample_matrix = hcat(stacked_samples...)

    covar_estimator.use_latitude_weights && _apply_lat_weights_to_samples!(
        stacked_sample_matrix,
        vars,
        start_date,
        end_date,
        min_cosd_lat = covar_estimator.min_cosd_lat,
    )

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
    model_error_scale = Diagonal(vec(model_error_scale))

    # Add regularization
    regularization = covar_estimator.regularization * I

    return EKP.SVDplusD(gamma_low_rank, model_error_scale + regularization)
end

"""
    _apply_lat_weights_to_samples!(stacked_sample_matrix,
                                   vars::Iterable{OutputVar},
                                   start_date,
                                   end_date,
                                   min_cosd_lat = 0.1)

Apply latitude weights to the matrix of samples.

The latitude weights applied is `1 / sqrt(max(cosd(lat), min_cosd_lat))` to each
column of the matrix.
"""
function _apply_lat_weights_to_samples!(
    stacked_sample_matrix,
    vars,
    start_date,
    end_date;
    min_cosd_lat = 0.1,
)
    flattened_lat_weights = (
        ClimaAnalysis.flatten(
            _lat_weights_var(
                ClimaAnalysis.window(
                    var,
                    "time",
                    left = start_date,
                    right = end_date,
                ),
                min_cosd_lat = min_cosd_lat,
            ),
        ).data for var in vars
    )
    flattened_lat_weights = sqrt.(vcat(flattened_lat_weights...))
    flattened_lat_weights =
        reshape(flattened_lat_weights, length(flattened_lat_weights), 1)
    # It is okay to find the latitude weights for a single column and apply it
    # to every other column, because the flattening of OutputVars should be the
    # same for each column
    stacked_sample_matrix .*= flattened_lat_weights
    return nothing
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

!!! note "Metadata"
    Metadata in `EKP.observation` is only added with versions of
    EnsembleKalmanProcesses later than v2.4.2.
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

    # Check if start date and end date exist in vars
    _check_dates_in_var(vars, start_date, end_date)

    # Check if dates are unique and short name exists
    for var in vars
        allunique(ClimaAnalysis.dates(var)) || @warn(
            "Dates in OutputVar with short name $(short_name(var)) are not unique. You will not be able to use GEnsembleBuilder",
        )
    end

    any(==(""), ClimaAnalysis.short_name.(vars)) && @warn(
        "There are OutputVar(s) with no short name. You will not be able to use GEnsembleBuilder"
    )

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

    # Concatenate names and separating them with a semicolon
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
    short_names(obs::EKP.Observation)

Get the short names of the variables from the metadata in the `EKP.Observation`.

If the short name is not available, then `nothing` is returned instead.
"""
function ObservationRecipe.short_names(obs::EKP.Observation)
    # Get the short names from the metadata rather than the name of the
    # observation, because the name can be changed by the user
    metadatas = EKP.get_metadata(obs)
    eltype(metadatas) <: ClimaAnalysis.Var.Metadata || error(
        "Getting the short names from an observation is only supported with metadata from ClimaAnalysis",
    )

    short_names = collect(
        get(metadata.attributes, "short_name", nothing) for
        metadata in metadatas
    )
    return short_names
end

"""
    seasonally_aligned_yearly_sample_date_ranges(var::OutputVar)

Generate sample dates that conform to a seasonally aligned year from
`dates(var)`.

A seasonally aligned year is defined to be from December to November of the
following year.

This function is useful for finding the sample dates of samples consisting of
all four seasons in a single year. For example, one can use this function to
find the `sample_date_ranges` when constructing `SVDplusDCovariance`.

!!! note "All four seasons in a year is not guaranteed"
    This function does not check whether the start and end dates of each sample
    contain all four seasons. A sample may be missing a season, especially at
    the beginning or end of the time series.
"""
function ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges(
    var::OutputVar,
)
    dates = ClimaAnalysis.dates(var)
    issorted(dates) || error("$dates is not sorted")
    seasonal_years =
        (year for (_, year) in ClimaAnalysis.Utils.find_season_and_year.(dates))
    prev_year, seasonal_years... = seasonal_years
    first_date, dates... = dates
    date_ranges = typeof(dates)[[first_date, first_date]]
    for (year, date) in zip(seasonal_years, dates)
        if year == prev_year
            last(date_ranges)[2] = date
        else
            push!(date_ranges, [date, date])
            prev_year = year
        end
    end
    return date_ranges
end

"""
    ObservationRecipe.change_data_type(var::OutputVar, data_type)

Return a `OutputVar` with `data` of type `data_type`.

This is useful if you want to make covariance matrix whose element type is
`data_type`.
"""
function ObservationRecipe.change_data_type(var::OutputVar, data_type)
    return ClimaAnalysis.remake(var, data = data_type.(var.data))
end

"""
    _stacked_samples(vars::Iterable{OutputVar}, sample_date_ranges)

Make a vector of stacked samples from multiple `OutputVar`s and an iterable of
2-iterable of dates.

Each sample corresponds to a single minibatch and the sample is stacked, because
it may contains multiple variables.
"""
function _stacked_samples(vars, sample_date_ranges)
    return map(sample_date_ranges) do (sample_start_date, sample_end_date)
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
    ret_dim_indices = [
        first(dim_indices) for
        dim_indices in dim_indices_groups if !isempty(dim_indices)
    ]

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

"""
    lat_weights_var(var::OutputVar)

Return a `OutputVar` where each data value corresponds to `(1 / max(cosd(lat),
min_cosd_lat))` if there is no `NaN` at its coordinate and `NaN` otherwise.
"""
function _lat_weights_var(var::OutputVar; min_cosd_lat = 0.1)
    ClimaAnalysis.has_latitude(var) || error(
        "Latitude dimension is not found in var with short name $(get(var.attributes, "short_name", nothing))",
    )
    # Because ClimaAnalysis units system does not know about degrees_north we
    # will check for units with a list of units instead
    deg_unit_names = ["degrees", "degree", "deg", "degs", "°", "degrees_north"]
    angle_dim_unit = ClimaAnalysis.dim_units(var, "latitude")
    lowercase(angle_dim_unit) in deg_unit_names ||
        error("The unit for latitude is missing or is not degree")

    lats = ClimaAnalysis.latitudes(var)
    FT = eltype(lats)

    # Take max to prevent small values in the covariance matrix so that taking
    # the inverse is stable
    lat_weights = one(FT) ./ max.(cosd.(lats), FT(min_cosd_lat))

    # Reshape for broadcasting
    lat_idx = var.dim2index[ClimaAnalysis.latitude_name(var)]
    reshape_tuple =
        (idx == lat_idx ? length(lats) : 1 for idx in 1:length(var.dims))
    lat_weights = reshape(lat_weights, reshape_tuple...)

    # Use broadcasting to compute the lat weight for each data point
    one_or_nan = x -> FT(isnan(x) ? x : one(x))
    lat_weights = lat_weights .* one_or_nan.(var.data)
    return ClimaAnalysis.remake(var, data = lat_weights)
end

"""
    get_metadata_for_nth_iteration(obs_series, N)

For the `N`th iteration, get the metadata of the observation(s) being processed.
"""
function ObservationRecipe.get_metadata_for_nth_iteration(obs_series, N)
    minibatch_obs =
        ObservationRecipe.get_observations_for_nth_iteration(obs_series, N)
    metadata_vec = map(obs -> EKP.get_metadata(obs), minibatch_obs)
    return vcat(metadata_vec...)
end

"""
    get_observations_for_nth_iteration(obs_series, N)

For the `N`th iteration, get the observation(s) being processed.
"""
function ObservationRecipe.get_observations_for_nth_iteration(obs_series, N)
    num_epoches = EKP.get_length_epoch(obs_series)
    # EKP.get_minibatch fails with N > num_epoches, so we use mod1 to go back to
    # the first epoch which seems consistent with what EKP does
    minibatch_indices = EKP.get_minibatch(obs_series, mod1(N, num_epoches))
    minibatch_obs = EKP.get_observations(obs_series)[minibatch_indices]
    return minibatch_obs
end

"""
    reconstruct_g_mean_final(ekp::EKP.EnsembleKalmanProcess)

Reconstruct the mean forward model evaluation at the last iteration as a
vector of `OutputVar`s.
"""
function ObservationRecipe.reconstruct_g_mean_final(
    ekp::EKP.EnsembleKalmanProcess,
)
    obs_series = EKP.get_observation_series(ekp)
    metadatas = ObservationRecipe.get_metadata_for_nth_iteration(
        obs_series,
        EKP.get_N_iterations(ekp),
    )
    eltype(metadatas) <: ClimaAnalysis.Var.Metadata || error(
        "Getting the short names from an observation is only supported with metadata from ClimaAnalysis",
    )
    g_mean = EKP.get_g_mean_final(ekp)

    # Check if length of g ensemble is the same as the length of the data in the metadatas
    total_metadata_length =
        sum(ClimaAnalysis.flattened_length(metadata) for metadata in metadatas)
    length(g_mean) != total_metadata_length && error(
        "Length of g_mean_final is not the same as the length of all the metadata",
    )

    # Reconstruct each OutputVar from the metadata
    minibatch_indices = _get_minibatch_indices_for_nth_iteration(
        obs_series,
        EKP.get_N_iterations(ekp),
    )
    vars = map(metadatas, minibatch_indices) do metadata, range
        ClimaAnalysis.unflatten(metadata, g_mean[range])
    end
    return vars
end

"""
    reconstruct_diag_cov(obs::EKP.Observation)

Reconstruct the diagonal of the covariance matrix in `obs` as a vector of
`OutputVar`s.

This function only supports observations that contain diagonal covariance
matrices.
"""
function ObservationRecipe.reconstruct_diag_cov(obs::EKP.Observation)
    all_metadata = EKP.get_metadata(obs)
    covs = EKP.get_covs(obs)

    eltype(covs) <: Diagonal || error(
        "The function reconstruct_diag_cov only supports observations with diagonal covariance matrices. Found covariance matrices of the type $(eltype(covs))",
    )

    # It would be nice to use a view instead of copying everything, but it makes
    # the indexing a bit more difficult
    cov_diags = mapreduce(cov -> view(cov, diagind(cov)), vcat, covs)

    start_index = 1
    vars = OutputVar[]
    for metadata in all_metadata
        data_size = ClimaAnalysis.flattened_length(metadata)
        var = ClimaAnalysis.unflatten(
            metadata,
            view(cov_diags, start_index:(start_index + data_size - 1)),
        )
        push!(vars, var)
        start_index += data_size
    end
    return vars
end

"""
    _get_minibatch_indices_for_nth_iteration(ekp::EKP.EnsembleKalmanProcess, N)

Get the indices that correspond to each metadata for the minibatch of the `N`th
iteration.

Note that the `indices` field in the `EKP.observation` cannot be used as the
multiple `OutputVar`s are flattened and concatenated together as a single
vector.
"""
function _get_minibatch_indices_for_nth_iteration(obs_series, N)
    metadatas = ObservationRecipe.get_metadata_for_nth_iteration(obs_series, N)
    minibatch_indices = UnitRange{Int}[]
    index = 1
    for metadata in metadatas
        data_size = ClimaAnalysis.flattened_length(metadata)
        start_idx = index
        index += data_size
        push!(minibatch_indices, start_idx:(start_idx + data_size - 1))
    end
    return minibatch_indices
end

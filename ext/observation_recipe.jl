import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.ObservationRecipe: AbstractCovarianceEstimator
import ClimaCalibrate.ObservationRecipe:
    ScalarCovariance, SeasonalDiagonalCovariance, SVDplusDCovariance
import ClimaCalibrate.ObservationRecipe: QuantileRegularization

"""
    covariance(
        covar_estimator::ScalarCovariance,
        osc::ObservedSampleCollection,
    )

Compute the scalar covariance matrix for the observation in `osc`.

When `covar_estimator.use_latitude_weights` is `true`, the data of the
observation is also used to determine which coordinates are `NaN`.

The matrix of samples in `osc` is ignored.
"""
function ObservationRecipe.covariance(
    covar_estimator::ScalarCovariance,
    osc::ObservedSampleCollection,
)
    FT = eltype(get_samples(osc))
    all_metadata = get_obs_metadata(osc)

    total_length = sum(ClimaAnalysis.flattened_length, all_metadata)
    diag_cov = fill(FT(covar_estimator.scalar), total_length)
    if covar_estimator.use_latitude_weights
        diag_cov .*= _flat_lat_weights(
            all_metadata,
            min_cosd_lat = covar_estimator.min_cosd_lat,
        )
    end
    return Diagonal(diag_cov)
end

"""
    covariance(
        covar_estimator::SeasonalDiagonalCovariance,
        osc::ObservedSampleCollection,
    )

Compute the diagonal covariance matrix of seasonal quantities for the
observation in `osc`.

The diagonal entries are the per-entry variance across the samples (columns) of
`osc.sample_collection` — i.e. the variance of each season across years
(`NaN`s ignored). Each sample (column) must represent the same sequence of
seasons with one time slice per season. This is validated against the metadata.
At least two samples (years) are required to estimate a variance.
"""
function ObservationRecipe.covariance(
    covar_estimator::SeasonalDiagonalCovariance,
    osc::ObservedSampleCollection,
)
    samples = get_samples(osc)
    n_samples = size(samples, 2)
    n_samples >= 2 || error(
        "SeasonalDiagonalCovariance needs at least 2 samples to estimate the covariance matrix; got $n_samples",
    )

    # The samples must represent seasons that line up across all samples
    _check_metadata_represent_seasons(get_metadata(osc))

    # Variance of each season across the samples (years)
    # Reduce per row, since we want the variance of each season
    diag_cov = collect(nanvar(view(samples, i, :)) for i in axes(samples, 1))

    # Add model error scale
    if !iszero(covar_estimator.model_error_scale)
        seasonal_mean =
            collect(nanmean(view(samples, i, :)) for i in axes(samples, 1))
        diag_cov .+= (covar_estimator.model_error_scale .* seasonal_mean) .^ 2
    end

    # Add regularization
    !iszero(covar_estimator.regularization) &&
        (diag_cov .+= covar_estimator.regularization)

    # Add latitude weights
    if covar_estimator.use_latitude_weights
        # The metadata is the same across all samples, so the observation's is fine
        all_metadata = get_obs_metadata(osc)
        diag_cov .*= _flat_lat_weights(
            all_metadata,
            min_cosd_lat = covar_estimator.min_cosd_lat,
        )
    end
    return Diagonal(diag_cov)
end

"""
    _check_metadata_represent_seasons(metadata_mat)

Validate that the samples with metadata in `metadata_mat` represent seasons.

For every row of the matrix of metadata, each metadata must represent a single
year of seasonal data (e.g. (2009 DJF, 2010 MAM, 2010 JJA, and 2010 SON)). Every
row must have the same ordering of seasons, but the year can differ between the
samples.

Note that the years between the samples can be the same, since the samples can
be generated from a simulation by varying the parameters or initial condition,
but keeping everything else the same.
"""
function _check_metadata_represent_seasons(metadata_mat)
    for metadata_row in eachrow(metadata_mat)
        all(ClimaAnalysis.has_time.(metadata_row)) || error(
            "SeasonalDiagonalCovariance require every variable to have a time dimension",
        )

        date_vecs = ClimaAnalysis.dates.(metadata_row)
        seasons_found = nothing
        for date_vec in date_vecs
            season_and_year_vec = find_season_and_year.(date_vec)
            allunique(season_and_year_vec) || error(
                "Multiple time points correspond to the same season and year",
            )

            length(season_and_year_vec) > 4 && error(
                "There are more than 4 combination of season and year identified for a single variable",
            )

            # The limitation of this is that we specify what constitutes a year
            # worth of seasons
            allequal(last.(season_and_year_vec)) ||
                error("The seasons do not come from the same year")
            isnothing(seasons_found) &&
                (seasons_found = first.(season_and_year_vec))

            seasons_found == first.(season_and_year_vec) || error(
                "Order of seasons for a variable across different samples are not the same",
            )
        end

    end
    return nothing
end

"""
    covariance(
        covar_estimator::SVDplusDCovariance,
        osc::ObservedSampleCollection,
    )

Compute the `EKP.SVDplusD` covariance matrix for the observation in `osc`.

The covariance is computed from all the samples (columns) in
`osc.sample_collection`. The metadata of the chosen observation column
(`osc.i`) is used to size and weight the diagonal terms.
"""
function ObservationRecipe.covariance(
    covar_estimator::SVDplusDCovariance,
    osc::ObservedSampleCollection,
)
    stacked_sample_matrix = copy(get_samples(osc))
    metadata = get_obs_metadata(osc)

    # Apply latitude weights first so that both the SVD and the model error
    # scale (the mean) are computed from the weighted matrix.
    covar_estimator.use_latitude_weights && _apply_lat_weights_to_samples!(
        stacked_sample_matrix,
        metadata,
        min_cosd_lat = covar_estimator.min_cosd_lat,
    )

    # Compute SVD of covariance matrix
    (; rank) = covar_estimator
    gamma_low_rank = if isnothing(rank)
        EKP.tsvd_cov_from_samples(stacked_sample_matrix)
    else
        EKP.tsvd_cov_from_samples(stacked_sample_matrix, rank)
    end

    rank_of_svd = length(gamma_low_rank.S)
    !isnothing(rank) &&
        rank_of_svd != rank &&
        @warn "Rank of SVD is $rank_of_svd but requested rank is $rank"

    # Add model error scale. This may not make sense if the samples do not
    # represent a single year. For example, if the stacked samples are seasonal
    # averages over two years, then this quantity is the mean of seasonal
    # averages spanned over two years, where the first DJF is the mean of every
    # other DJF and the second DJF is the mean of every other DJF.
    FT = eltype(stacked_sample_matrix)
    model_error_scale =
        (
            FT(covar_estimator.model_error_scale) .*
            mean(stacked_sample_matrix, dims = 2)
        ) .^ 2
    model_error_scale = Diagonal(vec(model_error_scale))

    # Add regularization
    regularization = create_regularization(
        covar_estimator.regularization,
        covar_estimator,
        metadata,
        model_error_scale,
    )

    return EKP.SVDplusD(gamma_low_rank, model_error_scale + regularization)
end

"""
    create_regularization(regularization::AbstractFloat, _, _, model_error_scale)

Create the regularization matrix of the form `regularization * I`.

The scalar is cast to the element type of `model_error_scale` so the resulting
covariance keeps a consistent element type (e.g. Float32).
"""
function create_regularization(
    regularization::AbstractFloat,
    _,
    _,
    model_error_scale,
)
    FT = eltype(model_error_scale)
    return FT(regularization) * I
end

"""
    create_regularization(
        regularization::QuantileRegularization,
        covar_estimator::SVDplusDCovariance,
        metadata,
        model_error_scale,
    )

Create the regularization matrix where each variable gets its own regularization
value based on the `regularization.qtl` quantile of its model error scale
vector.

For each variable, the `qtl` quantile of the model error scale diagonal entries
corresponding to that variable is computed and used as a constant regularization
term for all entries belonging to that variable. The per-variable index ranges
are determined from `metadata` (one `Metadata` per variable).
"""
function create_regularization(
    regularization::QuantileRegularization,
    covar_estimator::SVDplusDCovariance,
    metadata,
    model_error_scale,
)
    indices_vec = _get_indices_of_metadata(metadata)

    (; qtl) = regularization

    model_error_scale_vec = model_error_scale.diag
    FT = eltype(model_error_scale)

    regularization_vals_vec = []
    for (i, indices) in enumerate(indices_vec)
        var_model_error_scale_vec = view(model_error_scale_vec, indices)
        # Check that there is a sufficient number of samples (e.g. if qtl =
        # 0.05, there should be at least 20 samples for a meaningful
        # quantile computation)
        length(var_model_error_scale_vec) < 1 / qtl &&
            error("Insufficient samples for computing quantile")
        qtl_for_var = FT(Statistics.quantile(var_model_error_scale_vec, qtl))
        qtl_for_var ≈ 0.0 && error(
            "Zero found for the quantile ($qtl) of the model error scale for the variable ($(ClimaAnalysis.short_name(metadata[i]))). The model error scale ($(covar_estimator.model_error_scale)) might be too small",
        )
        push!(regularization_vals_vec, qtl_for_var)
    end

    return Diagonal(
        vcat(
            [
                fill(reg, length(indices)) for
                (reg, indices) in zip(regularization_vals_vec, indices_vec)
            ]...,
        ),
    )
end

"""
    _apply_lat_weights_to_samples!(
        stacked_sample_matrix,
        all_metadata;
        min_cosd_lat = 0.1,
    )

Apply latitude weights to every column of `stacked_sample_matrix` in place.

The latitude weight applied is `sqrt(1 / max(cosd(lat), min_cosd_lat))`. The
weights are derived from `all_metadata` (see [`_flat_lat_weights`](@ref)) and
are the same for every column, so the same weight vector is applied to all
columns.
"""
function _apply_lat_weights_to_samples!(
    stacked_sample_matrix,
    all_metadata;
    min_cosd_lat = 0.1,
)
    # It is okay to find the latitude weights for a single column and apply it
    # to every other column, because the flattening of OutputVars should be the
    # same for each column
    flat_lat_weights = _flat_lat_weights(all_metadata; min_cosd_lat)
    stacked_sample_matrix .*= sqrt.(flat_lat_weights)
    return nothing
end

"""
    observation(
        covar_estimator::AbstractCovarianceEstimator,
        osc::ObservedSampleCollection;
        name = nothing,
    )

Return an `EKP.Observation` with the observation in `osc` as the sample, a
covariance matrix defined by `covar_estimator`, `name` determined from the short
names of the observation, and metadata.

!!! note "Metadata"
    Metadata in `EKP.observation` is only added with versions of
    EnsembleKalmanProcesses later than v2.4.2.
"""
function ObservationRecipe.observation(
    covar_estimator::AbstractCovarianceEstimator,
    osc::ObservedSampleCollection;
    name = nothing,
)
    stacked_sample = collect(get_obs(osc))
    metadata = collect(get_obs_metadata(osc))

    any(==(""), ClimaAnalysis.short_name.(metadata)) && @warn(
        "There are OutputVar(s) with no short name. You will not be able to use GEnsembleBuilder"
    )

    covar = ObservationRecipe.covariance(covar_estimator, osc)

    # Concatenate names and separating them with a semicolon
    isnothing(name) &&
        (name = join([ClimaAnalysis.short_name(m) for m in metadata], ";"))
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
    metadata = EKP.get_metadata(obs)
    all(m isa ClimaAnalysis.Var.Metadata for m in metadata) || error(
        "Getting the short names from an observation is only supported with metadata from ClimaAnalysis",
    )

    short_names =
        collect(get(m.attributes, "short_name", nothing) for m in metadata)
    return short_names
end

"""
    seasonally_aligned_yearly_sample_date_ranges(var::OutputVar)

Generate sample dates that conform to a seasonally aligned year from
`dates(var)`.

A seasonally aligned year is defined to be from December to November of the
following year.

This function is useful for finding the sample dates of samples consisting of
all four seasons in a single year. For example, one can pass these date ranges
to `SampleBuilder.build_samples_by_times` to build the samples for a
`SVDplusDCovariance` or `SeasonalDiagonalCovariance`.

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
    seasonal_years = (year for (_, year) in find_season_and_year.(dates))
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
    change_data_type(var::OutputVar, data_type)

Return a `OutputVar` with `data` of type `data_type`.

This is useful if you want to make covariance matrix whose element type is
`data_type`.
"""
function ObservationRecipe.change_data_type(var::OutputVar, data_type)
    return ClimaAnalysis.remake(var, data = data_type.(var.data))
end

"""
    _flat_lat_weights(all_metadata; min_cosd_lat = 0.1)

Return the latitude weights `1 / max(cosd(lat), min_cosd_lat)`, flattened and
concatenated to match `all_metadata`, an iterable of
`ClimaAnalysis.Var.Metadata`.
"""
function _flat_lat_weights(all_metadata; min_cosd_lat = 0.1)
    parts = Vector[]
    for metadata in all_metadata
        data_length = ClimaAnalysis.flattened_length(metadata)
        var = ClimaAnalysis.unflatten(metadata, ones(data_length))
        push!(
            parts,
            ClimaAnalysis.flatten(
                _lat_weights_var(var; min_cosd_lat),
                metadata,
            ).data,
        )
    end
    return reduce(vcat, parts)
end

"""
    _lat_weights_var(var::OutputVar; min_cosd_lat = 0.1)

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
    reconstruct_g(ekp::EKP.EnsembleKalmanProcess, it::Integer)

Reconstruct the G ensemble matrix of the `it`th iteration as a matrix of
`OutputVar`s.
"""
function ObservationRecipe.reconstruct_g(
    ekp::EKP.EnsembleKalmanProcess,
    it::Integer,
)
    obs_series = EKP.get_observation_series(ekp)
    metadata = ClimaCalibrate.get_metadata_for_nth_iteration(obs_series, it)
    all(m isa ClimaAnalysis.Var.Metadata for m in metadata) || error(
        "Reconstructing g is only possible when the metadata are all ClimaAnalysis.Var.Metadata",
    )

    g_ens = EKP.get_g(ekp, it)
    num_cols = size(g_ens)[2]

    # Check if length of g ensemble is the same as the length of the data in the metadatas
    total_metadata_length =
        sum(ClimaAnalysis.flattened_length(m) for m in metadata)
    size(g_ens, 1) != total_metadata_length && error(
        "Length of g_ens is not the same as the length of all the metadata",
    )

    # Reconstruct each OutputVar for every ensemble member (column of g_ens)
    ranges = _get_indices_of_metadata(metadata)
    vars_per_ens = [
        map(metadata, ranges) do m, range
            ClimaAnalysis.unflatten(m, g_ens[range, col])
        end for col in 1:num_cols
    ]
    return hcat(vars_per_ens...)
end

"""
    reconstruct_g_mean(ekp::EKP.EnsembleKalmanProcess, it::Integer)

Reconstruct the mean forward model evaluation at the `it`th iteration as a
vector of `OutputVar`s.
"""
function ObservationRecipe.reconstruct_g_mean(
    ekp::EKP.EnsembleKalmanProcess,
    it::Integer,
)
    obs_series = EKP.get_observation_series(ekp)
    metadata = ClimaCalibrate.get_metadata_for_nth_iteration(obs_series, it)
    all(m isa ClimaAnalysis.Var.Metadata for m in metadata) || error(
        "Reconstructing g_mean is only possible when the metadata are all ClimaAnalysis.Var.Metadata",
    )

    g_mean = EKP.get_g_mean(ekp, it)

    # Check if length of g ensemble is the same as the length of the data in the metadatas
    total_metadata_length =
        sum(ClimaAnalysis.flattened_length(m) for m in metadata)
    length(g_mean) != total_metadata_length && error(
        "Length of g_mean is not the same as the length of all the metadata",
    )

    return _reconstruct_vars(g_mean, metadata)
end

"""
    reconstruct_g_mean_final(ekp::EKP.EnsembleKalmanProcess)

Reconstruct the mean forward model evaluation at the last iteration as a
vector of `OutputVar`s.
"""
function ObservationRecipe.reconstruct_g_mean_final(
    ekp::EKP.EnsembleKalmanProcess,
)
    ObservationRecipe.reconstruct_g_mean(ekp, EKP.get_N_iterations(ekp))
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

    return _reconstruct_vars(cov_diags, all_metadata)
end

"""
    reconstruct_vars(obs::EKP.Observation)

Reconstruct the `OutputVar`s from the `samples` in `obs`.
"""
function ObservationRecipe.reconstruct_vars(obs::EKP.Observation)
    all_metadata = EKP.get_metadata(obs)
    samples = EKP.get_samples(obs)
    stacked_sample = reduce(vcat, samples)

    return _reconstruct_vars(stacked_sample, all_metadata)
end

"""
    _get_minibatch_indices_for_nth_iteration(obs_series, N)

Get the indices that correspond to each metadata for the minibatch of the `N`th
iteration.

Note that the `indices` field in the `EKP.observation` cannot be used as the
multiple `OutputVar`s are flattened and concatenated together as a single
vector.
"""
function ObservationRecipe._get_minibatch_indices_for_nth_iteration(
    obs_series,
    N,
)
    all_metadata = ClimaCalibrate.get_metadata_for_nth_iteration(obs_series, N)
    minibatch_indices = _get_indices_of_metadata(all_metadata)
    return minibatch_indices
end

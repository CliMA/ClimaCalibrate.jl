module ObservationRecipe

export ScalarCovariance,
    SeasonalDiagonalCovariance,
    SVDplusDCovariance,
    QuantileRegularization,
    covariance,
    observation,
    short_names,
    seasonally_aligned_yearly_sample_date_ranges,
    change_data_type,
    reconstruct_g,
    reconstruct_g_mean,
    reconstruct_g_mean_final,
    reconstruct_diag_cov,
    reconstruct_vars

"""
    abstract type AbstractCovarianceEstimator end

An object that estimates the noise covariance matrix from the samples in an
`ObservedSampleCollection`.

`AbstractCovarianceEstimator` have to provide one function,
`ObservationRecipe.covariance`.

The function has to have the signature

```julia
ObservationRecipe.covariance(
    covar_estimator::AbstractCovarianceEstimator,
    osc::ObservedSampleCollection,
)
```

and return a noise covariance matrix. The `ObservedSampleCollection` (built with
`SampleBuilder.build_samples` / `build_samples_by_times` and `choose_obs`)
carries the matrix of flattened samples and their metadata; the chosen
observation is column `osc.i`.
"""
abstract type AbstractCovarianceEstimator end

"""
    ScalarCovariance <: AbstractCovarianceEstimator

Contain the necessary information to construct the scalar covariance matrix.
"""
struct ScalarCovariance{FT1 <: AbstractFloat, FT2 <: AbstractFloat} <:
       AbstractCovarianceEstimator
    """Scalar to multiply the identity matrix by"""
    scalar::FT1

    """Use latitude weights"""
    use_latitude_weights::Bool

    """The minimum cosine weight when using latitude weighting"""
    min_cosd_lat::FT2
end

"""
    ScalarCovariance(;
        scalar = 1.0,
        use_latitude_weights = false,
        min_cosd_lat = 0.1,
    )

Create a `ScalarCovariance` which specifies how the covariance matrix should be
formed. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `Diagonal` matrix.

Keyword arguments
=====================

- `scalar`: Scalar value to multiply the identity matrix by.

- `use_latitude_weights`: If `true`, then latitude weighting is applied to the
  covariance matrix. Latitude weighting is multiplying the values along the
  diagonal of the covariance matrix by `(1 / max(cosd(lat), min_cosd_lat))`. See
  the keyword argument `min_cosd_lat` for more information.

- `min_cosd_lat`: Control the minimum latitude weight when
  `use_latitude_weights` is `true`. The value for `min_cosd_lat` must be greater
  than zero as values close to zero along the diagonal of the covariance matrix
  can lead to issues when taking the inverse of the covariance matrix.
"""
function ScalarCovariance(;
    scalar = 1.0,
    use_latitude_weights = false,
    min_cosd_lat = 0.1,
)
    if scalar <= zero(scalar)
        error("The value for scalar ($scalar) should be positive")
    end
    if use_latitude_weights && min_cosd_lat <= zero(min_cosd_lat)
        error(
            "The value for min_cosd_lat ($min_cosd_lat) should be greater than zero",
        )
    end

    return ScalarCovariance(scalar, use_latitude_weights, min_cosd_lat)
end

"""
    SeasonalDiagonalCovariance <: AbstractCovarianceEstimator

Contain the necessary information to construct a diagonal covariance matrix
whose entries represent seasonal covariances from `ClimaAnalysis.OutputVar`s.
"""
struct SeasonalDiagonalCovariance{
    FT1 <: AbstractFloat,
    FT2 <: AbstractFloat,
    FT3 <: AbstractFloat,
} <: AbstractCovarianceEstimator
    """A model error scale term added to the diagonal of the covariance
    matrix"""
    model_error_scale::FT1

    """A regularization term added to the diagonal of the covariance matrix"""
    regularization::FT2

    """Use latitude weights"""
    use_latitude_weights::Bool

    """The minimum cosine weight when using latitude weighting"""
    min_cosd_lat::FT3
end

"""
    SeasonalDiagonalCovariance(;
        model_error_scale = 0.0,
        regularization = 0.0,
        use_latitude_weights = false,
        min_cosd_lat = 0.1,
    )

Create a `SeasonalDiagonalCovariance` which specifies how the covariance matrix
should be formed. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `Diagonal` matrix.

The samples used to compute the covariance matrix come from the
`ObservedSampleCollection` (e.g. built with
`SampleBuilder.build_samples_by_times`), where each sample is one year of
seasonal statistics. `NaN`s are ignored when computing the seasonal variance.

Keyword arguments
=====================

- `model_error_scale`: Noise from the model error added to the covariance
  matrix. This is `(model_error_scale * seasonal_mean).^2`, where
  `seasonal_mean` is the seasonal mean for each of the quantity for each of the
  season (DJF, MAM, JJA, SON).

- `regularization`: A diagonal matrix of the form `regularization * I` is added
  to the covariance matrix.

- `use_latitude_weights`: If `true`, then latitude weighting is applied to the
  covariance matrix. Latitude weighting is multiplying the values along the
  diagonal of the covariance matrix by `(1 / max(cosd(lat), min_cosd_lat))`. See
  the keyword argument `min_cosd_lat` for more information.

- `min_cosd_lat`: Control the minimum latitude weight when
  `use_latitude_weights` is `true`. The value for `min_cosd_lat` must be greater
  than zero as values close to zero along the diagonal of the covariance matrix
  can lead to issues when taking the inverse of the covariance matrix.
"""
function SeasonalDiagonalCovariance(;
    model_error_scale = 0.0,
    regularization = 0.0,
    use_latitude_weights = false,
    min_cosd_lat = 0.1,
)
    model_error_scale < zero(model_error_scale) &&
        error("Model_error_scale ($model_error_scale) should not be negative")
    regularization < zero(regularization) &&
        error("Regularization ($regularization) should not be negative")
    if use_latitude_weights && min_cosd_lat <= zero(min_cosd_lat)
        error(
            "The value for min_cosd_lat ($min_cosd_lat) should be greater than zero",
        )
    end

    return SeasonalDiagonalCovariance(
        model_error_scale,
        regularization,
        use_latitude_weights,
        min_cosd_lat,
    )
end

"""
    QuantileRegularization

Regularization using the quantile of the model error scale for each
`OutputVar`.

The same quantile is used for each `OutputVar` when making the observation.

This is used for the `SVDplusDCovariance` matrix.

Examples
========

In the example below, a regularization using the 0.05 quantile of the model
error scale for each variable is initialized.

```julia
qtl_regularization = QuantileRegularization(0.05)
```
"""
struct QuantileRegularization{FT <: AbstractFloat}
    qtl::FT
    function QuantileRegularization(qtl::AbstractFloat)
        (qtl <= 0 || qtl > 1) && error("Quantile must be in (0, 1], got $qtl")
        new{typeof(qtl)}(qtl)
    end
end

"""
    SVDplusDCovariance <: AbstractCovarianceEstimator

Contain the necessary information to construct a `EKP.SVDplusD` covariance
matrix from `ClimaAnalysis.OutputVar`s.
"""
struct SVDplusDCovariance{
    FT1 <: AbstractFloat,
    FT2 <: Union{AbstractFloat, QuantileRegularization},
    FT3 <: AbstractFloat,
    R <: Union{Integer, Nothing},
} <: AbstractCovarianceEstimator
    """A model error scale term added to the diagonal of the covariance
    matrix"""
    model_error_scale::FT1

    """A regularization term added to the diagonal of the covariance matrix"""
    regularization::FT2

    """Use latitude weights"""
    use_latitude_weights::Bool

    """The minimum cosine weight when using latitude weighting"""
    min_cosd_lat::FT3

    """Rank of the singular value decomposition (SVD)"""
    rank::R
end

"""
    SVDplusDCovariance(;
        model_error_scale = 0.0,
        regularization = 0.0,
        use_latitude_weights = false,
        min_cosd_lat = 0.1,
        rank = nothing
    )

Create a `SVDplusDCovariance` which specifies how the covariance matrix should
be formed. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `EKP.SVDplusD` covariance matrix.

The samples used to compute the covariance matrix come from the
`ObservedSampleCollection` (e.g. built with
`SampleBuilder.build_samples_by_times`), where each sample is one column.

!!! note "Recommended sample size"
    When constructing the samples (e.g. with `build_samples_by_times`), it is
    recommended that each sample contains data from a single year. For example,
    if the samples are created from time series data of seasonal averages, then
    each sample should contain all four seasons. Otherwise, the covariance matrix
    may not make sense. For example, if each sample contains two years of
    seasonally averaged data, then the sample mean is the seasonal mean of every
    other season across the years stacked vertically. For a concrete example, if
    the sample contain DJF for both 2010 and 2011. Then, the sample mean will be
    the mean of DJF 2010, 2012, and so on, and the mean of DJF 2011, 2013, and so
    on. As a result, if one were to use this covariance matrix with
    `model_error_scale`, the covariance matrix will not make sense.

Keyword arguments
=====================

- `model_error_scale`: Noise from the model error added to the covariance
  matrix. This is `(model_error_scale * mean(samples, dims = 2)).^2`, where
  `mean(samples, dims = 2)` is the mean of the samples.

- `regularization`: If a scalar is used, a diagonal matrix of the form
  `regularization * I` is added to the covariance matrix. See
  [`QuantileRegularization`](@ref) for another option for regularization.

- `use_latitude_weights`: If `true`, then latitude weighting is applied to the
  covariance matrix. Latitude weighting is multiplying the columns of the matrix
  of samples by `1 / sqrt(max(cosd(lat), 0.1))`. See the keyword argument
  `min_cosd_lat` for more information.

- `min_cosd_lat`: Control the minimum latitude weight when
  `use_latitude_weights` is `true`. The value for `min_cosd_lat` must be greater
  than zero as values close to zero along the diagonal of the covariance matrix
  can lead to issues when taking the inverse of the covariance matrix.

- `rank`: Rank of the singular value decomposition (SVD). If `nothing` is passed
  in, then the rank is automatically inferred from the data.
"""
function SVDplusDCovariance(;
    model_error_scale = 0.0,
    regularization = 0.0,
    use_latitude_weights = false,
    min_cosd_lat = 0.1,
    rank = nothing,
)
    model_error_scale < zero(model_error_scale) &&
        error("Model_error_scale ($model_error_scale) should not be negative")
    if regularization isa AbstractFloat
        regularization < zero(regularization) &&
            error("Regularization ($regularization) should not be negative")
    end
    if use_latitude_weights && min_cosd_lat <= zero(min_cosd_lat)
        error(
            "The value for min_cosd_lat ($min_cosd_lat) should be greater than zero",
        )
    end
    isnothing(rank) ||
        rank >= 0 ||
        error("Rank ($rank) should be nothing or non-negative")

    return SVDplusDCovariance(
        model_error_scale,
        regularization,
        use_latitude_weights,
        min_cosd_lat,
        rank,
    )
end

function covariance end

function observation end

function short_names end

function seasonally_aligned_yearly_sample_date_ranges end

function change_data_type end

function reconstruct_g end

function reconstruct_g_mean end

function reconstruct_g_mean_final end

function reconstruct_diag_cov end

function reconstruct_vars end

function _get_minibatch_indices_for_nth_iteration end

end

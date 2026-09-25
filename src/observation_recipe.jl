"""
    ClimaCalibrate.ObservationRecipe

Estimate a noise covariance from a `SampleCollection` and build an
`EKP.Observation` from it.

Three estimators are available: [`ScalarCovariance`](@ref) for a multiple of the
identity, [`SeasonalDiagonalCovariance`](@ref) for the per-season variance
across years, and [`SVDplusDCovariance`](@ref) for a low-rank sample
covariance plus a diagonal term. All of them take samples built by
[`ClimaCalibrate.SampleBuilder`](@ref).

Also reconstructs the flattened vectors back into `OutputVar`s
([`reconstruct_vars`](@ref), [`reconstruct_g`](@ref)), so a calibration's
observations and forward map output can be inspected.

Requires ClimaAnalysis to be loaded.
"""
module ObservationRecipe

export ScalarCovariance,
    SeasonalDiagonalCovariance,
    SVDplusDCovariance,
    QuantileRegularization,
    covariance,
    observation,
    short_names,
    seasonally_aligned_yearly_sample_date_ranges,
    reconstruct_g,
    reconstruct_g_mean,
    reconstruct_g_mean_final,
    reconstruct_diag_cov,
    reconstruct_vars,
    reconstruct_residual

import ..SampleBuilder: LatitudeWeighting

include("diagonal_term.jl")

"""
    AbstractCovarianceEstimator

An object that estimates the noise covariance matrix from the samples in a
`SampleCollection`.

`AbstractCovarianceEstimator` have to provide one function,
`ObservationRecipe.covariance`.

The function has to have the signature

```julia
ObservationRecipe.covariance(
    covar_estimator::AbstractCovarianceEstimator,
    sample_collection,
)
```

and return a noise covariance matrix. The `SampleCollection` carries the matrix
of flattened samples and their metadata. The covariance matrix does not depend
on which sample is chosen as the observation.

Subtypes:
- [`ScalarCovariance`](@ref): a multiple of the identity.
- [`SeasonalDiagonalCovariance`](@ref): the per-season variance across samples.
- [`SVDplusDCovariance`](@ref): a low-rank sample covariance plus a diagonal
  term.
"""
abstract type AbstractCovarianceEstimator end

"""
    ScalarCovariance <: AbstractCovarianceEstimator

Covariance estimator contain the necessary information to construct the scalar
covariance matrix.

`FT1` and `FT2` are the element types of `scalar` and `min_cosd_lat`.
"""
struct ScalarCovariance{FT1 <: AbstractFloat, FT2 <: AbstractFloat} <:
       AbstractCovarianceEstimator
    """Scalar to multiply the identity matrix by"""
    scalar::FT1

    """Whether to apply latitude weighting"""
    use_latitude_weights::Bool

    """The smallest `cosd(lat)` used in the latitude weight, which caps the
    weight at `1 / min_cosd_lat`"""
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

# Keyword Arguments

- `scalar`: Scalar value to multiply the identity matrix by.

- `use_latitude_weights`: If `true`, then latitude weighting is applied to the
  covariance matrix. Latitude weighting is multiplying the values along the
  diagonal of the covariance matrix by `(1 / max(cosd(lat), min_cosd_lat))`. See
  the keyword argument `min_cosd_lat` for more information.

- `min_cosd_lat`: Control the minimum latitude weight when
  `use_latitude_weights` is `true`. The weight is
  `1 / max(cosd(lat), min_cosd_lat)`, so this is the largest weight any point
  can be given, `1 / min_cosd_lat`. Without it the weight grows without bound
  toward the poles, where `cosd(lat)` reaches zero, and the diagonal entries
  span so many orders of magnitude that the covariance is badly conditioned.
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

Covariance estimator that contain the necessary information to construct a
diagonal matrix whose diagonal is the per-season variance across the samples of
a `SampleCollection`.
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

    """Whether to apply latitude weighting"""
    use_latitude_weights::Bool

    """The smallest `cosd(lat)` used in the latitude weight, which caps the
    weight at `1 / min_cosd_lat`"""
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
`SampleCollection`, where each sample is one year of seasonal statistics. `NaN`s
are ignored when computing the seasonal variance.

# Keyword Arguments

- `model_error_scale`: Noise from the model error added to the covariance
  matrix. This is `(model_error_scale * seasonal_mean).^2`, where
  `seasonal_mean` is the seasonal mean for each of the quantity for each of the
  season (DJF, MAM, JJA, SON).

- `regularization`: A diagonal matrix of the form `regularization * I` is added
  to the covariance matrix. It is added *before* latitude weighting, so with
  `use_latitude_weights = true` the effective regularization varies with
  latitude, unlike [`SVDplusDCovariance`](@ref), which adds it afterwards.

- `use_latitude_weights`: If `true`, then latitude weighting is applied to the
  covariance matrix. Latitude weighting is multiplying the values along the
  diagonal of the covariance matrix by `(1 / max(cosd(lat), min_cosd_lat))`. See
  the keyword argument `min_cosd_lat` for more information.

- `min_cosd_lat`: Control the minimum latitude weight when
  `use_latitude_weights` is `true`. The weight is
  `1 / max(cosd(lat), min_cosd_lat)`, so this is the largest weight any point
  can be given, `1 / min_cosd_lat`. Without it the weight grows without bound
  toward the poles, where `cosd(lat)` reaches zero, and the diagonal entries
  span so many orders of magnitude that the covariance is badly conditioned.
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

# Examples

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

Covariance estimator that returns an `EKP.SVDplusD`: a low-rank sample
covariance plus a diagonal term.
"""
struct SVDplusDCovariance{
    D <: AbstractDiagonalTerm,
    L <: Union{Nothing, LatitudeWeighting},
    R <: Union{Integer, Nothing},
} <: AbstractCovarianceEstimator
    """A diagonal term that describes the diagonal matrix added to the low rank
    approximation of the covariance matrix"""
    diagonal::D

    """The latitude weighting applied to the samples, or `nothing` for no
    latitude weighting"""
    latitude_weighting::L

    """Compute the diagonal term from the latitude weighted samples when
    latitude weights are used"""
    use_weighted_samples_for_diagonal::Bool

    """Rank of the singular value decomposition, or `nothing` to infer it from
    the data"""
    rank::R
end

"""
    SVDplusDCovariance(;
        model_error_scale = 0.0,
        regularization = 0.0,
        latitude_weighting = nothing,
        use_weighted_samples_for_diagonal = true,
        rank = nothing
    )

Create a `SVDplusDCovariance` which specifies how the covariance matrix should
be formed. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `EKP.SVDplusD` covariance matrix.

The samples used to compute the covariance matrix come from the
`SampleCollection`, where each sample is one column.

!!! note "Recommended sample size"
    When constructing the samples (e.g. with `build_samples_by_times`), it is
    recommended that each sample contains data from a single year. For example,
    if the samples are created from time series data of seasonal averages, then
    each sample should contain all four seasons. Otherwise, the covariance matrix
    may not make sense. For example, if each sample contains two years of
    seasonally averaged data, then the sample mean is the seasonal mean of every
    other season across the years stacked vertically. For a concrete example, if
    the samples contain DJF for both 2010 and 2011. Then, the sample mean will be
    the mean of DJF 2010, 2012, and so on, and the mean of DJF 2011, 2013, and so
    on. As a result, if one were to use this covariance matrix with
    `model_error_scale`, the covariance matrix will not make sense.

# Keyword Arguments

- `model_error_scale`: Noise from the model error added to the covariance
  matrix. This is `(model_error_scale * mean(samples, dims = 2)).^2`, where
  `mean(samples, dims = 2)` is the mean of the samples.

- `regularization`: If a scalar is used, a diagonal matrix of the form
  `regularization * I` is added to the covariance matrix. See
  [`QuantileRegularization`](@ref) for another option for regularization.

- `latitude_weighting`: Apply the latitude weighting to the matrix of samples.
  Without it the weight grows without bound toward the poles, where `cosd(lat)`
  reaches zero, and the diagonal entries span so many orders of magnitude that
  the covariance is badly conditioned.

- `use_weighted_samples_for_diagonal`: If `true` and `latitude_weighting` is not
  `nothing`, then the diagonal term is computed from the latitude weighted
  samples. Otherwise, the diagonal term is computed from the samples without
  latitude weighting. This has no effect when `latitude_weighting` is `nothing`.

- `rank`: Rank of the singular value decomposition (SVD). If `nothing` is passed
  in, then the rank is automatically inferred from the data.

!!! warning "Deprecated keyword arguments"
    The keyword arguments `use_latitude_weights` and `min_cosd_lat` are
    deprecated and have been replaced by `latitude_weighting`.
"""
function SVDplusDCovariance(;
    model_error_scale = 0.0,
    regularization = 0.0,
    latitude_weighting = nothing,
    min_cosd_lat = nothing,
    rank = nothing,
    # Deprecated keyword arguments
    use_latitude_weights = nothing,
    use_weighted_samples_for_diagonal = true,
)
    model_error_scale < zero(model_error_scale) &&
        error("Model_error_scale ($model_error_scale) should not be negative")
    if regularization isa AbstractFloat
        regularization < zero(regularization) &&
            error("Regularization ($regularization) should not be negative")
    end

    return SVDplusDCovariance(
        _diagonal_term(model_error_scale, regularization);
        latitude_weighting,
        use_latitude_weights,
        use_weighted_samples_for_diagonal,
        min_cosd_lat,
        rank,
    )
end

"""
    SVDplusDCovariance(
        diagonal::AbstractDiagonalTerm;
        latitude_weighting = nothing,
        use_weighted_samples_for_diagonal = true,
        rank = nothing,
    )

Create a `SVDplusDCovariance` whose diagonal matrix is described by the diagonal
term `diagonal`. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `EKP.SVDplusD` covariance matrix.

Passing `model_error_scale = x` and `regularization = y` to the keyword
constructor is the same as passing
`diagonal = ModelErrorScaleDiagonal(x) .+ ScalarDiagonal(y)`. Passing
`regularization = QuantileRegularization(q)` instead is the same as passing
`diagonal = ModelErrorScaleDiagonal(x) .+ QuantileDiagonal(q, ModelErrorScaleDiagonal(x))`.

# Keyword Arguments

- `latitude_weighting`: Apply the latitude weighting to the matrix of samples.
  Without it the weight grows without bound toward the poles, where `cosd(lat)`
  reaches zero, and the diagonal entries span so many orders of magnitude that
  the covariance is badly conditioned.

- `use_weighted_samples_for_diagonal`: If `true` and `latitude_weighting` is not
  `nothing`, then the diagonal term is computed from the latitude weighted
  samples. Otherwise, the diagonal term is computed from the samples without
  latitude weighting. This has no effect when `latitude_weighting` is `nothing`.

- `rank`: Rank of the singular value decomposition (SVD). If `nothing` is passed
  in, then the rank is automatically inferred from the data.

!!! warning "Deprecated keyword arguments"
    The keyword arguments `use_latitude_weights` and `min_cosd_lat` are
    deprecated and have been replaced by `latitude_weighting`.
"""
function SVDplusDCovariance(
    diagonal::AbstractDiagonalTerm;
    latitude_weighting = nothing,
    use_latitude_weights = nothing,
    use_weighted_samples_for_diagonal = true,
    min_cosd_lat = nothing,
    rank = nothing,
)
    latitude_weighting = _latitude_weighting(
        latitude_weighting,
        use_latitude_weights,
        min_cosd_lat,
    )
    isnothing(rank) ||
        rank >= 0 ||
        error("Rank ($rank) should be nothing or non-negative")

    return SVDplusDCovariance(
        diagonal,
        latitude_weighting,
        use_weighted_samples_for_diagonal,
        rank,
    )
end

"""
    _latitude_weighting(
        latitude_weighting,
        use_latitude_weights,
        min_cosd_lat,
    )

Return `latitude_weighting` if it is a `LatitudeWeighting`. Otherwise, return a `LatitudeWeighting` or `nothing` from `use_latitude_weights` and `min_cosd_lat`.
"""
function _latitude_weighting(
    latitude_weighting::Union{LatitudeWeighting, Nothing},
    use_latitude_weights,
    min_cosd_lat,
)
    isnothing(use_latitude_weights) &&
        isnothing(min_cosd_lat) &&
        return latitude_weighting
    isnothing(latitude_weighting) || error(
        "`latitude_weighting` cannot be combined with the deprecated keyword arguments `use_latitude_weights` and `min_cosd_lat`",
    )
    Base.depwarn(
        "The keyword arguments `use_latitude_weights` and `min_cosd_lat` are \
        deprecated. Use `latitude_weighting = LatitudeWeighting(; min_cosd_lat)` \
        for latitude weighting and `latitude_weighting = nothing` for no \
        latitude weighting.",
        :SVDplusDCovariance,
    )
    something(use_latitude_weights, false) || return nothing
    return LatitudeWeighting(min_cosd_lat = something(min_cosd_lat, 0.1))
end

"""
    _diagonal_term(model_error_scale, regularization)

Construct the diagonal term specified by `model_error_scale` and
`regularization`.
"""
function _diagonal_term(model_error_scale, regularization)
    return ModelErrorScaleDiagonal(model_error_scale) .+
           ScalarDiagonal(regularization)
end

function _diagonal_term(
    model_error_scale,
    regularization::QuantileRegularization,
)
    model_error_scale_term = ModelErrorScaleDiagonal(model_error_scale)
    return model_error_scale_term .+
           QuantileDiagonal(regularization.qtl, model_error_scale_term)
end

function covariance end

function observation end

function short_names end

function seasonally_aligned_yearly_sample_date_ranges end

function reconstruct_g end

function reconstruct_g_mean end

function reconstruct_g_mean_final end

function reconstruct_diag_cov end

function reconstruct_vars end

function reconstruct_residual end

function _get_minibatch_indices_for_nth_iteration end

end

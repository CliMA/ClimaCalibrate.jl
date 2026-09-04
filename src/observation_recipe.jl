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

Requires ClimaAnalysis and NaNStatistics to be loaded.
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
    reconstruct_vars

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

Covariance estimator that returns a multiple of the identity.

`FT1` and `FT2` are the element types of `scalar` and `min_cosd_lat`.

# Fields
- `scalar`: Scalar to multiply the identity matrix by.
- `use_latitude_weights`: Whether to apply latitude weighting.
- `min_cosd_lat`: The minimum cosine weight when using latitude weighting `[-]`.
"""
struct ScalarCovariance{FT1 <: AbstractFloat, FT2 <: AbstractFloat} <:
       AbstractCovarianceEstimator
    scalar::FT1
    use_latitude_weights::Bool
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

Covariance estimator whose diagonal is the per-season variance across the
samples of a `SampleCollection`.

`FT1`, `FT2`, and `FT3` are the element types of `model_error_scale`,
`regularization`, and `min_cosd_lat`.

# Fields
- `model_error_scale`: A model error scale term added to the diagonal of the
  covariance matrix.
- `regularization`: A regularization term added to the diagonal of the
  covariance matrix.
- `use_latitude_weights`: Whether to apply latitude weighting.
- `min_cosd_lat`: The minimum cosine weight when using latitude weighting `[-]`.
"""
struct SeasonalDiagonalCovariance{
    FT1 <: AbstractFloat,
    FT2 <: AbstractFloat,
    FT3 <: AbstractFloat,
} <: AbstractCovarianceEstimator
    model_error_scale::FT1
    regularization::FT2
    use_latitude_weights::Bool
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
`SampleCollection`, where each sample is one year of seasonal statistics.

`NaN`s are dropped when the samples are built, not here: `SampleBuilder` removes
them while flattening and requires the same coordinates to be dropped in every
sample, so a `NaN` whose position varies between samples is an error rather than
something silently ignored.

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

`FT1`, `FT2`, and `FT3` are the element types of `model_error_scale`,
`regularization`, and `min_cosd_lat`; `R` is the type of `rank`.

# Fields
- `model_error_scale`: A model error scale term added to the diagonal of the
  covariance matrix.
- `regularization`: A regularization term added to the diagonal of the
  covariance matrix, either a scalar or a
  [`QuantileRegularization`](@ref).
- `use_latitude_weights`: Whether to apply latitude weighting.
- `min_cosd_lat`: The minimum cosine weight when using latitude weighting `[-]`.
- `rank`: Rank of the singular value decomposition, or `nothing` to infer it
  from the data.
"""
struct SVDplusDCovariance{
    FT1 <: AbstractFloat,
    FT2 <: Union{AbstractFloat, QuantileRegularization},
    FT3 <: AbstractFloat,
    R <: Union{Integer, Nothing},
} <: AbstractCovarianceEstimator
    model_error_scale::FT1
    regularization::FT2
    use_latitude_weights::Bool
    min_cosd_lat::FT3
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

"""
    covariance(covar_estimator, sample_collection)

Estimate the observational noise covariance from `sample_collection`.

The result does not depend on which sample is used as the observation. See
[`ScalarCovariance`](@ref), [`SeasonalDiagonalCovariance`](@ref), and
[`SVDplusDCovariance`](@ref).

# Examples
```julia
import ClimaAnalysis, NaNStatistics
estimator = ClimaCalibrate.ObservationRecipe.SVDplusDCovariance(;
    regularization = 1e-3,
)
covar = ClimaCalibrate.ObservationRecipe.covariance(estimator, samples)
```

See also [`observation`](@ref).
"""
function covariance end

"""
    observation(covar_estimator, sample_collection, i; name, covariance)

Build an `EKP.Observation` from the `i`th sample of `sample_collection`, with a
noise covariance estimated by `covar_estimator`.

The covariance is the same for every sample in a collection, so pass a
precomputed one as `covariance` when building several observations from one
collection.

The observation carries the metadata of its samples, which is what
[`ClimaCalibrate.EnsembleBuilder`](@ref) uses to line model output up with it,
and what the `reconstruct_*` functions use to turn the flattened vectors back
into `OutputVar`s.

# Examples
```julia
import ClimaAnalysis, NaNStatistics
obs = ClimaCalibrate.ObservationRecipe.observation(estimator, samples, 1)

# For several observations from one collection, estimate the covariance once
covar = ClimaCalibrate.ObservationRecipe.covariance(estimator, samples)
observations = map(1:ClimaCalibrate.SampleBuilder.num_samples(samples)) do i
    ClimaCalibrate.ObservationRecipe.observation(
        estimator,
        samples,
        i;
        covariance = covar,
    )
end
```

See also [`covariance`](@ref), [`reconstruct_vars`](@ref).
"""
function observation end

"""
    short_names(obs)

Return the short names of the variables in an `EKP.Observation`, in the order
they were stacked.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function short_names end

"""
    seasonally_aligned_yearly_sample_date_ranges(var)

Return the `(start, stop)` date ranges that split `var` into one sample per
seasonal year, starting at December.

Pass the result to `SampleBuilder.build_samples_by_times` to build the samples
that [`SeasonalDiagonalCovariance`](@ref) expects.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function seasonally_aligned_yearly_sample_date_ranges end

"""
    reconstruct_g(ekp, iter)

Return the G ensemble matrix of iteration `iter` as a matrix of
`ClimaAnalysis.OutputVar`s, one row per variable and one column per ensemble
member.

Requires ClimaAnalysis and NaNStatistics to be loaded, and observations built by
this module.
"""
function reconstruct_g end

"""
    reconstruct_g_mean(ekp, iter)

Return the mean forward map evaluation of iteration `iter` as a vector of
`ClimaAnalysis.OutputVar`s.

Requires ClimaAnalysis and NaNStatistics to be loaded, and observations built by
this module.
"""
function reconstruct_g_mean end

"""
    reconstruct_g_mean_final(ekp)

Return the mean forward map evaluation of the last completed iteration as a
vector of `ClimaAnalysis.OutputVar`s.

Requires ClimaAnalysis and NaNStatistics to be loaded, and observations built by
this module.
"""
function reconstruct_g_mean_final end

"""
    reconstruct_diag_cov(obs)

Return the diagonal of an observation's noise covariance as a vector of
`ClimaAnalysis.OutputVar`s, so the noise can be plotted alongside the data.

Only meaningful for a diagonal covariance. Requires ClimaAnalysis and
NaNStatistics to be loaded.
"""
function reconstruct_diag_cov end

"""
    reconstruct_vars(obs)

Return the observation itself as a vector of `ClimaAnalysis.OutputVar`s.

This undoes the flattening that [`ClimaCalibrate.SampleBuilder`](@ref) applied,
so an observation can be plotted or compared against model output.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function reconstruct_vars end

function _get_minibatch_indices_for_nth_iteration end

end

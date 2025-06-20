module ObservationRecipe

export SeasonalDiagonalCovariance,
    SVDplusDCovariance,
    covariance,
    observation,
    seasonally_aligned_yearly_sample_date_ranges,
    change_data_type

import Dates

"""
    abstract type AbstractCovarianceEstimator end

An object that estimates the noise covariance matrix from observational data
that is appropriate for a sample between `start_date` and `end_date`.

`AbstractCovarianceEstimator` have to provide one function,
`ObservationRecipe.covariance`.

The function has to have the signature

```julia
ObservationRecipe.covariance(
    covar_estimator::AbstractCovarianceEstimator,
    vars,
    start_date,
    end_date,
)
```

and return a noise covariance matrix.
"""
abstract type AbstractCovarianceEstimator end

"""
    SeasonalDiagonalCovariance <: AbstractCovarianceEstimator

Contain the necessary information to construct a diagonal covariance matrix
whose entries represents seasonal covariances from `ClimaAnalysis.OutputVar`s.
"""
struct SeasonalDiagonalCovariance{FT1 <: AbstractFloat, FT2 <: AbstractFloat} <:
       AbstractCovarianceEstimator
    """A model error scale term added to the diagonal of the covariance
    matrix"""
    model_error_scale::FT1

    """A regularization term added to the diagonal of the covariance matrix"""
    regularization::FT2

    """All NaNs are ignored when computing the covariance matrix"""
    ignore_nan::Bool
end

"""
    SeasonalDiagonalCovariance(model_error_scale = 0.0,
                               regularization = 0.0,
                               ignore_nan = true)

Create a `SeasonalDiagonalCovariance` which specifies how the covariance matrix
should be formed. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `Diagonal` matrix.

Keyword arguments
=====================

- `model_error_scale`: Noise from the model error added to the covariance
  matrix. This is `(model_error_scale * seasonal_mean).^2`, where
  `seasonal_mean` is the seasonal mean for each of the quantity for each of the
  season (DJF, MAM, JJA, SON).

- `regularization`: A diagonal matrix of the form `regularization * I` is added
  to the covariance matrix.

- `ignore_nan`: If `true`, then `NaN`s are ignored when computing the covariance
  matrix. Otherwise, `NaN` are included in the intermediate calculation of the
  covariance matrix. Note that all `NaN`s are removed in the last step of
  forming the covariance matrix even if `ignore_nan` is `false`.
"""
function SeasonalDiagonalCovariance(;
    model_error_scale = 0.0,
    regularization = 0.0,
    ignore_nan = true,
)
    model_error_scale < zero(model_error_scale) &&
        error("Model_error_scale ($model_error_scale) should not be negative")
    regularization < zero(regularization) &&
        error("Regularization ($regularization) should not be negative")

    return SeasonalDiagonalCovariance(
        model_error_scale,
        regularization,
        ignore_nan,
    )
end

"""
    SVDplusDCovariance <: AbstractCovarianceEstimator

Contain the necessary information to construct a `EKP.SVDplusD` covariance
matrix from `ClimaAnalysis.OutputVar`s.
"""
struct SVDplusDCovariance{
    FT1 <: AbstractFloat,
    FT2 <: AbstractFloat,
    DATE <: Dates.AbstractDateTime,
} <: AbstractCovarianceEstimator
    """A model error scale term added to the diagonal of the covariance
    matrix"""
    model_error_scale::FT1

    """A regularization term added to the diagonal of the covariance matrix"""
    regularization::FT2

    """Tuple of the start and end dates of the samples"""
    sample_date_ranges::Vector{NTuple{2, DATE}}
end

"""
    SVDplusDCovariance(sample_date_ranges;
                       model_error_scale = 0.0,
                       regularization = 0.0,

Create a `SVDplusDCovariance` which specifies how the covariance matrix should
be formed. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `EKP.SVDplusD` covariance matrix.

!!! note "Recommended sample size"
    For `sample_date_ranges`, it is recommended that each sample contains data
    from a single year. For example, if the samples are created from time series
    data of seasonal averages, then each sample should contain all four seasons.
    Otherwise, the covariance matrix may not make sense. For example, if each
    sample contains two years of seasonally averaged data, then the sample mean
    is the seasonal mean of every other season across the years stacked
    vertically. For a concrete example, if the sample contain DJF for both 2010
    and 2011. Then, the sample mean will be of mean of DJF 2010, 2012, and so
    on, and the mean of DJF 2011, 2013, and so on. As a result, if one were to
    use this covariance matrix with `model_error_scale`, the covariance matrix
    will not make sense.

Positional arguments
=====================

- `sample_date_ranges`: The start and end dates of each samples. This is used to
  determine the sample from the time series data of the `OutputVar`s. These
  dates must be present in all the `OutputVar`s.

Keyword arguments
=====================

- `model_error_scale`: Noise from the model error added to the covariance
  matrix. This is `(model_error_scale * mean(samples, dims = 2)).^2`, where
  `mean(samples, dims = 2)` is the mean of the samples.

- `regularization`: A diagonal matrix of the form `regularization * I` is added
  to the covariance matrix.
"""
function SVDplusDCovariance(
    sample_date_ranges;
    model_error_scale = 0.0,
    regularization = 0.0,
)
    model_error_scale < zero(model_error_scale) &&
        error("Model_error_scale ($model_error_scale) should not be negative")
    regularization < zero(regularization) &&
        error("Regularization ($regularization) should not be negative")

    sample_date_ranges = [
        (Dates.DateTime(date_pair[1]), Dates.DateTime(date_pair[2])) for
        date_pair in sample_date_ranges
    ]
    for (first_date, last_date) in sample_date_ranges
        first_date <= last_date ||
            error("$first_date should not be later than $last_date")
    end
    return SVDplusDCovariance(
        model_error_scale,
        regularization,
        sample_date_ranges,
    )
end

function covariance end

function observation end

function seasonally_aligned_yearly_sample_date_ranges end

function change_data_type end

end

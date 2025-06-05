module ObservationRecipe

export SeasonalDiagonalCovariance, SVDplusDCovariance

import Dates

abstract type AbstractCovarianceEstimator end


"""
    SeasonalDiagonalCovariance <: AbstractCovarianceEstimator

Contain the necessary information to construct a diagonal covariance matrix
whose entries represents seasonal covariances from `ClimaAnalysis.OutputVar`s.
"""
struct SeasonalDiagonalCovariance{
    FT1 <: AbstractFloat,
    FT2 <: AbstractFloat,
    DT <: DataType,
} <: AbstractCovarianceEstimator
    """A model error scale term added to the diagonal of the covariance matrix"""
    model_error_scale::FT1

    """A regularization term added to the diagonal of the covariance matrix"""
    regularization::FT2

    """All NaNs are ignored when computing the covariance matrix"""
    ignore_nan::Bool

    """Float type of covariance matrix"""
    float_type::DT
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
matrix. This is `(model_error_scale * seasonal_means).^2`, where
`seasonal_means` is the seasonal mean for each of the season (DJF, MAM, JJA,
SON).

- `regularization`: A diagonal matrix of the form `regularization * I` is added
to the covariance matrix.

- `ignore_nan`: If `true`, then `NaN`s are ignored when computing the covariance
matrix. Otherwise, `NaN` are included in the intermediate calculation of the
covariance matrix. Note that all `NaN`s are removed in the last step of forming
the covariance matrix even if `ignore_nan` is `false`.

- `float_type`: The float type of the covariance matrix. The conversion to
`float_type` is done just before returning the covariance matrix.
"""
function SeasonalDiagonalCovariance(;
    model_error_scale = 0.0,
    regularization = 0.0,
    ignore_nan = true,
    float_type = Float32,
)
    model_error_scale < zero(model_error_scale) &&
        error("Model_error_scale ($model_error_scale) should not be negative")
    regularization < zero(regularization) &&
        error("Regularization ($regularization) should not be negative")

    return SeasonalDiagonalCovariance(
        model_error_scale,
        regularization,
        ignore_nan,
        float_type,
    )
end

"""
    SVDplusDCovariance <: AbstractCovarianceEstimator

Contain the necessary information to construct a `EKP.SVDplusD` covariance
matrix from `ClimaAnalysis.OutputVar`s.
"""
struct SVDplusDCovariance{ # TODO: Should change the name of this to `SVDplusD`
    FT1 <: AbstractFloat,
    FT2 <: AbstractFloat,
    DT <: DataType,
    DATE <: Dates.AbstractDateTime,
} <: AbstractCovarianceEstimator
    """A model error scale term added to the diagonal of the covariance matrix"""
    model_error_scale::FT1

    """A regularization term added to the diagonal of the covariance matrix"""
    regularization::FT2

    """Float type of covariance matrix"""
    float_type::DT

    """Tuple of the start and end dates of the samples"""
    sample_dates::Vector{NTuple{2, DATE}}
end

"""
    SVDplusDCovariance(model_error_scale = 0.0,
                       regularization = 0.0,
                       float_type = Float32,
                       sample_dates = nothing)

Create a `SVDplusDCovariance` which specifies how the covariance matrix should
be formed. When used with `ObservationRecipe.observation` or `ObservationRecipe.covariance`,
return a `EKP.SVDplusD` covariance matrix.

Positional arguments
=====================

- `sample_dates`: The start and end dates of each samples. This is used to
determine the sample from the time series data of the `OutputVar`s. These dates
must be present in all the `OutputVar`s.

Keyword arguments
=====================

- `model_error_scale`: Noise from the model error added to the covariance
matrix. This is `(model_error_scale * mean(samples, dims = 2)).^2`, where
`mean(samples, dims = 2)` is the mean of the samples.

- `regularization`: A diagonal matrix of the form `regularization * I` is added
  to the covariance matrix.

- `float_type`: The float type of the covariance matrix. The conversion to
`float_type` is first done to the samples.
"""
function SVDplusDCovariance(
    sample_dates;
    model_error_scale = 0.0,
    regularization = 0.0,
    float_type = Float32,
)
    model_error_scale < zero(model_error_scale) &&
        error("Model_error_scale ($model_error_scale) should not be negative")
    regularization < zero(regularization) &&
        error("Regularization ($regularization) should not be negative")

    sample_dates = [
        (Dates.DateTime(date_pair[1]), Dates.DateTime(date_pair[2])) for
        date_pair in sample_dates
    ]
    for (first_date, last_date) in sample_dates
        first_date <= last_date ||
            error("$first_date should not be later than $last_date")
    end
    return SVDplusDCovariance(
        model_error_scale,
        regularization,
        float_type,
        sample_dates,
    )
end

function covariance end

function observation end

end

"""
    ClimaCalibrate.ObservationRecipe

Estimate a noise covariance from a `SampleCollection` and build an
`EKP.Observation` from it.

Two estimators are available: [`ScalarCovariance`](@ref) for a multiple of the
identity and [`SVDplusDCovariance`](@ref) for a low-rank sample covariance plus
a diagonal term. Both take samples built by
[`ClimaCalibrate.SampleBuilder`](@ref).

Also reconstructs the flattened vectors back into `OutputVar`s
([`reconstruct_vars`](@ref), [`reconstruct_g`](@ref)), so a calibration's
observations and forward map output can be inspected.

Requires ClimaAnalysis to be loaded.
"""
module ObservationRecipe

export ScalarCovariance,
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

An object that estimates the noise covariance matrix from the samples in an
`AbstractSampleCollection`.

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

`sample_collection` may be a `TransformedSampleCollection`. Call
`SampleBuilder.apply_transform` on it to get the transformed samples, or
`SampleBuilder.base` to get the untransformed samples.

Subtypes:
- [`ScalarCovariance`](@ref): a multiple of the identity.
- [`SVDplusDCovariance`](@ref): a low-rank sample covariance plus a diagonal
  term.
"""
abstract type AbstractCovarianceEstimator end

"""
    ScalarCovariance <: AbstractCovarianceEstimator

Covariance estimator that contains the necessary information to construct the
scalar covariance matrix.
"""
struct ScalarCovariance{
    FT <: AbstractFloat,
    L <: Union{Nothing, LatitudeWeightDiagonal},
} <: AbstractCovarianceEstimator
    """Scalar to multiply the identity matrix by"""
    scalar::FT

    """The latitude weighting applied to the diagonal of the covariance matrix,
    or `nothing` for no latitude weighting"""
    latitude_weighting::L
end

"""
    ScalarCovariance(;
        scalar = 1.0,
        latitude_weighting = nothing,
    )

Create a `ScalarCovariance` which specifies how the covariance matrix should be
formed. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `Diagonal` matrix.

If the sample collection is a `TransformedSampleCollection`, an error is thrown.

# Keyword Arguments

- `scalar`: Scalar value to multiply the identity matrix by.

- `latitude_weighting`: A [`LatitudeWeightDiagonal`](@ref) that describes the
  latitude weighting applied to the covariance matrix, or `nothing` for no
  latitude weighting. Latitude weighting multiplies the diagonal entries of the
  weighted variables by `1 / max(cosd(lat), min_cosd_lat)` and leaves the other
  entries unchanged, where `min_cosd_lat` is the keyword argument of
  `LatitudeWeightDiagonal` that caps the weight toward the poles.
"""
function ScalarCovariance(;
    scalar = 1.0,
    latitude_weighting::Union{Nothing, LatitudeWeightDiagonal} = nothing,
)
    if scalar <= zero(scalar)
        error("The value for scalar ($scalar) should be positive")
    end

    return ScalarCovariance(scalar, latitude_weighting)
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
    R <: Union{Integer, Nothing},
} <: AbstractCovarianceEstimator
    """A diagonal term that describes the diagonal matrix added to the low rank
    approximation of the covariance matrix"""
    diagonal::D

    """Rank of the singular value decomposition, or `nothing` to infer it from
    the data"""
    rank::R
end

"""
    SVDplusDCovariance(
        diagonal::AbstractDiagonalTerm;
        rank = nothing,
    )

Create a `SVDplusDCovariance` whose diagonal matrix is described by the diagonal
term `diagonal`. When used with `ObservationRecipe.observation` or
`ObservationRecipe.covariance`, return a `EKP.SVDplusD` covariance matrix.

# Keyword Arguments

- `rank`: Rank of the singular value decomposition (SVD). If `nothing` is passed
  in, then the rank is automatically inferred from the data.

!!! note "Latitude weighting"
    To apply latitude weighting, pipe the sample collection into a
    [`LatitudeWeighting`](@ref ClimaCalibrate.SampleBuilder.LatitudeWeighting)
    before passing it to `observation`. Both the SVD and the diagonal term are
    computed from the weighted samples. Passing a diagonal term that contains a
    `LatitudeWeightDiagonal` throws an error, since it would weight the diagonal
    term but not the SVD.

    ```julia
    covar_estimator = SVDplusDCovariance(ModelErrorScaleDiagonal(0.05))
    weighted_collection = sample_collection |> SampleBuilder.LatitudeWeighting()
    obs = observation(covar_estimator, weighted_collection, 1)
    ```
"""
function SVDplusDCovariance(diagonal::AbstractDiagonalTerm; rank = nothing)
    _contains_term(diagonal, LatitudeWeightDiagonal) && error(
        "SVDplusDCovariance does not support a LatitudeWeightDiagonal in its diagonal term. To apply latitude weighting, pipe the sample collection into a SampleBuilder.LatitudeWeighting instead",
    )
    isnothing(rank) ||
        rank >= 0 ||
        error("Rank ($rank) should be nothing or non-negative")

    return SVDplusDCovariance(diagonal, rank)
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

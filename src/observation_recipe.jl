"""
    ClimaCalibrate.ObservationRecipe

Estimate a noise covariance from a `SampleCollection` and build an
`EKP.Observation` from it.

Three estimators are available: [`DiagonalCovariance`](@ref) for a diagonal
matrix, [`SVDplusDCovariance`](@ref) for a low-rank sample covariance plus a
diagonal term, and [`SVDplusModelErrorCovariance`](@ref) for a low-rank sample
covariance plus the residual variance and a model error term. All take samples
built by [`ClimaCalibrate.SampleBuilder`](@ref).

Also reconstructs the flattened vectors back into `OutputVar`s
([`reconstruct_vars`](@ref), [`reconstruct_g`](@ref)), so a calibration's
observations and forward map output can be inspected.

Requires ClimaAnalysis to be loaded.
"""
module ObservationRecipe

export DiagonalCovariance,
    SVDplusDCovariance,
    SVDplusModelErrorCovariance,
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

import LinearAlgebra

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
- [`DiagonalCovariance`](@ref): a diagonal matrix described by a diagonal term
  or given directly as a `Diagonal` matrix or a `UniformScaling`.
- [`SVDplusDCovariance`](@ref): a low-rank sample covariance plus a diagonal
  term.
- [`SVDplusModelErrorCovariance`](@ref): a low-rank sample covariance plus the
  residual variance and a model error term.
"""
abstract type AbstractCovarianceEstimator end

"""
    DiagonalCovariance <: AbstractCovarianceEstimator

Covariance estimator that returns a diagonal covariance matrix.

The matrix is described by a diagonal term, a subtype of
[`AbstractDiagonalTerm`](@ref), or given directly as a `Diagonal` matrix with
one entry per row of the samples or as a `UniformScaling`. When used with
`ObservationRecipe.covariance`, return a `Diagonal` matrix, or a
`UniformScaling` if one is given. The element type is converted to the element
type of the samples.

An error is thrown if any diagonal entry is not finite or not positive. The
diagonal of a covariance matrix holds variances, and a zero or `NaN` variance
leaves the matrix singular.

If the sample collection is a `TransformedSampleCollection`, an error is
thrown. To latitude weight the covariance matrix, multiply the diagonal term by
a [`LatitudeWeightDiagonal`](@ref).

# Example

```julia
DiagonalCovariance(
    VarianceDiagonal() .* LatitudeWeightDiagonal() .+ ScalarDiagonal(1e-6),
)
```

In the example, the entries are the latitude weighted variance of each entry
across the samples, with `1e-6` added to keep them positive.

!!! tip "Scalar covariance with latitude weighting"
    For a multiple of the identity matrix, use a [`ScalarDiagonal`](@ref). To
    latitude weight it, multiply it by a `LatitudeWeightDiagonal`.

    ```julia
    DiagonalCovariance(ScalarDiagonal(2.0) .* LatitudeWeightDiagonal())
    ```

    The entries are `2.0` times the latitude weights. The entries of variables
    without a latitude dimension are `2.0`.
"""
struct DiagonalCovariance{
    D <: Union{
        AbstractDiagonalTerm,
        LinearAlgebra.Diagonal,
        LinearAlgebra.UniformScaling,
    },
} <: AbstractCovarianceEstimator
    """A diagonal term that describes the covariance matrix, or the `Diagonal`
    or `UniformScaling` covariance matrix itself"""
    diagonal::D
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

    function SVDplusDCovariance(
        diagonal::D,
        rank::R,
    ) where {D <: AbstractDiagonalTerm, R <: Union{Integer, Nothing}}
        _contains_term(diagonal, LatitudeWeightDiagonal) && error(
            "SVDplusDCovariance does not support a LatitudeWeightDiagonal in its diagonal term. To apply latitude weighting, pipe the sample collection into a SampleBuilder.LatitudeWeighting instead",
        )
        isnothing(rank) ||
            rank >= 0 ||
            error("Rank ($rank) should be nothing or non-negative")
        _contains_term(
            diagonal,
            term -> term isa ResidualVarianceDiagonal && term.rank != rank,
        ) && error(
            "The rank of every ResidualVarianceDiagonal in the diagonal term must match the rank of SVDplusDCovariance ($rank), so that the diagonal subtracts the modes the SVD keeps",
        )
        return new{D, R}(diagonal, rank)
    end
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
    return SVDplusDCovariance(diagonal, rank)
end

"""
    SVDplusModelErrorCovariance <: AbstractCovarianceEstimator

Covariance estimator that returns an `EKP.SVDplusD` that is a low-rank sample
covariance plus a diagonal of the residual variance and a model error term.
"""
struct SVDplusModelErrorCovariance{S <: SVDplusDCovariance} <:
       AbstractCovarianceEstimator
    """The covariance estimator that computes the covariance matrix"""
    inner::S
end

@doc raw"""
    SVDplusModelErrorCovariance(;
        rank = 2,
        beta = 0.05,
        sigma = 1e-3,
    )

Create a `SVDplusModelErrorCovariance`. When used with
`ObservationRecipe.observation` or `ObservationRecipe.covariance`, return an
`EKP.SVDplusD` covariance matrix

```math
\mathrm{Cov}_{\mathrm{tsvd}}(y_1, \ldots, y_n) + D_y + D_{\mathrm{diff}},
```

where ``y_1, \ldots, y_n`` are the samples and

- ``\mathrm{Cov}_{\mathrm{tsvd}}(y_1, \ldots, y_n)`` is the truncated SVD of
  rank `rank` of the sample covariance matrix and is the low-rank part of
  [`SVDplusDCovariance`](@ref),
- ``D_y`` is the residual variance and is the term
  [`ResidualVarianceDiagonal(rank)`](@ref ResidualVarianceDiagonal), and
- ``D_{\mathrm{diff}}`` is the model error and is the term

  ```julia
  ScalarDiagonal(beta) .^ 2 .* max.(
      ModelErrorScaleDiagonal(1.0),
      sigma^2 .* QuantileDiagonal(0.5, VarianceDiagonal()),
  )
  ```

Without ``D_{\mathrm{diff}}``, the diagonal of the covariance matrix is the
sample variance for any `rank`, since ``D_y`` adds back the variance that the
truncated SVD drops. In the model error term, the `max` floors the squared mean
at `sigma^2` times the median variance of the variable, so the model error does
not vanish when the entries of `ModelErrorScaleDiagonal(1.0)` are near zero.

# Keyword Arguments

- `rank`: Rank of the truncated SVD.

- `beta`: The model error of each variable relative to the mean, so `beta =
  0.05` is a model error with a standard deviation of 5% of the mean, the same
  as `ModelErrorScaleDiagonal(0.05)`. If `beta` is a number, then the same value
  is used for all variables. Otherwise, `beta[i]` is used for the `i`th
  variable.

- `sigma`: The floor of the squared mean is `sigma^2` times the median variance
  of each variable. The floor keeps the covariance matrix invertible and is not
  an estimate of the uncertainty. Since the floor is scaled by `beta^2`, a
  variable whose `beta` is zero has no floor.

When the covariance matrix is computed, an error is thrown if the number of
values in `beta` is neither one nor the number of variables, or if the median
variance of a variable is zero, which happens when most of the entries of the
variable do not vary across the samples.

!!! note "Latitude weighting"
    To apply latitude weighting, pipe the sample collection into a
    [`LatitudeWeighting`](@ref ClimaCalibrate.SampleBuilder.LatitudeWeighting)
    before passing it to `observation`. Every term of the covariance matrix is
    computed from the weighted samples.

    ```julia
    covar_estimator = SVDplusModelErrorCovariance(beta = 0.05)
    weighted_collection = sample_collection |> SampleBuilder.LatitudeWeighting()
    obs = observation(covar_estimator, weighted_collection, 1)
    ```
"""
function SVDplusModelErrorCovariance(; rank::Int = 2, beta = 0.05, sigma = 1e-3)
    sigma >= zero(sigma) || error("Sigma ($sigma) should not be negative")
    # The floor only replaces the squared mean where it is smaller, so the model
    # error term does not vanish where the mean of the variable is zero
    floor_term = sigma^2 .* QuantileDiagonal(0.5, VarianceDiagonal())
    diagonal =
        ResidualVarianceDiagonal(rank) .+
        ScalarDiagonal(float.(vcat(beta))) .^ 2 .*
        max.(ModelErrorScaleDiagonal(1.0), floor_term)
    return SVDplusModelErrorCovariance(SVDplusDCovariance(diagonal; rank))
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

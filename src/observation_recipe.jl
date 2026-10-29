"""
    ClimaCalibrate.ObservationRecipe

Estimate a noise covariance from a `SampleCollection` and build an
`EKP.Observation` from it.

Three estimators are available: [`ScalarCovariance`](@ref) for a multiple of the
identity, [`SVDplusDCovariance`](@ref) for a low-rank sample covariance plus a
diagonal term, and [`ToBeDecidedCovariance`](@ref) for a low-rank sample
covariance plus the residual variance and a model error term. All take samples
built by [`ClimaCalibrate.SampleBuilder`](@ref).

Also reconstructs the flattened vectors back into `OutputVar`s
([`reconstruct_vars`](@ref), [`reconstruct_g`](@ref)), so a calibration's
observations and forward map output can be inspected.

Requires ClimaAnalysis to be loaded.
"""
module ObservationRecipe

export ScalarCovariance,
    SVDplusDCovariance,
    ToBeDecidedCovariance,
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
- [`ToBeDecidedCovariance`](@ref): a low-rank sample covariance plus the
  residual variance and a model error term.
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
    _contains_term(
        diagonal,
        term -> term isa ResidualVarianceDiagonal && term.rank != rank,
    ) && error(
        "The rank of every ResidualVarianceDiagonal in the diagonal term must match the rank of SVDplusDCovariance ($rank), so that the diagonal subtracts the modes the SVD keeps",
    )

    return SVDplusDCovariance(diagonal, rank)
end

"""
    ToBeDecidedCovariance <: AbstractCovarianceEstimator

Covariance estimator that returns an `EKP.SVDplusD` that is a low-rank sample
covariance plus a diagonal of the residual variance and a model error term.
"""
struct ToBeDecidedCovariance{S <: SVDplusDCovariance} <:
       AbstractCovarianceEstimator
    """The covariance estimator that computes the covariance matrix"""
    inner::S
end

@doc raw"""
    ToBeDecidedCovariance(;
        rank = 2,
        beta = 0.05,
        sigma = 1e-3,
    )

Create a `ToBeDecidedCovariance`. When used with `ObservationRecipe.observation`
or `ObservationRecipe.covariance`, return an `EKP.SVDplusD` covariance matrix

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
    covar_estimator = ToBeDecidedCovariance(beta = 0.05)
    weighted_collection = sample_collection |> SampleBuilder.LatitudeWeighting()
    obs = observation(covar_estimator, weighted_collection, 1)
    ```
"""
function ToBeDecidedCovariance(; rank::Int = 2, beta = 0.05, sigma = 1e-3)
    sigma >= zero(sigma) || error("Sigma ($sigma) should not be negative")
    # The floor only replaces the squared mean where it is smaller, so the model
    # error term does not vanish where the mean of the variable is zero
    floor_term = sigma^2 .* QuantileDiagonal(0.5, VarianceDiagonal())
    diagonal =
        ResidualVarianceDiagonal(rank) .+
        ScalarDiagonal(float.(vcat(beta))) .^ 2 .*
        max.(ModelErrorScaleDiagonal(1.0), floor_term)
    return ToBeDecidedCovariance(SVDplusDCovariance(diagonal; rank))
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

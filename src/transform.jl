export AbstractTransform,
    AbstractWeighting,
    LatitudeWeighting,
    PerVariableWeighting,
    PerCollectionWeighting,
    apply_transform,
    apply_transform!,
    compute_weights,
    transform_sequence

"""
    AbstractTransform

Represent a transform done on an `AbstractSampleCollection`.

Transforms are lazy because an observation consists of untransformed samples,
metadata, and a covariance matrix generated from transformed samples.

# Interface

To define a new `AbstractTransform`, your subtype must implement
[`SampleBuilder.apply_transform!`](@ref). If the transform multiplies the
samples by weights, subtype [`AbstractWeighting`](@ref) and implement
[`SampleBuilder.compute_weights`](@ref) instead. Every `AbstractTransform` can
then be called on an `AbstractSampleCollection` to create a
`TransformedSampleCollection`, and works with
[`SampleBuilder.apply_transform`](@ref), which is implemented in terms of
[`SampleBuilder.apply_transform!`](@ref).
"""
abstract type AbstractTransform end

"""
    TransformedSampleCollection

An object that keeps the [`AbstractSampleCollection`](@ref) and records a sequence
of `AbstractTransform`s applied to it.

Transforms are not immediately applied since an observation consists of
untransformed samples, metadata, and a covariance matrix generated from
transformed samples.
"""
struct TransformedSampleCollection{
    ASC <: AbstractSampleCollection,
    T <: AbstractTransform,
} <: AbstractSampleCollection
    parent::ASC
    transform::T
end

function apply_transform end

"""
    apply_transform!

Apply a transform to a `SampleCollection` and return the transformed
`SampleCollection`.

Every `AbstractTransform` that is not an [`AbstractWeighting`](@ref) must
implement its own `apply_transform!`. An implementation may mutate the
`SampleCollection`, but it must return the transformed `SampleCollection` with
the metadata unchanged. An `AbstractWeighting` implements
[`SampleBuilder.compute_weights`](@ref) instead.
"""
function apply_transform! end

function transform_sequence end

"""
    AbstractWeighting <: AbstractTransform

A transform that multiplies the matrix of samples by positive weights.

# Interface

To define a new `AbstractWeighting`, your subtype must implement
[`SampleBuilder.compute_weights`](@ref). Its
[`SampleBuilder.apply_transform!`](@ref) is already defined and multiplies the
samples by the weights, so the weighting works like the built-in ones.

# Examples

A weighting that multiplies the samples of the `i`th variable by `i`.

```julia
import ClimaCalibrate.SampleBuilder

struct IndexWeighting <: SampleBuilder.AbstractWeighting end

function SampleBuilder.compute_weights(::IndexWeighting, sample_collection)
    ranges = SampleBuilder.var_indices(sample_collection)
    # One weight per entry of a sample
    return [Float64(i) for (i, rows) in enumerate(ranges) for _ in rows]
end

weighted = sample_collection |> IndexWeighting()
```
"""
abstract type AbstractWeighting <: AbstractTransform end

"""
    compute_weights(weighting::AbstractWeighting, sample_collection)

Return the weights that `weighting` multiplies the matrix of samples of the
`SampleCollection` `sample_collection` by.

The weights are either a `Real`, which multiplies every entry, or an
`AbstractVector` of `Real`s with one weight per entry of a sample, that is, one
per row of the matrix of samples. Every weight must be positive. Use
[`SampleBuilder.var_indices`](@ref) to find the rows that belong to each
variable.

Every [`AbstractWeighting`](@ref) must implement `compute_weights`, and it must
not modify `sample_collection`.
"""
function compute_weights end

"""
    LatitudeWeighting <: AbstractWeighting

A transform that applies latitude weighting to the matrix of samples.

Latitude weighting multiplies each sample value at latitude `lat` by
`sqrt(1 / max(cosd(lat), min_cosd_lat))` if the latitude exists. The variances
are multiplied by `1 / max(cosd(lat), min_cosd_lat)`, so values toward the poles
have less influence on the calibration.
"""
struct LatitudeWeighting{
    S <: Union{Nothing, AbstractSet},
    FT <: AbstractFloat,
    B,
} <: AbstractWeighting
    min_cosd_lat::FT
    by::B
    selected::S
    function LatitudeWeighting(
        min_cosd_lat::FT,
        by::B,
        selected::S,
    ) where {FT <: AbstractFloat, B, S <: Union{Nothing, AbstractSet}}
        min_cosd_lat > zero(FT) || error(
            "The value for min_cosd_lat ($min_cosd_lat) should be greater than zero",
        )
        isnothing(by) == isnothing(selected) || error(
            "`by` and `selected` must either both be given or both be omitted",
        )
        return new{S, FT, B}(min_cosd_lat, by, selected)
    end
end

"""
    LatitudeWeighting(; min_cosd_lat::AbstractFloat = 0.1)

Return a latitude weighting transform that applies latitude weighting wherever
possible.
"""
function LatitudeWeighting(; min_cosd_lat::AbstractFloat = 0.1)
    return LatitudeWeighting(min_cosd_lat, nothing, nothing)
end

"""
    PerVariableWeighting <: AbstractWeighting

A transform that applies per-variable weighting to the matrix of samples.
"""
struct PerVariableWeighting{
    W <: Union{AbstractVector{<:Real}, AbstractDict{<:Any, <:Real}},
    B,
} <: AbstractWeighting
    weights::W
    by::B
    function PerVariableWeighting(
        weights::W,
        by::B,
    ) where {W <: Union{AbstractVector{<:Real}, AbstractDict{<:Any, <:Real}}, B}
        (weights isa AbstractDict) == !isnothing(by) || error(
            "`by` must be given for a dictionary of weights and omitted for a vector of weights",
        )
        return new{W, B}(weights, by)
    end
end

"""
    PerVariableWeighting(weights::AbstractVector{<:Real})

Return a per-variable weighting transform that weights the samples of the `i`th
variable by the `i`th weight.
"""
function PerVariableWeighting(weights::AbstractVector{<:Real})
    return PerVariableWeighting(weights, nothing)
end

"""
    PerCollectionWeighting <: AbstractWeighting

A transform that applies the same weight to the matrix of samples.

# Example

We apply a weight of 3.0 to all samples which is the same as multiplying the
sample values by 3.0.

```julia
transform = PerCollectionWeighting(3.0)
weighted = sample_collection |> transform
```
"""
struct PerCollectionWeighting{FT <: AbstractFloat} <: AbstractWeighting
    weight::FT
end

"""
    show(io::IO, transform::LatitudeWeighting)

Show the `min_cosd_lat` from `transform` and if `transform` only weights
selected variables, then also show `by` and `selected`.
"""
function Base.show(io::IO, transform::LatitudeWeighting)
    print(io, "LatitudeWeighting(")
    if !isnothing(transform.selected)
        show(io, transform.selected)
        print(io, "; by = ")
        show(io, transform.by)
        print(io, ", ")
    end
    print(io, "min_cosd_lat = ")
    show(io, transform.min_cosd_lat)
    print(io, ")")
    return nothing
end

"""
    show(io::IO, transform::PerVariableWeighting)

Show the `weights` from `transform` and if `transform` uses a dictionary of
weights, then also show `by`.
"""
function Base.show(io::IO, transform::PerVariableWeighting)
    print(io, "PerVariableWeighting(")
    show(io, transform.weights)
    if !isnothing(transform.by)
        print(io, "; by = ")
        show(io, transform.by)
    end
    print(io, ")")
    return nothing
end

"""
    show(io::IO, transform::PerCollectionWeighting)

Show the weight from `transform` applied to all samples.
"""
function Base.show(io::IO, transform::PerCollectionWeighting)
    print(io, "PerCollectionWeighting(")
    show(io, transform.weight)
    print(io, ")")
    return nothing
end

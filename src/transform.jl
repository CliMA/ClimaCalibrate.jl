export AbstractTransform,
    LatitudeWeighting,
    PerVariableWeighting,
    PerCollectionWeighting,
    apply_transform,
    apply_transform!,
    transform_sequence

"""
    AbstractTransform

Represent a transform done on an `AbstractSampleCollection`.

Transforms are lazy because an observation consists of untransformed samples,
metadata, and a covariance matrix generated from transformed samples.

# Interface

To define a new `AbstractTransform`, your subtype must implement
[`SampleBuilder.apply_transform!`](@ref). By default, all `AbstractTransform`s
can be called on an `AbstractSampleCollection` to create a
`TransformedSampleCollection` and [`SampleBuilder.apply_transform`](@ref) which
is implemented in terms of [`SampleBuilder.apply_transform!`](@ref).

# Examples

A transform that zeroes out the values of the samples.

```julia
import ClimaCalibrate.SampleBuilder

struct ZeroTransform <: SampleBuilder.AbstractTransform end

function SampleBuilder.apply_transform!(transform::ZeroTransform, sample_collection)
    SampleBuilder.get_samples(sample_collection) .*= 0.0
    return sample_collection
end

weighted = sample_collection |> ZeroTransform()
```
"""
abstract type AbstractTransform end

function apply_transform end

"""
    apply_transform!

Apply a transform in-place on a `SampleCollection` and return the
`SampleCollection`.

All `AbstractTransform`s must implement their own `apply_transform!`.
"""
function apply_transform! end

function transform_sequence end

"""
    LatitudeWeighting <: AbstractTransform

A transform that applies latitude weighting to the matrix of samples.

Latitude weighting multiplies each sample value at latitude `lat` by
`sqrt(1 / max(cosd(lat), min_cosd_lat))` if the latitude exists.
"""
struct LatitudeWeighting{
    S <: Union{Nothing, AbstractSet},
    FT <: AbstractFloat,
    B,
} <: AbstractTransform
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
    LatitudeWeighting(; min_cosd_lat = 0.1)

Return a latitude weighting transform that applies latitude weighting wherever
possible.
"""
function LatitudeWeighting(; min_cosd_lat = 0.1)
    return LatitudeWeighting(min_cosd_lat, nothing, nothing)
end

"""
    PerVariableWeighting <: AbstractTransform

A transform that applies per-variable weighting to the matrix of samples.
"""
struct PerVariableWeighting{
    W <: Union{AbstractVector{<:Real}, AbstractDict{<:Any, <:Real}},
    B,
} <: AbstractTransform
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
    PerCollectionWeighting <: AbstractTransform

A transform that applies the same weight to the matrix of samples.

# Example

We apply a weight of 3.0 to all samples which is the same as multiplying the
sample values by 3.0.

```julia
transform = PerCollectionWeighting(3.0)
weighted = sample_collection |> transform
```
"""
struct PerCollectionWeighting{FT <: AbstractFloat} <: AbstractTransform
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

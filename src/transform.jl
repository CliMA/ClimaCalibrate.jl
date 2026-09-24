export AbstractWeighting,
    LatitudeWeighting,
    PerVariableWeighting,
    PerCollectionWeighting,
    apply_transform,
    apply_transform!,
    compute_weights,
    transform_sequence

"""
    AbstractWeighting

Represent a transform done on an `AbstractSampleCollection` that multiplies the
matrix of samples by positive weights.

Transforms are lazy because an observation consists of untransformed samples,
metadata, and a covariance matrix generated from transformed samples.

Every `AbstractWeighting` can be called on an `AbstractSampleCollection` to
create a `TransformedSampleCollection`, and works with
[`SampleBuilder.apply_transform`](@ref) and
[`SampleBuilder.apply_transform!`](@ref).
"""
abstract type AbstractWeighting end

"""
    TransformedSampleCollection

An object that keeps the [`AbstractSampleCollection`](@ref) and records a sequence
of `AbstractWeighting`s applied to it.

Transforms are not immediately applied since an observation consists of
untransformed samples, metadata, and a covariance matrix generated from
transformed samples.
"""
struct TransformedSampleCollection{
    ASC <: AbstractSampleCollection,
    T <: AbstractWeighting,
} <: AbstractSampleCollection
    parent::ASC
    transform::T
end

function apply_transform end

"""
    apply_transform!

Apply a transform to a `SampleCollection` in place and return the transformed
`SampleCollection`.
"""
function apply_transform! end

function transform_sequence end

"""
    compute_weights(weighting::AbstractWeighting, sample_collection)

Return the weights that `weighting` multiplies the matrix of samples of the
`SampleCollection` `sample_collection` by.

The weights are either an `AbstractFloat`, which multiplies every entry, or an
`AbstractVector` of `AbstractFloat`s with one weight per entry of a sample, that
is, one per row of the matrix of samples. Every weight must be positive. Use
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

An error is thrown if `min_cosd_lat` is not greater than zero. When the
transform is applied, an error is thrown if

- none of the variables have a latitude dimension,
- the latitudes of a weighted variable are not the same across samples, or
- the latitude dimension of a weighted variable does not have units of degrees.
"""
function LatitudeWeighting(; min_cosd_lat::AbstractFloat = 0.1)
    return LatitudeWeighting(min_cosd_lat, nothing, nothing)
end

"""
    PerVariableWeighting <: AbstractWeighting

A transform that applies per-variable weighting to the matrix of samples.
"""
struct PerVariableWeighting{
    W <: Union{
        AbstractVector{<:AbstractFloat},
        AbstractDict{<:Any, <:AbstractFloat},
    },
    B,
} <: AbstractWeighting
    weights::W
    by::B
    function PerVariableWeighting(
        weights::W,
        by::B,
    ) where {
        W <: Union{
            AbstractVector{<:AbstractFloat},
            AbstractDict{<:Any, <:AbstractFloat},
        },
        B,
    }
        (weights isa AbstractDict) == !isnothing(by) || error(
            "`by` must be given for a dictionary of weights and omitted for a vector of weights",
        )
        return new{W, B}(weights, by)
    end
end

"""
    PerVariableWeighting(weights::AbstractVector{<:AbstractFloat})

Return a per-variable weighting transform that weights the samples of the `i`th
variable by the `i`th weight.

When the transform is applied, an error is thrown if the number of weights is
not the same as the number of variables or if a weight is not positive.
"""
function PerVariableWeighting(weights::AbstractVector{<:AbstractFloat})
    return PerVariableWeighting(weights, nothing)
end

"""
    PerCollectionWeighting <: AbstractWeighting

A transform that applies the same weight to the matrix of samples.

When the transform is applied, an error is thrown if the weight is not positive.

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

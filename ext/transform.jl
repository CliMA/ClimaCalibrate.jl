import ClimaCalibrate.SampleBuilder:
    AbstractSampleCollection,
    AbstractTransform,
    LatitudeWeighting,
    PerCollectionWeighting,
    PerVariableWeighting,
    apply_transform,
    apply_transform!,
    base,
    var_indices

"""
    TransformedSampleCollection

An object that keeps the [`AbstractSampleCollection`](@ref) and records a sequence
of `AbstractTransform`s applied to it.

Transforms are not immediately applied since an observation consists of
untransformed samples, metadata, and a covariance matrix generated from
transformed samples.
"""
struct TransformedSampleCollection{
    P <: AbstractSampleCollection,
    T <: AbstractTransform,
} <: AbstractSampleCollection
    parent::P
    transform::T
end

"""
    (transform::AbstractTransform)(sample_collection::AbstractSampleCollection)

Lazily apply `transform` to `sample_collection`, returning a
`TransformedSampleCollection` that is evaluated with `apply_transform`.
"""
function (transform::AbstractTransform)(
    sample_collection::AbstractSampleCollection,
)
    return TransformedSampleCollection(sample_collection, transform)
end

"""
    base(sample_collection::SampleCollection)

Return `sample_collection`.
"""
function SampleBuilder.base(sample_collection::SampleCollection)
    return sample_collection
end

"""
    base(sample_collection::TransformedSampleCollection)

Return the `SampleCollection` underlying `sample_collection` with no
transformations applied to it.
"""
function SampleBuilder.base(sample_collection::TransformedSampleCollection)
    return base(sample_collection.parent)
end

"""
    apply_transform(sample_collection::SampleCollection)

Return `sample_collection` as no transforms are applied.

The returned `sample_collection` is not a copy of the input.

This method exists so that `apply_transform` on an `AbstractSampleCollection`
always returns a `SampleCollection`.

See also [`SampleBuilder.apply_transform(::TransformedSampleCollection)`](@ref).
"""
function SampleBuilder.apply_transform(sample_collection::SampleCollection)
    return sample_collection
end

"""
    apply_transform(sample_collection::TransformedSampleCollection)
"""
function SampleBuilder.apply_transform(
    sample_collection::TransformedSampleCollection,
)
    return _apply_transform(sample_collection)
end

"""
    _apply_transform(
        sample_collection::Union{SampleCollection, TransformedSampleCollection}
    )

Recursively apply the transforms recorded by `TransformedSampleCollection` to a
`SampleCollection`.

The base case is `_apply_transform(::SampleCollection)` which makes a copy, so
`_apply_transform(::TransformedSampleCollection)` can apply the transform
in-place.
"""
_apply_transform(sample_collection::SampleCollection) = SampleCollection(
    copy(get_samples(sample_collection)),
    get_metadata(sample_collection),
)
_apply_transform(sample_collection::TransformedSampleCollection) =
    apply_transform!(
        sample_collection.transform,
        _apply_transform(sample_collection.parent),
    )

"""
    apply_transform(
        transform::AbstractTransform,
        sample_collection::AbstractSampleCollection,
    )

Eagerly apply a transform to `sample_collection`.

This is implemented in terms of `apply_transform!`.
"""
function SampleBuilder.apply_transform(
    transform::AbstractTransform,
    sample_collection::AbstractSampleCollection,
)
    return apply_transform(transform(sample_collection))
end

"""
    num_samples(sample_collection::TransformedSampleCollection)

Return the number of samples in `sample_collection`.
"""
function SampleBuilder.num_samples(
    sample_collection::TransformedSampleCollection,
)
    num_samples(base(sample_collection))
end


"""
    LatitudeWeighting(
        selected::Union{AbstractVector, AbstractSet, Tuple};
        by = ClimaAnalysis.short_name,
        min_cosd_lat::AbstractFloat = 0.1,
    )

Return a latitude weighting transform that only weights the variables whose key
is in `selected`.

The key of a variable is `by(metadata)`, where `metadata` is the
`ClimaAnalysis.Var.Metadata` of that variable. An error is thrown if a selected
variable does not have a latitude dimension or if no variable is selected.

# Example

Weight only the variables with the short names `pr` and `tas`, leaving the
other variables in the sample collection unweighted:

```julia
transform = LatitudeWeighting(["pr", "tas"])
weighted = sample_collection |> transform
```
"""
function LatitudeWeighting(
    selected::Union{AbstractVector, AbstractSet, Tuple};
    by = ClimaAnalysis.short_name,
    min_cosd_lat::AbstractFloat = 0.1,
)
    return LatitudeWeighting(min_cosd_lat, by, Set(selected))
end

"""
    apply_transform!(
        transform::LatitudeWeighting,
        sample_collection::SampleCollection,
    )

Apply latitude weighting to `sample_collection` according to `transform`.

An error is thrown if
- no variable is weighted, either because none of the variables have a latitude
  dimension or because none of the variables are selected,
- a selected variable does not have a latitude dimension,
- the latitudes of a weighted variable are not the same across samples, or
- the latitude dimension of a weighted variable is not in degrees.
"""
function SampleBuilder.apply_transform!(
    transform::LatitudeWeighting,
    sample_collection::SampleCollection,
)
    metadata = get_metadata(sample_collection)
    metadata_col = _metadata_of_first_sample(sample_collection)
    mask = _lat_weighting_mask(transform, metadata_col)
    _check_lats_across_samples(metadata[mask, :])
    samples = get_samples(sample_collection)
    (; min_cosd_lat) = transform
    for (md, rows) in
        zip(metadata_col[mask], var_indices(sample_collection)[mask])
        samples[rows, :] .*= sqrt.(_flat_lat_weights(md; min_cosd_lat))
    end
    return SampleCollection(samples, metadata)
end

"""
    _lat_weighting_mask(
        ::Union{LatitudeWeighting{Nothing}, LatitudeWeighting{<:AbstractSet}},
        metadata_col
    )

Return a vector of `Bool`s with one entry per metadata in `metadata_col`. The `i`th entry is `true` if latitude
weighting should be applied to the samples of the variable described by
`metadata_col[i]`.
"""
function _lat_weighting_mask(::LatitudeWeighting{Nothing}, metadata_col)
    mask = [ClimaAnalysis.has_latitude(md) for md in metadata_col]
    any(mask) || error(
        "None of the variables have a latitude dimension, so latitude weighting cannot be applied",
    )
    return mask
end

function _lat_weighting_mask(
    transform::LatitudeWeighting{<:AbstractSet},
    metadata_col,
)
    (; by) = transform
    mask = [by(md) in transform.selected for md in metadata_col]
    any(mask) || error(
        "None of the variables are selected for latitude weighting; the selected keys are $(collect(transform.selected))",
    )
    for md in metadata_col[mask]
        ClimaAnalysis.has_latitude(md) || error(
            "The variable with the short name $(ClimaAnalysis.short_name(md)) is selected for latitude weighting, but it does not have a latitude dimension",
        )
    end
    return mask
end

"""
    PerVariableWeighting(
        weights::AbstractDict{<:Any, <:Real};
        by = ClimaAnalysis.short_name,
    )

Return a per-variable weighting transform that weights the variables by the
value of `weights[key]` where `key` is `by(metadata)`.

The key of a variable is `by(metadata)`, where `metadata` is the
`ClimaAnalysis.Var.Metadata` of that variable. An error is thrown if a weight
is not supplied.

# Example

We want to weight the variables the short names `pr` and `tas`.

```julia
transform = PerVariableWeighting(Dict("pr" => 2.0, "tas" => 3.0))
weighted = sample_collection |> transform
```
"""
function PerVariableWeighting(
    weights::AbstractDict{<:Any, <:Real};
    by = ClimaAnalysis.short_name,
)
    return PerVariableWeighting(weights, by)
end

"""
    apply_transform!(
        transform::PerVariableWeighting{V},
        sample_collection::SampleCollection,
    ) where {V <: AbstractVector}

Apply per-variable weighting according to `transform`.

An error is thrown if the number of weights is not equal to the number of
variables.
"""
function SampleBuilder.apply_transform!(
    transform::PerVariableWeighting{V},
    sample_collection::SampleCollection,
) where {V <: AbstractVector}
    metadata = get_metadata(sample_collection)
    (; weights) = transform
    n_weights, n_vars = length(weights), size(metadata, 1)
    n_weights == n_vars || error(
        "The number of weights ($n_weights) is not the same as the number of variables ($n_vars)",
    )
    samples = get_samples(sample_collection)
    for (weight, range) in zip(weights, var_indices(sample_collection))
        samples[range, :] .*= weight
    end
    return SampleCollection(samples, metadata)
end

"""
    apply_transform!(
        transform::PerVariableWeighting{D},
        sample_collection::SampleCollection,
    ) where {D <: AbstractDict}

Apply per-variable weighting according to `transform`.

An error is thrown if a variable in `sample_collection` does not have a weight,
that is, if `by(metadata)` is not a key of `weights`.
"""
function SampleBuilder.apply_transform!(
    transform::PerVariableWeighting{D},
    sample_collection::SampleCollection,
) where {D <: AbstractDict}
    metadata = get_metadata(sample_collection)
    (; weights, by) = transform
    samples = get_samples(sample_collection)
    metadata_col = _metadata_of_first_sample(sample_collection)
    for (md, range) in zip(metadata_col, var_indices(sample_collection))
        weight = weights[by(md)]
        samples[range, :] .*= weight
    end
    return SampleCollection(samples, metadata)
end

"""
    apply_transform!(
        transform::PerCollectionWeighting,
        sample_collection::SampleCollection,
    )

Apply per-collection weighting by the scalar in `transform`.
"""
function SampleBuilder.apply_transform!(
    transform::PerCollectionWeighting,
    sample_collection::SampleCollection,
)
    get_samples(sample_collection) .*= transform.weight
    return sample_collection
end

"""
    Base.show(io::IO, sample_collection::TransformedSampleCollection)

Show method for `TransformedSampleCollection`. It prints the information about
the number of transforms, the size of the matrix of samples, the number of
samples, values, and variables, and calls the show method of each transform.
"""
function Base.show(io::IO, sample_collection::TransformedSampleCollection)
    chain = _transform_sequence(sample_collection)
    printstyled(io, "TransformedSampleCollection", bold = true)
    print(io, " ($(length(chain)) transform(s) not yet applied)\n")
    _show_summary(io, base(sample_collection))
    for transform in chain
        print(io, "\n  ↳ ", transform)
    end
    return nothing
end

"""
    _transform_sequence(
        sample_collection::Union{SampleCollection, TransformedSampleCollection}
    )

Recursively create the sequence of transformations applied to
`sample_collection`.
"""
_transform_sequence(::SampleCollection) = ()
_transform_sequence(sample_collection::TransformedSampleCollection) = (
    _transform_sequence(sample_collection.parent)...,
    sample_collection.transform,
)

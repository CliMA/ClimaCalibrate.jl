import ClimaCalibrate.SampleBuilder:
    AbstractSampleCollection,
    AbstractTransform,
    AbstractWeighting,
    LatitudeWeighting,
    PerCollectionWeighting,
    PerVariableWeighting,
    TransformedSampleCollection,
    apply_transform,
    apply_transform!,
    base,
    compute_weights,
    transform_sequence,
    var_indices

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

Return a `SampleCollection` with every recorded transform applied to it.
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
        _apply_transform(sample_collection.parent),
        sample_collection.transform,
    )

"""
    apply_transform(
        sample_collection::AbstractSampleCollection,
        transform::AbstractTransform,
    )

Eagerly apply a transform to `sample_collection`.

This is implemented in terms of `apply_transform!`.
"""
function SampleBuilder.apply_transform(
    sample_collection::AbstractSampleCollection,
    transform::AbstractTransform,
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
    transform_sequence(
        sample_collection::Union{SampleCollection, TransformedSampleCollection}
    )

Return the sequence of transformations applied to `sample_collection` as a
tuple.
"""
function SampleBuilder.transform_sequence(::SampleCollection)
    return ()
end

function SampleBuilder.transform_sequence(
    sample_collection::TransformedSampleCollection,
)
    return (
        transform_sequence(sample_collection.parent)...,
        sample_collection.transform,
    )
end

"""
    apply_transform!(
        sample_collection::SampleCollection,
        weighting::AbstractWeighting,
    )

Apply the weights to the matrix of samples in `sample_collection` by
broadcasting.

Weights must be positive.
"""
function SampleBuilder.apply_transform!(
    sample_collection::SampleCollection,
    weighting::AbstractWeighting,
)
    weights = compute_weights(weighting, sample_collection)
    weighting_name = nameof(typeof(weighting))
    weights isa Union{Real, AbstractVector{<:Real}} || error(
        "The weights computed by $weighting_name must be a Real or an AbstractVector of Reals, but they are a $(typeof(weights))",
    )
    if weights isa AbstractVector
        n_weights = length(weights)
        n_entries = size(get_samples(sample_collection), 1)
        n_weights == n_entries || error(
            "The number of weights ($n_weights) computed by $weighting_name is not the same as the number of entries in a sample ($n_entries)",
        )
    end

    # Check all weights are positive. We do it here since weighting might be
    # user provided
    FT = eltype(weights)
    all(>(FT(0)), weights) ||
        error("The weights computed by $weighting_name are not all positive")

    get_samples(sample_collection) .*= weights
    return sample_collection
end

"""
    _entry_weights(var_weights, sample_collection::SampleCollection)

Return a vector of weight per entry of a sample from a vector of one weight per
variable.
"""
function _entry_weights(var_weights, sample_collection::SampleCollection)
    return reduce(
        vcat,
        fill.(var_weights, length.(var_indices(sample_collection))),
    )
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
`ClimaAnalysis.Var.Metadata` of that variable. When the transform is applied,
an error is thrown if a selected variable does not have a latitude dimension, if
no variable is selected, or if a key in `selected` does not match any variable.

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
    compute_weights(
        transform::LatitudeWeighting,
        sample_collection::SampleCollection,
    )

Return the latitude weights of `sample_collection` according to `transform`,
with one weight per entry of a sample. The entries of the variables that are not
weighted have a weight of one.

An error is thrown if
- no variable is weighted, either because none of the variables have a latitude
  dimension or because none of the variables are selected,
- a key in `selected` does not match any variable,
- a selected variable does not have a latitude dimension,
- the latitudes of a weighted variable are not the same across samples, or
- the latitude dimension of a weighted variable is not in degrees.
"""
function SampleBuilder.compute_weights(
    transform::LatitudeWeighting,
    sample_collection::SampleCollection,
)
    metadata = get_metadata(sample_collection)
    metadata_col = _metadata_of_first_sample(sample_collection)
    mask = _lat_weighting_mask(transform, metadata_col)
    _check_lats_across_samples(metadata[mask, :])
    (; min_cosd_lat) = transform
    FT = eltype(get_samples(sample_collection))
    # vcat promotes the element type, so no latitude weight is rounded
    var_weights = map(
        metadata_col,
        var_indices(sample_collection),
        mask,
    ) do md, rows, weighted
        weighted ? sqrt.(_flat_lat_weights(md; min_cosd_lat)) :
        ones(FT, length(rows))
    end
    return reduce(vcat, var_weights)
end

"""
    _lat_weighting_mask(
        ::Union{LatitudeWeighting{Nothing}, LatitudeWeighting{<:AbstractSet}},
        metadata_col
    )

Return a vector of `Bool`s with one entry per metadata in `metadata_col`. The
`i`th entry is `true` if latitude weighting should be applied to the samples of
the variable described by `metadata_col[i]`.
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
    (; by, selected) = transform
    var_keys = [by(md) for md in metadata_col]
    mask = [key in selected for key in var_keys]
    any(mask) || error(
        "None of the variables are selected for latitude weighting. The selected keys are $(collect(selected))",
    )
    unused_keys = setdiff(selected, var_keys)
    isempty(unused_keys) || error(
        "The selected keys $(collect(unused_keys)) do not match any variable",
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

Return a per-variable weighting transform that weights each variable by
`weights[key]`.

The key of a variable is `by(metadata)`, where `metadata` is the
`ClimaAnalysis.Var.Metadata` of that variable. When the transform is applied,
an error is thrown if a variable does not have a weight.

# Example

We want to weight the variables with the short names `pr` and `tas`.

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
    compute_weights(
        transform::PerVariableWeighting{V},
        sample_collection::SampleCollection,
    ) where {V <: AbstractVector}

Return the per-variable weights of `sample_collection` according to
`transform`, with one weight per entry of a sample.

An error is thrown if the number of weights is not equal to the number of
variables.
"""
function SampleBuilder.compute_weights(
    transform::PerVariableWeighting{V},
    sample_collection::SampleCollection,
) where {V <: AbstractVector}
    metadata = get_metadata(sample_collection)
    (; weights) = transform
    n_weights, n_vars = length(weights), size(metadata, 1)
    n_weights == n_vars || error(
        "The number of weights ($n_weights) is not the same as the number of variables ($n_vars)",
    )
    return _entry_weights(weights, sample_collection)
end

"""
    compute_weights(
        transform::PerVariableWeighting{D},
        sample_collection::SampleCollection,
    ) where {D <: AbstractDict}

Return the per-variable weights of `sample_collection` according to
`transform`, with one weight per entry of a sample.

An error is thrown if a variable in `sample_collection` does not have a weight,
that is, if `by(metadata)` is not a key of `weights`.
"""
function SampleBuilder.compute_weights(
    transform::PerVariableWeighting{D},
    sample_collection::SampleCollection,
) where {D <: AbstractDict}
    (; weights, by) = transform
    var_keys = [by(md) for md in _metadata_of_first_sample(sample_collection)]
    missing_keys = filter(key -> !haskey(weights, key), var_keys)
    isempty(missing_keys) || error(
        "No weights are given for the variables with the keys $missing_keys",
    )
    return _entry_weights([weights[key] for key in var_keys], sample_collection)
end

"""
    compute_weights(
        transform::PerCollectionWeighting,
        sample_collection::SampleCollection,
    )

Return the scalar in `transform`, which weights every entry of
`sample_collection`.
"""
function SampleBuilder.compute_weights(
    transform::PerCollectionWeighting,
    ::SampleCollection,
)
    return transform.weight
end

"""
    Base.show(io::IO, sample_collection::TransformedSampleCollection)

Show method for `TransformedSampleCollection`. It prints the information about
the number of transforms, the size of the matrix of samples, the number of
samples, values, and variables, and calls the show method of each transform.
"""
function Base.show(io::IO, sample_collection::TransformedSampleCollection)
    chain = transform_sequence(sample_collection)
    printstyled(io, "TransformedSampleCollection", bold = true)
    print(io, " ($(length(chain)) transform(s) not yet applied)\n")
    _show_summary(io, base(sample_collection))
    for transform in chain
        print(io, "\n  ↳ ", transform)
    end
    return nothing
end

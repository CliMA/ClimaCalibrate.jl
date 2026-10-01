import ClimaCalibrate.ObservationRecipe:
    AbstractDiagonalTerm,
    ScalarDiagonal,
    ModelErrorScaleDiagonal,
    VarianceDiagonal,
    QuantileDiagonal,
    LatitudeWeightDiagonal,
    BroadcastedDiagonal,
    compute_diagonal

"""
    compute_diagonal(
        diagonal_term::ScalarDiagonal,
        sample_collection::SampleCollection,
    )

Compute a diagonal matrix whose entries are constant for each variable in
`sample_collection`.
"""
function compute_diagonal(
    diagonal_term::ScalarDiagonal,
    sample_collection::SampleCollection,
)
    ranges = SampleBuilder.var_indices(sample_collection)
    (; scalars) = diagonal_term
    length(scalars) == 1 ||
        length(scalars) == length(ranges) ||
        error(
            "The number of scalars ($(length(scalars))) does not match the number of variables ($(length(ranges)))",
        )
    FT = eltype(get_samples(sample_collection))
    return Diagonal(
        convert(Vector{FT}, reduce(vcat, fill.(scalars, length.(ranges)))),
    )
end

"""
    compute_diagonal(
        diagonal_term::ModelErrorScaleDiagonal,
        sample_collection::SampleCollection,
    )

Compute a diagonal matrix whose entries are `(scale * mean)^2`, where `mean` is
the mean of each entry across the samples in `sample_collection`, ignoring
`NaN`s, and `scale` is the model error scale of the variable that the entry
belongs to.
"""
function compute_diagonal(
    diagonal_term::ModelErrorScaleDiagonal,
    sample_collection::SampleCollection,
)
    ranges = SampleBuilder.var_indices(sample_collection)
    (; model_error_scales) = diagonal_term
    length(model_error_scales) == 1 ||
        length(model_error_scales) == length(ranges) ||
        error(
            "The number of model error scales ($(length(model_error_scales))) does not match the number of variables ($(length(ranges)))",
        )
    samples = get_samples(sample_collection)
    scales = reduce(vcat, fill.(model_error_scales, length.(ranges)))
    FT = eltype(samples)
    # There shouldn't be NaNs in the samples because ClimaAnalysis.flatten
    # automatically remove NaNs (when not using a mask) and it doesn't make
    # sense to calibrate with NaNs in the samples, but we use nanmean for
    # future proofing
    return Diagonal(
        convert(Vector{FT}, vec((scales .* nanmean(samples, dims = 2)) .^ 2)),
    )
end

"""
    compute_diagonal(
        diagonal_term::VarianceDiagonal,
        sample_collection::SampleCollection,
    )

Compute a diagonal matrix whose entries are the variance of each row of the
sample matrix in `sample_collection`, taken across the samples and ignoring
`NaN`s.
"""
function compute_diagonal(
    diagonal_term::VarianceDiagonal,
    sample_collection::SampleCollection,
)
    samples = get_samples(sample_collection)
    FT = eltype(samples)
    return Diagonal(convert(Vector{FT}, vec(nanvar(samples, dims = 2))))
end

"""
    compute_diagonal(
        diagonal_term::QuantileDiagonal,
        sample_collection::SampleCollection,
    )

Compute a diagonal matrix whose entries are constant for each variable in
`sample_collection`. The constant for each variable is the quantile of the
entries that the wrapped diagonal term produces for that variable.

An error is thrown if a variable has fewer than `1 / quantile` entries, since
the quantile would not be meaningful, or if the quantile value is zero.
"""
function compute_diagonal(
    diagonal_term::QuantileDiagonal,
    sample_collection::SampleCollection,
)
    metadata = _metadata_of_first_sample(sample_collection)
    ranges = SampleBuilder.var_indices(sample_collection)
    (; quantiles, diag_term) = diagonal_term
    length(quantiles) == 1 ||
        length(quantiles) == length(ranges) ||
        error(
            "The number of quantiles ($(length(quantiles))) does not match the number of variables ($(length(ranges)))",
        )

    diagonal_vec = diag(compute_diagonal(diag_term, sample_collection))
    FT = eltype(get_samples(sample_collection))

    quantile_vals_vec = []
    for (i, indices) in enumerate(ranges)
        qtl = quantiles[length(quantiles) == 1 ? 1 : i]
        var_diagonal_vec = view(diagonal_vec, indices)
        # Check that there is a sufficient number of entries (e.g. if qtl =
        # 0.05, there should be at least 20 entries for a meaningful
        # quantile computation)
        length(var_diagonal_vec) < 1 / qtl && error(
            "QuantileDiagonal with quantile $qtl needs a variable with at least \
            $(ceil(Int, 1 / qtl)) entries to take a meaningful quantile, but \
            variable $(ClimaAnalysis.short_name(metadata[i])) has only \
            $(length(var_diagonal_vec)). Use a larger quantile or a constant \
            regularization instead.",
        )
        qtl_for_var = FT(Statistics.quantile(var_diagonal_vec, qtl))
        iszero(qtl_for_var) && error(
            "Zero found for the quantile ($qtl) of the diagonal term ($(typeof(diag_term))) for the variable ($(ClimaAnalysis.short_name(metadata[i]))). The values of the diagonal term might be too small",
        )
        push!(quantile_vals_vec, qtl_for_var)
    end

    return Diagonal(
        convert(
            Vector{FT},
            reduce(vcat, fill.(quantile_vals_vec, length.(ranges))),
        ),
    )
end

"""
    LatitudeWeightDiagonal(
        selected::Union{AbstractVector, AbstractSet, Tuple};
        by = ClimaAnalysis.short_name,
        min_cosd_lat::AbstractFloat = 0.1,
    )

A diagonal term whose entries are `1 / max(cosd(lat), min_cosd_lat)` for
variables whose key is in `selected` and `1` for all other variables.

The key of a variable is `by(metadata)`, where `metadata` is the
`ClimaAnalysis.Var.Metadata` of that variable.

See [`LatitudeWeighting`](@ref
ClimaCalibrate.SampleBuilder.LatitudeWeighting(::Union{AbstractVector,
AbstractSet, Tuple})) for the errors that are thrown.

# Example

Weight only the variables with the short names `pr` and `tas`, leaving the
other variables in the sample collection unweighted:

```julia
lat_weight_diagonal = LatitudeWeightDiagonal(["pr", "tas"])
```
"""
function LatitudeWeightDiagonal(
    selected::Union{AbstractVector, AbstractSet, Tuple};
    by = ClimaAnalysis.short_name,
    min_cosd_lat::AbstractFloat = 0.1,
)
    return LatitudeWeightDiagonal(LatitudeWeighting(selected; by, min_cosd_lat))
end

"""
    compute_diagonal(
        diagonal_term::LatitudeWeightDiagonal,
        sample_collection::SampleCollection,
    )

Compute a diagonal matrix whose entries are the latitude weights
`1 / max(cosd(lat), min_cosd_lat)` for the weighted variables in
`sample_collection` and one for unweighted variables.
"""
function compute_diagonal(
    diagonal_term::LatitudeWeightDiagonal,
    sample_collection::SampleCollection,
)
    weights = _lat_weights(diagonal_term.latitude_weighting, sample_collection)
    FT = eltype(get_samples(sample_collection))
    return Diagonal(convert(Vector{FT}, weights))
end

"""
    compute_diagonal(
        diagonal_term::BroadcastedDiagonal,
        sample_collection::SampleCollection,
    )

Compute a diagonal matrix by computing every diagonal term in the broadcast
expression from `sample_collection` and broadcasting the expression over the
resulting `Diagonal` matrices.

The broadcast follows the rules for `Diagonal` matrices, so an error is thrown
if the result is not diagonal, for example when adding a scalar.
"""
function compute_diagonal(
    diagonal_term::BroadcastedDiagonal,
    sample_collection::SampleCollection,
)
    # To evaluate the broadcasted expression, we do a post order traversal which
    # evaluate the AbstractDiagonalTerm we see and combine them with the
    # function applied
    FT = eltype(get_samples(sample_collection))
    args = map(diagonal_term.args) do arg
        arg isa AbstractDiagonalTerm ?
        compute_diagonal(arg, sample_collection) : arg
    end
    diag_cov = diagonal_term.f.(args...)
    diag_cov isa Diagonal || error(
        "Broadcasting `$(diagonal_term.f)` over the diagonal terms did not produce a diagonal matrix, but a $(typeof(diag_cov))",
    )
    return Diagonal(convert(Vector{FT}, diag_cov.diag))
end

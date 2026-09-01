import ClimaCalibrate.ObservationRecipe:
    AbstractDiagonalTerm,
    ScalarDiagonal,
    ModelErrorScaleDiagonal,
    VarianceDiagonal,
    QuantileDiagonal,
    SumDiagonal,
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
            "The number of scalars ($scalars) does not match the number of ranges ($ranges)",
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
the mean of each entry across the samples in `sample_collection` and `scale` is
the model error scale of the variable that the entry belongs to.
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
            "The number of model error scales ($model_error_scales) does not match the number of ranges ($ranges)",
        )
    samples = get_samples(sample_collection)
    scales = reduce(vcat, fill.(model_error_scales, length.(ranges)))
    FT = eltype(samples)
    # There shouldn't be NaNs in the samples because ClimaAnalysis.flatten
    # automatically remove NaNs (when not using a mask) and it doesn't make
    # sense to calibrate with NaNs in the samples, but we use nanmean to be safe
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
sample matrix in `sample_collection`, taken across the samples.
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
`sample_collection`. The constant for a variable is a quantile of the entries
that the wrapped diagonal term produces for that variable.

An error is thrown if a variable has fewer than `1 / quantile` entries, since the
quantile would not be meaningful, or if the quantile is zero.
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
            "The number of quantiles ($quantiles) does not match the number of ranges ($ranges)",
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
        length(var_diagonal_vec) < 1 / qtl &&
            error("Insufficient samples for computing quantile")
        qtl_for_var = FT(Statistics.quantile(var_diagonal_vec, qtl))
        qtl_for_var ≈ 0.0 && error(
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
    compute_diagonal(
        diagonal_term::SumDiagonal,
        sample_collection::SampleCollection,
    )

Compute a diagonal matrix by computing each diagonal term in `diagonal_term` from
`sample_collection` and summing the terms.
"""
function compute_diagonal(
    diagonal_term::SumDiagonal,
    sample_collection::SampleCollection,
)
    FT = eltype(get_samples(sample_collection))
    sum_diagonal = mapreduce(
        term -> compute_diagonal(term, sample_collection),
        +,
        diagonal_term.diagonal_terms,
    )
    return Diagonal(convert(Vector{FT}, diag(sum_diagonal)))
end

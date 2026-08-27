import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.ObservationRecipe:
    ScalarDiagonal, ModelErrorScaleDiagonal, QuantileDiagonal, SumDiagonal

"""
    build_diagonal(
        diagonal_builder::ScalarDiagonal,
        sample_collection::SampleCollection,
    )

Build the diagonal matrix of the form `value * I`, whose side length is the
number of rows of the matrix of samples in `sample_collection`.

The data in the matrix of samples in `sample_collection` is ignored except for
its element type and number of rows. The scalar is cast to the element type of
the matrix of samples so the resulting diagonal matrix keeps a consistent
element type (e.g. Float32).
"""
function ObservationRecipe.build_diagonal(
    diagonal_builder::ScalarDiagonal,
    sample_collection::SampleCollection,
)
    samples = get_samples(sample_collection)
    FT = eltype(samples)
    return Diagonal(fill(FT(diagonal_builder.value), size(samples, 1)))
end

"""
    build_diagonal(
        diagonal_builder::ModelErrorScaleDiagonal,
        sample_collection::SampleCollection,
    )

Build the diagonal matrix whose diagonal is
`vec((model_error_scale .* mean(samples, dims = 2)).^2)`, where `samples` is
the matrix of samples in `sample_collection`.

The model error scale is cast to the element type of the matrix of samples so
the resulting diagonal matrix keeps a consistent element type (e.g. Float32).

This may not make sense if the samples do not represent a single year. For
example, if the stacked samples are seasonal averages over two years, then the
mean of the samples is the mean of seasonal averages spanned over two years,
where the first DJF is the mean of every other DJF and the second DJF is the
mean of every other DJF.
"""
function ObservationRecipe.build_diagonal(
    diagonal_builder::ModelErrorScaleDiagonal,
    sample_collection::SampleCollection,
)
    samples = get_samples(sample_collection)
    FT = eltype(samples)
    return Diagonal(
        vec(
            (
                FT(diagonal_builder.model_error_scale) .*
                mean(samples, dims = 2)
            ) .^ 2,
        ),
    )
end

"""
    build_diagonal(
        diagonal_builder::QuantileDiagonal,
        sample_collection::SampleCollection,
    )

Build the diagonal matrix where each variable gets its own constant value
along the diagonal, computed as the `diagonal_builder.qtl` quantile of the
diagonal built from `diagonal_builder.builder`.

For each variable, the `qtl` quantile of the diagonal entries corresponding to
that variable is computed and used as a constant value for all the entries
belonging to that variable. The per-variable index ranges are determined from
the metadata in `sample_collection` (one `Metadata` per variable).
"""
function ObservationRecipe.build_diagonal(
    diagonal_builder::QuantileDiagonal,
    sample_collection::SampleCollection,
)
    inner_diag_cov = ObservationRecipe.build_diagonal(
        diagonal_builder.builder,
        sample_collection,
    )
    isdiag(inner_diag_cov) || error(
        "The matrix from build_diagonal with $(diagonal_builder.builder) is not a diagonal matrix",
    )
    inner_diag_vec = diag(inner_diag_cov)
    FT = eltype(inner_diag_vec)
    (; qtl) = diagonal_builder

    metadata = _metadata_of_first_sample(sample_collection)
    indices_vec = _get_indices_of_metadata(metadata)

    qtl_diag_vec = similar(inner_diag_vec)
    for (i, indices) in enumerate(indices_vec)
        var_diag_vec = view(inner_diag_vec, indices)
        # Check that there is a sufficient number of values (e.g. if qtl =
        # 0.05, there should be at least 20 values for a meaningful quantile
        # computation)
        length(var_diag_vec) < 1 / qtl &&
            error("Insufficient samples for computing quantile")
        qtl_for_var = FT(Statistics.quantile(var_diag_vec, qtl))
        qtl_for_var ≈ 0.0 && error(
            "Zero found for the quantile ($qtl) of the diagonal built from $(diagonal_builder.builder) for the variable ($(ClimaAnalysis.short_name(metadata[i]))). The values along the diagonal might be too small",
        )
        qtl_diag_vec[indices] .= qtl_for_var
    end
    return Diagonal(qtl_diag_vec)
end

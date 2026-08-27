"""
    AbstractDiagonalBuilder

An object that builds a diagonal matrix from the samples in a
`SampleCollection`, such as the diagonal matrix `D` of the `EKP.SVDplusD`
covariance matrix (see `SVDplusDCovariance`).

`AbstractDiagonalBuilder`s have to provide one function,
`ObservationRecipe.build_diagonal`.

The function has to have the signature

```julia
ObservationRecipe.build_diagonal(
    diagonal_builder::AbstractDiagonalBuilder,
    sample_collection,
)
```

and return a diagonal matrix whose side length is the number of rows of the
matrix of samples in `sample_collection`.

If `use_latitude_weights = true` in `SVDplusDCovariance`, then the samples in
the `sample_collection` passed to `build_diagonal` already have latitude
weights applied.

Any parameters needed to build the diagonal matrix should be stored as fields
of the diagonal builder (see `QuantileDiagonal` for an example).

Diagonal builders can be added together with `+`. The resulting diagonal
builder builds the sum of the diagonal matrices of each diagonal builder (see
`SumDiagonal`).

Diagonal builders are not `AbstractCovarianceEstimator`s and cannot be passed
to `ObservationRecipe.observation` or `ObservationRecipe.covariance`.
"""
abstract type AbstractDiagonalBuilder end

"""
    build_diagonal(diagonal_builder, sample_collection)

Build a diagonal matrix from `diagonal_builder` and the samples in
`sample_collection`.
"""
function build_diagonal end

"""
    ScalarDiagonal <: AbstractDiagonalBuilder

Build a diagonal matrix of the form `value * I`.

Examples
========

```julia
scalar_diagonal = ObservationRecipe.ScalarDiagonal(1e-6)
```
"""
struct ScalarDiagonal{FT <: AbstractFloat} <: AbstractDiagonalBuilder
    """Value along the diagonal of the matrix"""
    value::FT
    function ScalarDiagonal(value::AbstractFloat)
        value < zero(value) &&
            error("The value ($value) should not be negative")
        return new{typeof(value)}(value)
    end
end

"""
    ModelErrorScaleDiagonal <: AbstractDiagonalBuilder

Build a diagonal matrix whose diagonal is
`vec((model_error_scale .* mean(samples, dims = 2)).^2)`, where
`mean(samples, dims = 2)` is the mean of the samples.

Examples
========

```julia
model_error_scale = ObservationRecipe.ModelErrorScaleDiagonal(0.05)
```
"""
struct ModelErrorScaleDiagonal{FT <: AbstractFloat} <: AbstractDiagonalBuilder
    """Scale of the noise from the model error"""
    model_error_scale::FT
    function ModelErrorScaleDiagonal(model_error_scale::AbstractFloat)
        model_error_scale < zero(model_error_scale) && error(
            "Model_error_scale ($model_error_scale) should not be negative",
        )
        return new{typeof(model_error_scale)}(model_error_scale)
    end
end

"""
    QuantileDiagonal <: AbstractDiagonalBuilder

Build a diagonal matrix where each variable gets its own constant value along
the diagonal, computed as the `qtl` quantile of the diagonal built from
`builder`.

For each variable, the `qtl` quantile of the diagonal entries corresponding to
that variable is computed and used as a constant value for all the entries
belonging to that variable.

Examples
========

In the example below, each variable gets the 0.05 quantile of that variable's
entries of the diagonal of the model error scale.

```julia
model_error_scale = ObservationRecipe.ModelErrorScaleDiagonal(0.05)
qtl_diagonal = ObservationRecipe.QuantileDiagonal(0.05, model_error_scale)
```
"""
struct QuantileDiagonal{FT <: AbstractFloat, B <: AbstractDiagonalBuilder} <:
       AbstractDiagonalBuilder
    """Quantile in the interval (0, 1]"""
    qtl::FT
    """Diagonal builder whose diagonal the quantiles are computed from"""
    builder::B
    function QuantileDiagonal(
        qtl::AbstractFloat,
        builder::AbstractDiagonalBuilder,
    )
        (qtl <= 0 || qtl > 1) && error("Quantile must be in (0, 1], got $qtl")
        return new{typeof(qtl), typeof(builder)}(qtl, builder)
    end
end

"""
    SumDiagonal <: AbstractDiagonalBuilder

Build a diagonal matrix that is the sum of the diagonal matrices built from
`builders`.

A `SumDiagonal` is constructed by adding diagonal builders together with `+`.

Examples
========

```julia
sum_diagonal =
    ObservationRecipe.ModelErrorScaleDiagonal(0.05) +
    ObservationRecipe.ScalarDiagonal(1e-6)
```
"""
struct SumDiagonal{T <: Tuple{Vararg{AbstractDiagonalBuilder}}} <:
       AbstractDiagonalBuilder
    """Diagonal builders whose diagonal matrices are summed"""
    builders::T
end

Base.:+(a::AbstractDiagonalBuilder, b::AbstractDiagonalBuilder) =
    SumDiagonal((a, b))
Base.:+(a::SumDiagonal, b::AbstractDiagonalBuilder) =
    SumDiagonal((a.builders..., b))
Base.:+(a::AbstractDiagonalBuilder, b::SumDiagonal) =
    SumDiagonal((a, b.builders...))
Base.:+(a::SumDiagonal, b::SumDiagonal) =
    SumDiagonal((a.builders..., b.builders...))

"""
    build_diagonal(sum_diagonal::SumDiagonal, sample_collection)

Build the diagonal matrix that is the sum of the diagonal matrices built from
the diagonal builders in `sum_diagonal`.
"""
function build_diagonal(sum_diagonal::SumDiagonal, sample_collection)
    return mapreduce(
        builder -> build_diagonal(builder, sample_collection),
        +,
        sum_diagonal.builders,
    )
end

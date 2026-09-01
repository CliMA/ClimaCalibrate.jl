"""
    AbstractDiagonalTerm

An object that builds a diagonal matrix from the samples in a
`SampleCollection`, such as the diagonal matrix `D` of the `EKP.SVDplusD`
covariance matrix (see `SVDplusDCovariance`).

`AbstractDiagonalTerm`s have to provide one function,
`ObservationRecipe.compute_diagonal`.

The function has to have the signature

```julia
ObservationRecipe.compute_diagonal(
    diagonal_term::AbstractDiagonalTerm,
    sample_collection,
)
```

and return a diagonal matrix whose side length is the number of rows of the
matrix of samples in `sample_collection`.

If `use_latitude_weights = true` in `SVDplusDCovariance`, then the samples in
the `sample_collection` passed to `compute_diagonal` already have latitude
weights applied.

Any parameters needed to build the diagonal matrix should be stored as fields
of the diagonal term (see `QuantileDiagonal` for an example).

Diagonal terms can be added together with `+`. The resulting diagonal
term builds the sum of the diagonal matrices of each diagonal term (see
`SumDiagonal`).

Diagonal terms are not `AbstractCovarianceEstimator`s and cannot be passed
to `ObservationRecipe.observation` or `ObservationRecipe.covariance`.
"""
abstract type AbstractDiagonalTerm end

"""
    compute_diagonal(diagonal_term, sample_collection)

Compute a diagonal matrix from `diagonal_term` and the samples in
`sample_collection`.
"""
function compute_diagonal end

"""
    ScalarDiagonal <: AbstractDiagonalTerm

Compute a diagonal matrix of the form `value * I`.

Examples
========

```julia
scalar_diagonal = ObservationRecipe.ScalarDiagonal(1e-6)
```
"""
struct ScalarDiagonal{FT <: AbstractFloat} <: AbstractDiagonalTerm
    """Value along the diagonal of the matrix"""
    value::FT
    function ScalarDiagonal(value::AbstractFloat)
        value < zero(value) &&
            error("The value ($value) should not be negative")
        return new{typeof(value)}(value)
    end
end

"""
    ModelErrorScaleDiagonal <: AbstractDiagonalTerm

Compute a diagonal matrix whose diagonal is
`vec((model_error_scale .* mean(samples, dims = 2)).^2)`, where
`mean(samples, dims = 2)` is the mean of the samples.

Examples
========

```julia
model_error_scale = ObservationRecipe.ModelErrorScaleDiagonal(0.05)
```
"""
struct ModelErrorScaleDiagonal{FT <: AbstractFloat} <: AbstractDiagonalTerm
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
    QuantileDiagonal <: AbstractDiagonalTerm

Compute a diagonal matrix where each variable gets its own constant value along
the diagonal, computed as the `qtl` quantile of the diagonal built from
`term`.

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
struct QuantileDiagonal{FT <: AbstractFloat, B <: AbstractDiagonalTerm} <:
       AbstractDiagonalTerm
    """Quantile in the interval (0, 1]"""
    qtl::FT
    """Diagonal term whose diagonal the quantiles are computed from"""
    term::B
    function QuantileDiagonal(qtl::AbstractFloat, term::AbstractDiagonalTerm)
        (qtl <= 0 || qtl > 1) && error("Quantile must be in (0, 1], got $qtl")
        return new{typeof(qtl), typeof(term)}(qtl, term)
    end
end

"""
    SumDiagonal <: AbstractDiagonalTerm

Compute a diagonal matrix that is the sum of the diagonal matrices built from
`terms`.

A `SumDiagonal` is constructed by adding diagonal terms together with `+`.

Examples
========

```julia
sum_diagonal =
    ObservationRecipe.ModelErrorScaleDiagonal(0.05) +
    ObservationRecipe.ScalarDiagonal(1e-6)
```
"""
struct SumDiagonal{T <: Tuple{Vararg{AbstractDiagonalTerm}}} <:
       AbstractDiagonalTerm
    """Diagonal terms whose diagonal matrices are summed"""
    terms::T
end

Base.:+(a::AbstractDiagonalTerm, b::AbstractDiagonalTerm) = SumDiagonal((a, b))
Base.:+(a::SumDiagonal, b::AbstractDiagonalTerm) = SumDiagonal((a.terms..., b))
Base.:+(a::AbstractDiagonalTerm, b::SumDiagonal) = SumDiagonal((a, b.terms...))
Base.:+(a::SumDiagonal, b::SumDiagonal) = SumDiagonal((a.terms..., b.terms...))

"""
    compute_diagonal(sum_diagonal::SumDiagonal, sample_collection)

Compute the diagonal matrix that is the sum of the diagonal matrices built from
the diagonal terms in `sum_diagonal`.
"""
function compute_diagonal(sum_diagonal::SumDiagonal, sample_collection)
    return mapreduce(
        term -> compute_diagonal(term, sample_collection),
        +,
        sum_diagonal.terms,
    )
end

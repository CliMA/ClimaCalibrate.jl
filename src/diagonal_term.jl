export ScalarDiagonal,
    ModelErrorScaleDiagonal,
    VarianceDiagonal,
    QuantileDiagonal,
    SumDiagonal,
    compute_diagonal

"""
    abstract type AbstractDiagonalTerm end

A description of how to build the diagonal matrix of a covariance matrix from a
`SampleCollection`. It is not a matrix itself.

A diagonal term is lazy because the covariance recipe may transform the samples
before the diagonal is computed. The term only records what to compute, and
[`compute_diagonal`](@ref) builds the matrix once the final samples are
available.

To define a custom diagonal term, subtype `AbstractDiagonalTerm` and implement a
method of [`compute_diagonal`](@ref) for it.
"""
abstract type AbstractDiagonalTerm end

"""
    compute_diagonal(diagonal_term::AbstractDiagonalTerm, sample_collection)

Compute the diagonal matrix described by `diagonal_term` from the samples in
`sample_collection`.

!!! note "Implementing `compute_diagonal` for a custom diagonal term"
    Define a method with the signature
    `compute_diagonal(term::YourType, sample_collection)`. The method must
    return an `n × n` diagonal matrix, where `n` is the number of rows of the
    samples in `sample_collection`.
"""
function compute_diagonal end

"""
    ScalarDiagonal{FT <: AbstractFloat} <: AbstractDiagonalTerm

A diagonal term whose entries are constant for each corresponding variable.

# Example

```julia
scalars = [1e-6, 1e-4]
ScalarDiagonal(scalars)
```

In the example, `scalars[i]` fills every diagonal entry belonging to the `i`th
variable. If `scalars` is length `1`, then the same constant is used for all
variables.
"""
struct ScalarDiagonal{FT <: AbstractFloat} <: AbstractDiagonalTerm
    scalars::Vector{FT}
    function ScalarDiagonal(scalars::Vector{FT}) where {FT <: AbstractFloat}
        isempty(scalars) && error("Scalars should not be empty")
        return new{FT}(scalars)
    end
end

"""
    ScalarDiagonal(scalar::AbstractFloat)

A diagonal term where every entry is `scalar`.
"""
function ScalarDiagonal(scalar::AbstractFloat)
    ScalarDiagonal([scalar])
end

"""
    ModelErrorScaleDiagonal{FT <: AbstractFloat} <: AbstractDiagonalTerm

A diagonal term whose entries are `(scale * mean)^2`, where `mean` is the mean
of each entry across the samples and `scale` is the model error scale of the
variable that the entry belongs to.

# Example

```julia
model_error_scales = [0.05, 0.1]
ModelErrorScaleDiagonal(model_error_scales)
```

In the example, `model_error_scales[i]` scales the mean of every entry belonging
to the `i`th variable. Scales must not be negative. If `model_error_scales` is
length `1`, then the same model error scale is used for all variables.
"""
struct ModelErrorScaleDiagonal{FT <: AbstractFloat} <: AbstractDiagonalTerm
    model_error_scales::Vector{FT}
    function ModelErrorScaleDiagonal(
        model_error_scales::Vector{FT},
    ) where {FT <: AbstractFloat}
        isempty(model_error_scales) &&
            error("Model error scales should not be empty")
        any(scale -> scale < zero(scale), model_error_scales) && error(
            "Model error scales ($model_error_scales) should not be negative",
        )
        return new{FT}(model_error_scales)
    end
end

"""
    ModelErrorScaleDiagonal(model_error_scale::AbstractFloat)

A diagonal term whose entries are `(model_error_scale * mean)^2`, using the same
scale for every variable.
"""
function ModelErrorScaleDiagonal(model_error_scale::AbstractFloat)
    ModelErrorScaleDiagonal([model_error_scale])
end

"""
    VarianceDiagonal <: AbstractDiagonalTerm

A diagonal term whose entries are the variance of each row of the sample matrix,
taken across the samples.

# Example

```julia
VarianceDiagonal()
```
"""
struct VarianceDiagonal <: AbstractDiagonalTerm end

"""
    QuantileDiagonal <: AbstractDiagonalTerm

A diagonal term whose entries are constant for each variable, where the constant
is a quantile of the entries that another diagonal term produces for that
variable.

# Example

```julia
quantiles = [0.5, 0.05]
QuantileDiagonal(quantiles, VarianceDiagonal())
```

In the example, the variances are computed first. Then, for the `i`th variable,
the `quantiles[i]` quantile of that variable's variances becomes the value of
every diagonal entry belonging to that variable. Quantiles must be in `(0, 1]`.
This is useful for smoothing out or flooring a diagonal term computed from the
samples. If `quantiles` is length `1`, then the same quantile is used for all
variables.
"""
struct QuantileDiagonal{FT, D <: AbstractDiagonalTerm} <: AbstractDiagonalTerm
    quantiles::Vector{FT}
    diag_term::D
    function QuantileDiagonal(
        quantiles::Vector{FT},
        diag_term::D,
    ) where {FT, D <: AbstractDiagonalTerm}
        isempty(quantiles) && error("Quantiles should not be empty")
        all(qtl -> 0 < qtl <= 1, quantiles) ||
            error("Quantiles must be in (0, 1], got $quantiles")
        return new{FT, D}(quantiles, diag_term)
    end
end

"""
    QuantileDiagonal(
        quantile::AbstractFloat,
        diag_term::AbstractDiagonalTerm,
    )

A diagonal term whose entries are constant for each variable, where the constant
is the `quantile` of the entries that `diag_term` produces for that variable.
The same `quantile` is used for every variable.
"""
function QuantileDiagonal(
    quantile::AbstractFloat,
    diag_term::AbstractDiagonalTerm,
)
    QuantileDiagonal([quantile], diag_term)
end

"""
    SumDiagonal <: AbstractDiagonalTerm

A diagonal term whose entries are the sum of other diagonal terms.

# Example

```julia
ModelErrorScaleDiagonal(0.05) + ScalarDiagonal(1e-6)
```

You should not directly construct a `SumDiagonal` since it can be constructed
with `+`.
"""
struct SumDiagonal{T <: Tuple{Vararg{AbstractDiagonalTerm}}} <:
       AbstractDiagonalTerm
    diagonal_terms::T
    function SumDiagonal(
        diagonal_terms::T,
    ) where {T <: Tuple{Vararg{AbstractDiagonalTerm}}}
        isempty(diagonal_terms) && error("Diagonal terms should not be empty")
        return new{T}(diagonal_terms)
    end
end

# Implement addition for diagonal terms. It is unlikely we will support
# any other operations since users can write their own AbstractDiagonalTerm
Base.:+(term1::AbstractDiagonalTerm, term2::AbstractDiagonalTerm) =
    SumDiagonal((term1, term2))
Base.:+(term1::SumDiagonal, term2::AbstractDiagonalTerm) =
    SumDiagonal((term1.diagonal_terms..., term2))
Base.:+(term1::AbstractDiagonalTerm, term2::SumDiagonal) =
    SumDiagonal((term1, term2.diagonal_terms...))
Base.:+(term1::SumDiagonal, term2::SumDiagonal) =
    SumDiagonal((term1.diagonal_terms..., term2.diagonal_terms...))

# Design notes: diagonal terms for `ObservationRecipe`

Notes from a design discussion of the wip commit (`562481b7e`), which introduces
composable "diagonal builders" for constructing the diagonal matrix of the
`EKP.SVDplusD` covariance produced by `SVDplusDCovariance`. The points below
are agreed in principle but not final; nothing has been implemented yet.

## Overall assessment

The composition design is sound and already matches the vision that
`ObservationRecipe` structs *describe* the covariance matrix rather than how to
make it: users of the built-ins write
`SVDplusDCovariance(model_error_scale = 0.05, regularization = 1e-6)` and never
see the term layer. The spec structs hold no data; `build_diagonal` is the
interpreter. This is the same "spec + interpreter over samples" pattern that
`AbstractCovarianceEstimator`/`covariance` already uses, so "takes the samples"
is not itself the ick — noise estimated from samples has to meet the data
somewhere. The real ick is that the terms receive a whole `SampleCollection`
when they need far less (see "Narrow the signature").

## Decisions

### No new module

The diagonal terms stay inside `ObservationRecipe` (as they are in the wip
commit, via an included file). Rationale:

- The only plausible second consumer (`SeasonalDiagonalCovariance`) lives
  inside `ObservationRecipe` anyway, so a module boundary adds import ceremony
  without decoupling anything.
- Candidate module names were all bad in some way: `Utilities` /
  `CovarianceHelper` are grab-bag names scoped by consumer instead of content;
  `DiagonalBuilders` / `CovarianceBuilders` reuse "builder", which
  `SampleBuilder` and `GEnsembleBuilder` already own; kitchen-analogy names
  (`Utensils`, `Pantry`) require knowing the metaphor to guess the contents.
- If a genuine external consumer appears later, extraction to a submodule with
  re-exports from `ObservationRecipe` is mechanical and non-breaking. To keep
  it mechanical: keep the file boundary clean (terms must not reach into
  recipe/estimator internals, only the `compute_diagonal` contract).

### Rename sweep: drop the "builder" vocabulary

- `AbstractDiagonalBuilder` → `AbstractDiagonalTerm`. "Term" matches the
  existing docstring vocabulary ("a model error scale *term* added to the
  diagonal", "a regularization *term*") and the `+` composition: `SumDiagonal`
  is literally a sum of terms. (`AbstractDiagonalComponent` was the runner-up;
  it only wins if non-additive composition is expected later.)
- `build_diagonal` → `compute_diagonal`. Terms *estimate* numbers from samples;
  "compute" says numeric work on data, "create"/"build" suggest instantiating
  an object. Also separates the vocabularies: `SampleBuilder` *builds*
  containers, terms *compute* values. The name survives the vector-returning
  contract unchanged (it returns *the diagonal*, i.e. its entries).
- Knock-on renames: `SumDiagonal.builders` → `terms`,
  `QuantileDiagonal.builder` → `term`, the "diagonal builder" prose in
  docstrings / `docs/src/observation_recipe.md` / NEWS, and the
  `diagonal_builder.jl` file names (`src/`, `ext/`, `test/`).
- Avoid "Estimator" in any new name: `AbstractCovarianceEstimator` already
  means "the thing you pass to `observation`", and terms deliberately are not
  that.

### Narrow the signature and return a vector

New contract:

```julia
compute_diagonal(term, samples::AbstractMatrix, var_ranges) -> AbstractVector
```

where `var_ranges` is a vector of `UnitRange` (one per variable's flattened
block), instead of `build_diagonal(builder, sample_collection) -> Diagonal`.

- **Why narrow:** the built-ins need only the samples matrix and the block
  structure (`ScalarDiagonal`: eltype + row count; `ModelErrorScaleDiagonal`:
  samples; `QuantileDiagonal`: samples + per-variable ranges — its metadata use
  is solely `_get_indices_of_metadata` plus a short name in an error message).
  The block structure is not incidental plumbing — `QuantileDiagonal` is
  *defined* per variable block — the smell is only that the term receives more
  than it needs. Narrowing moves all implementations from
  `ext/diagonal_builder.jl` into `src/`, testable without ClimaAnalysis, and
  resolves the current src/ext oddity (SumDiagonal's method in src, everything
  else in ext).
- **Why a vector:** the "return a diagonal matrix" contract forced ad hoc
  enforcement at every seam — `isdiag` + size checks in
  `covariance(::SVDplusDCovariance, ...)`, a second `isdiag` inside
  `QuantileDiagonal` that immediately unwraps with `diag`, and a
  dense-to-`Diagonal` conversion path (with a `DenseDiagonal` test struct
  existing only to exercise it). With a vector: size check becomes a length
  check, `SumDiagonal` is vector `+`, `QuantileDiagonal` stops
  wrapping/unwrapping, and `covariance` wraps in `Diagonal` exactly once at
  the end.
- The extension keeps a thin shim
  `compute_diagonal(term, sc::SampleCollection)` that extracts samples and
  ranges and forwards. Custom terms that genuinely need metadata (dates,
  latitudes) can still override at the `SampleCollection` level, so no power
  is lost.

### Central eltype conversion

Currently each term casts to `eltype(samples)`, the docstrings say so three
separate times, and nothing enforces it — a custom term (including the
`VarianceDiagonal` example in the docs) can silently return Float64 against
Float32 samples and ship an `EKP.SVDplusD` with mixed eltypes. Instead:
convert once where the diagonal is consumed (in `covariance`) and delete the
per-term responsibility.

### `SeasonalDiagonalCovariance` uses the term machinery (agreed)

It currently reimplements model-error-scale and regularization inline in
`ext/observation_recipe.jl` (`covariance(::SeasonalDiagonalCovariance, ...)`),
so the codebase has both the old and the new way of expressing the same two
concepts. It already works in diagonal-vector space, which fits the
vector-returning contract. This is also the reuse that justifies the term
abstraction beyond `SVDplusDCovariance`.

### Per-variable values (new feature)

Terms accept either a single value (same for all variables) or a vector with
one value per variable (filling that variable's block); a length mismatch is
an error.

- Strengthens the case for `(samples, var_ranges)`: with this feature every
  term needs the block structure, not just `QuantileDiagonal`.
- Validation splits: constructors check elementwise non-negativity / quantile
  range; the length check must happen in `compute_diagonal` (nvars unknown at
  construction). With no metadata in the narrow signature, the mismatch error
  states expected vs. actual counts, not variable names; the `SampleCollection`
  shim is the place to enrich with names if counts prove insufficient.
- One shared helper (validate length, expand a scalar to per-block values) so
  the scalar-or-vector handling is not implemented three times.
- Struct shape: parametrize on the stored type, e.g.
  `T <: Union{AbstractFloat, AbstractVector{<:AbstractFloat}}` — cannot
  normalize to a vector at construction because nvars is unknown.
- Subsumes the `PerVariableScalar` custom-estimator example in the docs; that
  example should demonstrate something the built-ins genuinely can't do, or
  acknowledge the feature is now built in.

## Review findings still to settle

Valid but not yet decided on exact treatment:

1. **`QuantileDiagonal` guard rails.** `qtl_for_var ≈ 0.0` with default
   tolerances is only true for *exactly* zero (`isapprox` against zero uses
   `atol = 0`), so the "might be too small" check doesn't do what it says.
   `length(var_diag_vec) < 1 / qtl` errors with "Insufficient samples for
   computing quantile", but it counts diagonal entries per variable, not
   samples, and the message carries no context (which variable, how many
   entries, how many required). Open question: should a
   well-defined-but-statistically-thin quantile be a hard error at all, or a
   warning?
2. **Default `diagonal` for `SVDplusDCovariance()`.** Currently
   `ModelErrorScaleDiagonal(0.0) + ScalarDiagonal(0.0)` — a two-term sum of
   zeros that shows up when printing the struct and does two builds to produce
   a zero vector, handing EKP a `SVDplusD` with `D = 0`. `ScalarDiagonal(0.0)`
   alone is the honest default. Whether a zero `D` should warn is a separate
   (pre-existing) question.
3. **Test gaps.** All `test/diagonal_builder.jl` tests use a single variable,
   so the per-variable block logic — the entire point of `QuantileDiagonal` —
   is untested with ≥ 2 variables, as are the two error paths above. New tests
   needed for multi-variable blocks, per-variable values, and mismatch/error
   paths.
4. **Smaller cleanups.** `covariance(::SVDplusDCovariance, ...)` copies the
   sample matrix even when `use_latitude_weights = false`; the `diagonal`
   kwarg accepts anything and validates with a manual `isa` check where a
   typed keyword would do.

## Open questions

- Rename `ScalarDiagonal` → `ConstantDiagonal`? Once it can hold a vector,
  `ScalarDiagonal([1e-6, 1e-3])` is an oxymoron; `ConstantDiagonal` reads
  correctly in both forms (one constant everywhere, or one constant per
  variable's block). Decide during the rename sweep.
- Does `QuantileDiagonal` get the same scalar-or-vector treatment for its
  quantile? (Recommended yes, for consistency.)
- Do the legacy `model_error_scale` / `regularization` kwargs on
  `SVDplusDCovariance` accept vectors? (They lower to the terms, so it's free;
  recommended yes.)
- Long term: the legacy kwargs are a permanently duplicated API surface
  (documented in the docstring, the doc section, and the FAQ). Deliberate for
  back-compat, but worth a deprecation plan or at least one canonical doc
  location.

## Atomic change sequence

Each step is self-contained and leaves the tests green, so they can be applied
(and reviewed) one at a time in this order. Steps marked *(independent)* can
be reordered freely; the rest build on what precedes them.

1. **Fill the test gaps against current behavior.** Multi-variable
   `QuantileDiagonal` blocks (≥ 2 variables) and the two error paths
   (insufficient entries, zero quantile). No source changes — this is the
   safety net for every refactor that follows.
2. **Rename sweep (mechanical, no behavior change).**
   `AbstractDiagonalBuilder` → `AbstractDiagonalTerm`, `build_diagonal` →
   `compute_diagonal`, `SumDiagonal.builders` → `terms`,
   `QuantileDiagonal.builder` → `term`; file renames
   (`diagonal_builder.jl` → e.g. `diagonal_term.jl` in `src/`, `ext/`,
   `test/`); "diagonal builder" prose in docstrings,
   `docs/src/observation_recipe.md`, and the NEWS entry. If the
   `ScalarDiagonal` → `ConstantDiagonal` rename is a yes (decide before this
   step), include it here to avoid a second sweep.
3. **Vector-returning contract.** `compute_diagonal` returns the diagonal
   entries as an `AbstractVector` (still taking a `SampleCollection` for now).
   `covariance(::SVDplusDCovariance, ...)` wraps in `Diagonal` exactly once;
   delete the `isdiag`/size/dense-conversion enforcement, `QuantileDiagonal`'s
   inner `isdiag` + `diag` unwrapping, and the `DenseDiagonal` test struct;
   `SumDiagonal` becomes vector `+`.
4. **Narrow the signature.** `compute_diagonal(term, samples, var_ranges)`;
   move all implementations from `ext/` to `src/`; add the thin
   `SampleCollection` shim in the extension; move the term tests off
   ClimaAnalysis where possible.
5. **Central eltype conversion.** Convert once in `covariance`; delete the
   per-term casts, the three docstring notes, and the per-term eltype test
   loop (replace with one test at the covariance level).
6. **Quantile guard-rail fixes.** Replace the exact-zero-only `≈ 0.0` check
   with an explicit intent, rewrite the "insufficient samples" message
   (entries per variable, with counts and context), and settle error-vs-warn
   for statistically thin quantiles. *(independent — needs the open decision)*
7. **Simpler default diagonal.** `SVDplusDCovariance()` defaults to a single
   zero term instead of `ModelErrorScaleDiagonal(0.0) + ScalarDiagonal(0.0)`.
   *(independent)*
8. **Per-variable values.** Shared expand-and-validate helper; scalar-or-vector
   fields on the constant and model-error-scale terms (and the quantile of
   `QuantileDiagonal`, if agreed); length-mismatch error in
   `compute_diagonal`; vector pass-through for the legacy
   `model_error_scale`/`regularization` kwargs (if agreed); tests for
   per-variable values and mismatch paths; rework the `PerVariableScalar`
   docs example. Requires step 4 (`var_ranges` in the signature).
9. **Port `SeasonalDiagonalCovariance` to the term machinery.** Replace its
   inline model-error-scale/regularization with the equivalent terms. Requires
   steps 3–4; existing seasonal tests pin the behavior.
10. **Small cleanups.** Skip the sample-matrix copy when
    `use_latitude_weights = false`; type the `diagonal` keyword.
    *(independent)*
11. **Final docs/NEWS pass.** Reconcile `observation_recipe.md`, the API
    reference, and the NEWS entry with everything above (each step updates its
    own docs; this is the consistency check).

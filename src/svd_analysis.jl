import LinearAlgebra: SVD, Diagonal, norm, dot, UniformScaling
import LinearMaps: LinearMap
import ArnoldiMethod: partialschur, partialeigen
import Statistics: mean

export compute_structured_energy,
    compute_structured_energy_by_variable,
    compute_normalized_projections,
    analyze_residual

# TODO: Use the function in EKP once it's moved there
"""
    _create_compact_linear_map(A)

Wrap an SVD-structured covariance matrix (or a vector of them) as a `LinearMap`
for use with matrix-free eigensolvers such as `ArnoldiMethod.partialschur`.

Supports `LinearAlgebra.SVD`, `LinearAlgebra.Diagonal`, and `EKP.SVDplusD`
inputs. If `A` is a vector, the matrices are treated as block-diagonal.
"""
function _create_compact_linear_map(A)
    Avec = isa(A, AbstractVector) ? A : [A]

    Us = []
    Ss = []
    VTs = []
    ds = []
    batches = []
    shift = 0
    for a in Avec
        if isa(a, UniformScaling)
            throw(
                ArgumentError(
                    "Detected `UniformScaling` (i.e. \"λI\") StructureMatrix, and unable to infer dimensionality. \n Please recast this as a diagonal matrix, defining \"λI(d)\" for dimension d",
                ),
            )
        elseif isa(a, SVD)
            push!(Us, a.U)
            push!(Ss, a.S)
            push!(VTs, a.Vt)
            push!(ds, zeros(size(a.U, 1)))
            bsize = size(a.U, 1)
        elseif isa(a, Diagonal)
            # No low-rank component; represent as empty U, S, Vt so the
            # LinearMap formula U*(S.*(Vt*x)) + d.*x reduces to just d.*x
            n = length(a.diag)
            push!(Us, zeros(n, 0))
            push!(Ss, zeros(0))
            push!(VTs, zeros(0, n))
            push!(ds, a.diag)
            bsize = n
        elseif isa(a, EKP.SVDplusD)
            svda = a.svd_cov
            diaga = (a.diag_cov).diag
            push!(Us, svda.U)
            push!(Ss, svda.S)
            push!(VTs, svda.Vt)
            push!(ds, diaga)
            bsize = length(diaga)
        else
            throw(
                ArgumentError(
                    "Unsupported matrix type: $(typeof(a)). Expected SVD, Diagonal, or EKP.SVDplusD.",
                ),
            )
        end

        batch = (shift + 1):(shift + bsize)
        push!(batches, batch)
        shift = batch[end]
    end

    Amap = LinearMap(
        x -> reduce(
            vcat,
            [
                U * (S .* (Vt * x[batch])) + d .* x[batch] for
                (U, S, Vt, d, batch) in zip(Us, Ss, VTs, ds, batches)
            ],
        ),
        x -> reduce(
            vcat,
            [
                Vt' * (S .* (U' * x[batch])) + d .* x[batch] for
                (U, S, Vt, d, batch) in zip(Us, Ss, VTs, ds, batches)
            ],
        ),
        sum(size(U, 1) for U in Us),
        sum(size(Vt, 2) for Vt in VTs),
    )

    return Amap
end

"""
    compute_structured_energy(projections)

Given the matrix of normalized projections from `compute_normalized_projections`,
compute the total structured energy in the whitened space:

    energy = (1/n_eig) * ∑ᵢ zᵢ²,   where zᵢ = ∑ᵥ projections[i, v] = aᵢ / √λᵢ

`zᵢ` is the global whitened projection onto eigenvector `i`. Under the noise
model, `zᵢ ~ N(0, 1)`, so `energy ≈ 1` is consistent with noise. Values >> 1
indicate mismatch beyond what the structured noise explains. Values << 1
suggest overfitting to noise or an overestimated noise covariance.
"""
function compute_structured_energy(projections)
    z = sum(projections, dims = 2)
    return sum(z .^ 2) / length(z)
end

"""
    compute_structured_energy_by_variable(projections)

Given the matrix of normalized projections from `compute_normalized_projections`,
compute the per-variable structured energy in the whitened space:

    energy_v = (1/n_eig) * ∑ᵢ projections[i, v]²

Returns a vector of length `n_variables`. Values >> 1 for variable `v` indicate
that variable's contribution to the eigenvector projections exceeds noise-model
predictions. Values ≈ 1 are consistent with noise, and values << 1 suggest
overfitting or an overestimated noise covariance for that variable.

See also [`analyze_residual`](@ref).
"""
function compute_structured_energy_by_variable(projections)
    n_eig = size(projections, 1)
    return [sum(projections[:, v] .^ 2) / n_eig for v in axes(projections, 2)]
end

"""
    compute_normalized_projections(diff, eigvectors, eigvalues, ranges)

For each eigenvector `i` and variable `v`, compute the variable-specific
contribution to the normalized projection `aᵢᵛ / √λᵢ`, where
`aᵢᵛ = vᵢ[rᵥ]ᵀ diff[rᵥ]`.

This is a z-score: values >> 1 indicate model-data mismatch beyond noise;
values ≤ 1 are consistent with structured noise.

Returns a `Matrix` of shape `(n_eigenvectors, n_variables)`. `ranges` is a
vector of index ranges, one per variable, giving the observation indices for
that variable. Pass the result to `compute_structured_energy` or
`compute_structured_energy_by_variable` for scalar summaries.

See also [`analyze_residual`](@ref).
"""
function compute_normalized_projections(diff, eigvectors, eigvalues, ranges)
    n_eigenvectors = size(eigvectors, 2)
    n_variables = length(ranges)
    projections = zeros(n_eigenvectors, n_variables)
    for (v, r_v) in enumerate(ranges)
        for i in 1:n_eigenvectors
            # aᵢᵛ: variable v's contribution to projection onto eigenvector i
            a_i_v = dot(eigvectors[r_v, i], diff[r_v])
            projections[i, v] = a_i_v / sqrt(eigvalues[i])
        end
    end
    return projections
end

"""
    analyze_residual(ekp, iter; n_eigenvectors = 3)

Analyze the model-data residual (y - G(u)) at iteration `iter` of an EKP calibration
using the top eigenvectors of the noise covariance.

The noise covariance is obtained via `EKP.get_obs_noise_cov` with `build = false`,
so it works with any `StructuredMatrix` type supported by EKP (`SVD`, `Diagonal`,
`SVDplusD`), not only `SVDplusD`.

Returns a named tuple with:
- `normalized_projections`: `(n_eigenvectors × n_variables)` matrix of z-scores
  per variable (values >> 1 indicate mismatch beyond noise)
- `structured_energy`: normalized whitened energy across all variables (≈ 1 under noise model)
- `structured_energy_by_variable`: per-variable whitened energy
- `residual_norm_by_variable`: `norm(diff[rᵥ])` for each variable
- `metadata`: vector of `ClimaAnalysis.Var.Metadata` for each variable, in the
  same order as the columns of `normalized_projections` and elements of
  `structured_energy_by_variable` and `residual_norm_by_variable`

Requires ClimaAnalysis to be loaded.
"""
function analyze_residual(ekp, iter; n_eigenvectors = 3)
    obs_series = EKP.get_observation_series(ekp)
    N_iters = EKP.get_N_iterations(ekp)
    obs_noise_cov = EKP.get_obs_noise_cov(obs_series, N_iters; build = false)
    linear_map = _create_compact_linear_map(obs_noise_cov)
    result, _ = partialschur(linear_map; nev = n_eigenvectors)
    eigvalues, eigvectors = partialeigen(result)

    # Reverse so index 1 = dominant eigenvector
    reverse!(eigvalues)
    reverse!(eigvectors, dims = 2)

    g = EKP.get_g(ekp, iter)
    succ_ens, _ = EKP.split_indices_by_success(g)
    mean_g = dropdims(mean(g[:, succ_ens], dims = 2), dims = 2)
    diff = EKP.get_obs(ekp, iter) - mean_g

    metadata = EKPUtils.get_metadata_for_nth_iteration(obs_series, iter)
    ranges = ObservationRecipe._get_minibatch_indices_for_nth_iteration(
        obs_series,
        iter,
    )
    projections =
        compute_normalized_projections(diff, eigvectors, eigvalues, ranges)
    return (;
        normalized_projections = projections,
        structured_energy = compute_structured_energy(projections),
        structured_energy_by_variable = compute_structured_energy_by_variable(
            projections,
        ),
        residual_norm_by_variable = map(r -> norm(diff[r]), ranges),
        metadata,
    )
end

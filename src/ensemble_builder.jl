"""
    ClimaCalibrate.EnsembleBuilder

Assemble the G ensemble matrix from `ClimaAnalysis.OutputVar`s.

[`GEnsembleBuilder`](@ref) reads the metadata off the observations in an
`EnsembleKalmanProcess` and works out where each variable belongs in the matrix,
so index ranges do not have to be tracked by hand. It validates each `OutputVar`
against the observation it is filling in, checking short name, units, dimension
names, dimension units, and dimension values. A mismatch between model output
and observations raises an error instead of being calibrated against silently.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
module EnsembleBuilder

export GEnsembleBuilder,
    fill_g_ens_col!,
    is_complete,
    get_g_ensemble,
    ranges_by_short_name,
    metadata_by_short_name,
    missing_short_names

"""
    GEnsembleBuilder(ekp)

Create a builder for the G ensemble matrix of `ekp`.

The builder reads the metadata off the observations that `ekp` is being scored
against, so it knows which rows of the matrix each variable occupies. Fill it
column by column with [`fill_g_ens_col!`](@ref), then hand the matrix to EKP
with [`get_g_ensemble`](@ref).

# Examples
```julia
import ClimaAnalysis, NaNStatistics
builder = ClimaCalibrate.EnsembleBuilder.GEnsembleBuilder(ekp)
for member in 1:EKP.get_N_ens(ekp)
    for var in preprocess_member_output(member)
        ClimaCalibrate.EnsembleBuilder.fill_g_ens_col!(builder, member, var)
    end
end
G_ensemble = ClimaCalibrate.EnsembleBuilder.get_g_ensemble(builder)
```

See also [`is_complete`](@ref), [`missing_short_names`](@ref).
"""
function GEnsembleBuilder end

"""
    fill_g_ens_col!(builder, col_idx, var; checkers = ..., verbose = false)
    fill_g_ens_col!(builder, col_idx, value::AbstractFloat)

Fill the part of ensemble member `col_idx`'s column that `var` corresponds to,
and return whether `var` was used.

`var` is matched against the observation metadata by short name, and validated
against it by the `checkers` before anything is written, so model output that
does not line up with the observation is rejected instead of being calibrated
against silently. Pass `verbose = true` to have each failed check say why.

The second form fills the member's whole column with a single value, which is
how a failed forward model is marked (with `NaN`).

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function fill_g_ens_col! end

"""
    is_complete(builder)

Return `true` once every entry of the G ensemble matrix has been filled.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function is_complete end

"""
    get_g_ensemble(builder)

Return the G ensemble matrix, with `NaN` wherever nothing was filled in.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function get_g_ensemble end

"""
    ranges_by_short_name(builder)

Return the row ranges of the G ensemble matrix, keyed by short name.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function ranges_by_short_name end

"""
    metadata_by_short_name(builder)

Return the observation metadata the builder is matching against, keyed by short
name.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function metadata_by_short_name end

"""
    missing_short_names(builder, col_idx)

Return the short names that ensemble member `col_idx` is still missing.

This is the first thing to check when [`is_complete`](@ref) returns `false`.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function missing_short_names end

end

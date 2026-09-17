"""
    ClimaCalibrate.EnsembleBuilder

Assemble the G ensemble matrix from `ClimaAnalysis.OutputVar`s.

[`GEnsembleBuilder`](@ref) reads the metadata off the observations in an
`EnsembleKalmanProcess` and works out where each variable belongs in the matrix,
so index ranges do not have to be tracked by hand. It validates each `OutputVar`
against the observation it is filling in, checking short name, units, dimension
names, dimension units, and dimension values. A mismatch between model output
and observations raises an error instead of being calibrated against silently.

Requires ClimaAnalysis to be loaded.
"""
module EnsembleBuilder

export GEnsembleBuilder,
    fill_g_ens_col!,
    is_complete,
    get_g_ensemble,
    ranges_by_short_name,
    metadata_by_short_name,
    missing_short_names

function GEnsembleBuilder end

function fill_g_ens_col! end

function is_complete end

function get_g_ensemble end

function ranges_by_short_name end

function metadata_by_short_name end

function missing_short_names end

end

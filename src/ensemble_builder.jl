module EnsembleBuilder

export GEnsembleBuilder, fill_g_ens_col!, is_complete, get_g_ensemble

function GEnsembleBuilder end

function fill_g_ens_col! end

function is_complete end

function get_g_ensemble end

function ranges_by_short_name end

function metadata_by_short_name end

function missing_short_names end


"""
    abstract type AbstractChecker end

An object that checks for things between the simulational data and metadata
from the observational data. This is used by `GEnsembleBuilder` to check
`OutputVar`s from simulation data and the `Metadata` in the observations in the
`EnsembleKalmanProcess` object.

`AbstractChecker` have to provide one function, `EnsembleBuilder.check`.

The function has to have the signature

```julia
EnsembleBuilder.check(
    checker::AbstractChecker;
    verbose = false
)
```

and return `true` or `false`.
"""
abstract type AbstractChecker end

struct ShortNameChecker <: AbstractChecker end

struct DimNameChecker <: AbstractChecker end

struct DimUnitsChecker <: AbstractChecker end

struct UnitsChecker <: AbstractChecker end

struct DimValuesChecker <: AbstractChecker end

"""
    check(checker::AbstractChecker; verbose = false)

Return `true` if the check passes, `false` otherwise.

If `verbose=true`, then provides information for why a check did not succeed.
"""
function check(checker::AbstractChecker, var, metadata; verbose = false)
    error("Not yet implemented!")
end

end

module EnsembleBuilder

export GEnsembleBuilder, fill_g_ens_col!, is_complete, get_g_ensemble

function GEnsembleBuilder end

function fill_g_ens_col! end

function is_complete end

function get_g_ensemble end

function ranges_by_short_name end

function metadata_by_short_name end

function missing_short_names end

# TODO: Make another module and put this in it

"""
    abstract type AbstractMatcher end

An object that matches for things between the simulational data and metadata
from the observational data. This is used by `GEnsembleBuilder` to match
`OutputVar`s from simulation data to `Metadata` in the observations in the
`EnsembleKalmanProcess` object.

`AbstractMatcher` have to provide one function, `EnsembleBuilder.match`.

The function has to have the signature

```julia
EnsembleBuilder.match(
    matcher::AbstractMatcher;
    verbose = false
)
```

and return `true` or `false`.
"""
abstract type AbstractMatcher end

struct ShortNameMatcher <: AbstractMatcher end

struct DimNameMatcher <: AbstractMatcher end

struct DimUnitsMatcher <: AbstractMatcher end

struct UnitsMatcher <: AbstractMatcher end

struct DimValuesMatcher <: AbstractMatcher end

struct SignMatcher <: AbstractMatcher end

struct SequentialIndicesMatcher <: AbstractMatcher end

"""
    match(matcher::AbstractMatcher; verbose = false)

Return `true` if the match passes, `false` otherwise.

When `verbose=true`, provides information for why a match did not succeed.
"""
function match(matcher::AbstractMatcher, var, metadata; verbose = false)
    error("Not yet implemented!")
end

end

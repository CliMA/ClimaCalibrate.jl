module Checker

"""
    abstract type AbstractChecker end

An object that performs validation checks between the simulation data and
metadata from observational data. This is used by `GEnsembleBuilder` to validate
`OutputVar`s from simulation data against the `Metadata` in the observations in
the `EnsembleKalmanProcess` object.

An `AbstractChecker` must implement the `Checker.check` function.

The function must have the signature:

```julia
import ClimaCalibrate.Checker
Checker.check(::YourChecker,
              var::OutputVar,
              metadata::Metadata;
              data = nothing,
              verbose = false)
```

and return `true` or `false`.

!!! note "What is var and metadata?"

    For more information about `OutputVar` and `Metadata`, see the ClimaAnalysis
    [documentation](https://clima.github.io/ClimaAnalysis.jl/dev/).

"""
abstract type AbstractChecker end

"""
    struct ShortNameChecker <: AbstractChecker end

A struct that checks the short name between simulation data and metadata.
"""
struct ShortNameChecker <: AbstractChecker end

"""
    struct DimNameChecker <: AbstractChecker end

A struct that checks the dimension names between simulation data and metadata.
"""
struct DimNameChecker <: AbstractChecker end

"""
    struct DimUnitsChecker <: AbstractChecker end

A struct that checks the units of the dimensions between simulation data and
metadata.
"""
struct DimUnitsChecker <: AbstractChecker end

"""
    struct UnitsChecker <: AbstractChecker end

A struct that checks the units between the simulation data and metadata.
"""
struct UnitsChecker <: AbstractChecker end

"""
    struct DimValuesChecker <: AbstractChecker end

A struct that checks the values of the dimensions between the simulation data
and metadata.
"""
struct DimValuesChecker <: AbstractChecker end

"""
    struct SequentialIndicesChecker <: AbstractChecker end

A struct that checks that the indices of the dates of the simulation data
corresponding to the dates of the metadata is sequential.
"""
struct SequentialIndicesChecker <: AbstractChecker end

"""
A struct that checks that the proportion of positive values in the simulation
data and observational data is roughly the same.
"""
struct SignChecker <: AbstractChecker end

"""
    check(checker::AbstractChecker; verbose = false)

"""
    check(checker::AbstractChecker,
          var,
          metadata;
          data = nothing,
          verbose = false)

Return `true` if the check passes, `false` otherwise.

If `verbose=true`, then provides information for why a check did not succeed.
"""
function check(
    checker::AbstractChecker,
    var,
    metadata;
    data = nothing,
    verbose = false,
)
    error("Not yet implemented!")
end

end

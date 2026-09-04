"""
    ClimaCalibrate.SampleBuilder

Turn `ClimaAnalysis.OutputVar`s into a matrix of flattened samples.

This is the first of the two steps in building an observation: `SampleBuilder`
produces a `SampleCollection`, and [`ClimaCalibrate.ObservationRecipe`](@ref)
then estimates a noise covariance from it and assembles the `EKP.Observation`.

Each column of the collection is one sample, and each carries the metadata
needed to reconstruct the `OutputVar`s later.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
module SampleBuilder

export build_samples,
    build_samples_by_times,
    num_samples,
    reconstruct_col,
    get_samples,
    get_metadata

"""
    build_samples(vars; FT = Float32, dims = ...)

Build a `SampleCollection` from `ClimaAnalysis.OutputVar`s.

Accepts a single `OutputVar` or a `Vector` of them (one sample made of one or
more variables), or a `Matrix` whose rows are variables and whose columns are
samples. The `Matrix` method also takes `ignore_dims`, to exclude dimensions
from the compatibility checks between samples.

Each variable is flattened in the order given by `dims`, dropping `NaN`s, and
the same coordinates must be dropped in every sample.

# Examples
```julia
import ClimaAnalysis, NaNStatistics
samples = ClimaCalibrate.SampleBuilder.build_samples([ta, hus]; FT = Float64)
```

See also [`build_samples_by_times`](@ref).
"""
function build_samples end

"""
    build_samples_by_times(vars, time_ranges; FT = Float32, dims = ...)

Build a `SampleCollection` by windowing `vars` into one sample per time range.

Each element of `time_ranges` is a `(start, stop)` pair of dates or times, and
becomes one column of the collection. The time dimension is excluded from the
between-sample compatibility checks, since each sample covers a different span.

Windows should not overlap: samples that share time slices are correlated, which
biases a covariance estimated from them.

# Examples
```julia
import ClimaAnalysis, NaNStatistics, Dates
ranges = [
    (Dates.DateTime(y, 12, 1), Dates.DateTime(y + 1, 9, 1)) for y in 2007:2015
]
samples = ClimaCalibrate.SampleBuilder.build_samples_by_times([ta], ranges)
```

See also [`build_samples`](@ref).
"""
function build_samples_by_times end

"""
    num_samples(sample_collection)

Return the number of samples (columns) in a `SampleCollection`.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function num_samples end

"""
    reconstruct_col(sample_collection, i)

Return the `i`th sample as a vector of `ClimaAnalysis.OutputVar`s.

This undoes the flattening that [`build_samples`](@ref) applied, so a sample
can be inspected or plotted.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function reconstruct_col end

"""
    get_samples(sample_collection)

Return the matrix of flattened samples, one sample per column.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function get_samples end

"""
    get_metadata(sample_collection)

Return the matrix of `ClimaAnalysis.Var.Metadata`, one entry per variable per
sample.

Requires ClimaAnalysis and NaNStatistics to be loaded.
"""
function get_metadata end

end

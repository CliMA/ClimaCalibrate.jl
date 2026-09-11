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

function build_samples end

function build_samples_by_times end

function num_samples end

function reconstruct_col end

function get_samples end

function get_metadata end

end

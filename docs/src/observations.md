# Observations

Robust observations and accurate error covariances are essential for successful calibration. When calibrating climate models, it is advisable to use long-term climate statistics, such as monthly or seasonal averages, to reduce the influence of internal variability. This results in a more stable and representative target for inversion.

`EnsembleKalmanProcesses.jl` provides several containers for managing observations, with documentation provided [here](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/observations/).
As inputs to a calibration, observations can consist of a `Vector`, an `EKP.Observation` (a single observation), or an `EKP.ObservationSeries` (many observations).

To iterate through an `EKP.ObservationSeries`, you must provide a minibatcher. This package provides two helper functions to faciliate the creation of simple batches:

- [`ClimaCalibrate.minibatcher_over_samples`](@ref) takes in samples or (a number of samples) and a batch size and returns a minibatcher which divides the samples into the batch size, dropping remaining samples.
- [`ClimaCalibrate.observation_series_from_samples`](@ref) takes in a vector of `Observation`s and a batch size and returns an `ObservationSeries` with a minibatcher.

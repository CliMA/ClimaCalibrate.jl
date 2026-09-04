```@meta
CurrentModule = ClimaCalibrate
```

# Observations

An observation is two things: a vector of values the model should reproduce, and
a covariance matrix saying how much of any mismatch to attribute to error rather
than to the parameters.

The covariance does two things. It sets the relative weighting of the
observation entries, so it decides which parts of the data the calibration fits.
Its scale sets how far the ensemble moves in response to a given residual. Too
large and the ensemble barely updates; too small and it collapses within an
iteration or two onto whatever the initial draw favored.

When calibrating climate models, use long-term statistics such as monthly or
seasonal averages, not instantaneous fields. A climate model and the real
atmosphere do
not share a trajectory, only a distribution, so a single time slice differs from
the observation by an amount that no parameter choice can remove. Averaging over
a season or a year reduces that internal variability and leaves a target the
parameters can move.

## Building one

`EnsembleKalmanProcesses.jl` provides the containers: an `EKP.Observation` for a
single observation and an `EKP.ObservationSeries` for many, documented in the
[EKP observations guide](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/observations/).
A plain `Vector` also works if you supply the covariance separately.

If your data is already in `ClimaAnalysis.OutputVar`s, ClimaCalibrate can build
the observation for you, in two steps:

1. [Building samples](sample_builder.md) turns `OutputVar`s into a
   `SampleCollection`: a matrix whose columns are samples and whose rows are the
   flattened variables, with the metadata needed to reconstruct them.
2. [Building observations](observation_recipe.md) estimates a noise covariance
   from those samples and assembles the `EKP.Observation`.

Then [building the G ensemble matrix](ensemble_builder.md) fills that matrix
from your model output. It uses the observation's own metadata to work out where
each variable goes, which keeps the ordering of the model output and the
ordering of the observation from silently diverging.

## Ordering

The `i`th entry of each column of the G ensemble matrix must correspond to the
`i`th entry of the observation. Getting this wrong is the most common way for a
calibration to run to completion and mean nothing: nothing errors, and the
residual stays large for every parameter choice. Using `ObservationRecipe` with
`GEnsembleBuilder` avoids it. If you build the matrix by hand, check the
ordering against `ObservationRecipe.short_names(obs)`.

## Minibatching

With more observations than you want to score against in one iteration, an
`EKP.ObservationSeries` and a minibatcher divide them into subsets, one per
iteration. Two helpers construct fixed-size batches:

- [`ClimaCalibrate.minibatcher_over_samples`](@ref) takes samples (or a number of
  samples) and a batch size, and returns a minibatcher that divides them into
  batches of that size, dropping any remainder.
- [`ClimaCalibrate.observation_series_from_samples`](@ref) takes a vector of
  `Observation`s and a batch size and returns an `ObservationSeries` with such a
  minibatcher.

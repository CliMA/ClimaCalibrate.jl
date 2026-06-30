ClimaCalibrate.jl Release Notes
========================

main
-------

- Add `reconstruct_g` for reconstructing the G ensemble matrix as a matrix of
  `OutputVar`s and `reconstruct_g_mean` for reconstructing the mean of the G
  ensemble matrix as a vector of `OutputVar`s
  [#319](https://github.com/CliMA/ClimaCalibrate.jl/pull/319)
- Add "How do I?" section in the documentation
  [#330](https://github.com/CliMA/ClimaCalibrate.jl/pull/330)
- Add `ClimaCalibrateMakie` extension for plotting ensemble members, the mean
  forward map evaluation, and the observations
  [#331](https://github.com/CliMA/ClimaCalibrate.jl/pull/331)
- Add the `SampleBuilder` module and refactor `ObservationRecipe`
  [#334](https://github.com/CliMA/ClimaCalibrate.jl/pull/334)
  - The `SampleBuilder` module handles transforming one or more
    `ClimaAnalysis.OutputVar`s into a matrix of samples with metadata.
  - Building an observation is now a two-step process: use `SampleBuilder` to
    turn `ClimaAnalysis.OutputVar`s into an `ObservedSampleCollection`, then
    pass that to `ObservationRecipe` to estimate the covariance and build the
    `EKP.Observation`.
  - **Breaking**: `ObservationRecipe.observation` and
    `ObservationRecipe.covariance` now take an `ObservedSampleCollection` (e.g
    `observation(covar_estimator, osc)`) instead of
    `(covar_estimator, vars, start_date, end_date)`. The `dims` keyword (flatten
    order) is moved to `SampleBuilder.generate_samples`.
  - **Breaking**: `SVDplusDCovariance` no longer takes `sample_date_ranges`.
    Instead, you can window the time series into samples with
    `SampleBuilder.generate_samples_by_times`.
  - **Breaking**: `SeasonalDiagonalCovariance` now estimates the variance across
    the sample columns and requires at least two samples. A single multi-year
    `OutputVar` is no longer accepted as one sample; split it into one sample
    per year with `generate_samples_by_times`. The `ignore_nan` keyword was
    removed (`NaN`s are always ignored).
  - **Breaking**: removed `ObservationRecipe.change_data_type`. The covariance
    inherits its element type from the samples, so set it with the `FT` keyword
    to `generate_samples`/`generate_samples_by_times` (default `Float32`); pass
    `FT = Float64` if you need `Float64`.

v0.3.1
-------

- Add `dims` keyword argument to observation and covariance constructors in
  the `ObservationRecipe` module [#318](https://github.com/CliMA/ClimaCalibrate.jl/pull/318)

v0.3.0
-------

- Refactor codebase into three modules
  [#295](https://github.com/CliMA/ClimaCalibrate.jl/pull/295):
  - `EKPUtils`: standalone EKP utility functions with no dependency on the rest
    of ClimaCalibrate
  - `BackendManager`: handles job submission for `HPCBackend`s (Slurm/PBS
    scripts)
  - `Calibration`: orchestrates the calibration loop, using the `BackendManager`
    module for job dispatch
- Add `SlurmConfig` and `PBSConfig` structs for `HPCBackend`s, allowing users to
  specify job directives, environment variables, and modules to load
  [#303](https://github.com/CliMA/ClimaCalibrate.jl/pull/303)
- Add `AbstractModelInterface` abstract type; users subtype this to define their
  model interface struct
  [#312](https://github.com/CliMA/ClimaCalibrate.jl/pull/312)
- Improve general documentation
  [#314](https://github.com/CliMA/ClimaCalibrate.jl/pull/314)
- Update climacommon for `ClimaGPUBackend`
  [#289](https://github.com/CliMA/ClimaCalibrate.jl/pull/289)
- Add noise covariance and residual analysis tools
  [#286](https://github.com/CliMA/ClimaCalibrate.jl/pull/286)
- Remove unused functionality and cleanup
  [#293](https://github.com/CliMA/ClimaCalibrate.jl/pull/293),
  [#298](https://github.com/CliMA/ClimaCalibrate.jl/pull/298),
  [#302](https://github.com/CliMA/ClimaCalibrate.jl/pull/302),
  [#305](https://github.com/CliMA/ClimaCalibrate.jl/pull/305)
- Bug fix with analyze_residual
  [#301](https://github.com/CliMA/ClimaCalibrate.jl/pull/301)

v0.2.2
-------
- Add quantile regularization to the SVDPlusDCovariance
  [#277](https://github.com/CliMA/ClimaCalibrate.jl/pull/277)

v0.2.0
-------
- Refactor backend structs to store relevant information
  [#245](https://github.com/CliMA/ClimaCalibrate.jl/pull/245)

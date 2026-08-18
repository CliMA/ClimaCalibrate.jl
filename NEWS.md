ClimaCalibrate.jl Release Notes
========================

main
-------

- Fix a bug where `SampleBuilder.build_samples` did not work with `OutputVar`s
  with no dimensions. `SampleBuilder`, `ObservationRecipe`, and
  `EnsembleBuilder` now support `OutputVar`s with no dimensions. This requires
  versions of ClimaAnalysis after v0.5.23.

v0.4.0
-------

- Update minimum Julia version to 1.10
  [#340](https://github.com/CliMA/ClimaCalibrate.jl/pull/340)
- Add the `workers_per_node` keyword argument to `add_workers`, which runs
  multiple independent workers on a single allocation
  [#341](https://github.com/CliMA/ClimaCalibrate.jl/pull/341)
- Add the `SampleBuilder` module and refactor `ObservationRecipe`
  [#334](https://github.com/CliMA/ClimaCalibrate.jl/pull/334)
  - The `SampleBuilder` module handles transforming one or more
    `ClimaAnalysis.OutputVar`s into a matrix of samples with metadata.
  - Building an observation is now a two-step process: use `SampleBuilder` to
    turn `ClimaAnalysis.OutputVar`s into a `SampleCollection`, then pass a
    covariance estimator, the `SampleCollection`, and the index of the sample to
    use as the observation to `ObservationRecipe` to estimate the covariance and
    build the `EKP.Observation`.
  - **Breaking**: sample construction moved from the covariance estimators to
    `SampleBuilder`, changing the `ObservationRecipe` API:
    - `ObservationRecipe.observation` is now
      `observation(covar_estimator, sample_collection, i)` and
      `ObservationRecipe.covariance` is now
      `covariance(covar_estimator, sample_collection)`; both take a
      `SampleCollection` instead of `OutputVar`s and dates. The covariance
      matrix does not depend on which sample is chosen as the observation.
    - Keywords that control how samples are built moved to
      `SampleBuilder.build_samples` and `build_samples_by_times`: `dims`
      (flatten order) and the element type of the samples and their metadata
      (`FT`, default `Float32`, replacing the removed
      `ObservationRecipe.change_data_type`).
    - `SVDplusDCovariance` no longer takes `sample_date_ranges`; window the
      time series into samples with `SampleBuilder.build_samples_by_times`
      instead.
  - **Breaking**: `SeasonalDiagonalCovariance` now estimates the variance across
    the sample columns and requires at least two samples. A single multi-year
    `OutputVar` is no longer accepted as one sample; split it into one sample
    per year with `build_samples_by_times`. The `ignore_nan` keyword was
    removed (`NaN`s are always ignored).

v0.3.2
-------

- Add asynchronous workers
  [#338](https://github.com/CliMA/ClimaCalibrate.jl/pull/338)
- Add support for `OutputVar`s with no time dimension in `ObservationRecipe`
  and `GEnsembleBuilder`
  [#320](https://github.com/CliMA/ClimaCalibrate.jl/pull/320)
- Add `reconstruct_g` for reconstructing the G ensemble matrix as a matrix of
  `OutputVar`s and `reconstruct_g_mean` for reconstructing the mean of the G
  ensemble matrix as a vector of `OutputVar`s
  [#319](https://github.com/CliMA/ClimaCalibrate.jl/pull/319)
- Add "How do I?" section in the documentation
  [#330](https://github.com/CliMA/ClimaCalibrate.jl/pull/330)
- Add `ClimaCalibrateMakie` extension for plotting ensemble members, the mean
  forward map evaluation, and the observations
  [#331](https://github.com/CliMA/ClimaCalibrate.jl/pull/331)

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

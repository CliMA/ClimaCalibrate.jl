ClimaCalibrate.jl Release Notes
========================

main
-------

### Breaking

- `calibrate` is now a single method for all backends instead of three
  near-identical ones. Backends select only how an iteration's ensemble members
  are run, via `run_iteration`. `run_iteration(::HPCBackend, ...)` now takes the
  same arguments as the other backends
  (`backend, interface, iter, ensemble_size, output_dir`) and derives the
  experiment directory, model interface path, and exeflags from the interface.
- All backends now honor a `failure_rate`, the fraction of an iteration's
  ensemble members that may fail before the calibration is halted (default 0.5).
  `JuliaBackend` had no such setting and stopped only when *every* member
  failed, as did the `HPCBackend`s. Construct one with
  `JuliaBackend(; failure_rate = 1.0)` to restore that tolerance.
- `JuliaBackend` now writes the same member checkpoints as the other backends,
  so an interrupted iteration resumes instead of rerunning every member, and its
  output directory can be resumed by any backend.
- `HPCBackend`s now cancel their submitted jobs when the driver process exits.
  Killing a calibration used to leave the whole ensemble running and billing.
- `get_backend` is renamed `backend_type`, because it returns a backend *type*,
  not a backend that can be passed to `calibrate`. The old name still works and
  warns.
- `default_worker_pool` is renamed `calibration_worker_pool`. `Distributed`
  exports a function of the same name, so `using Distributed, ClimaCalibrate`
  made the unqualified name ambiguous.
- `YAML` and `Dates` are no longer dependencies. `Dates` remains a test
  dependency.
- The `SampleBuilder`, `ObservationRecipe`, `EnsembleBuilder`, and `Checker`
  modules are exported, alongside `Visualization`. `write_model_started` is
  exported, as `write_model_completed` already was.
- Calling a function that lives in a package extension without that extension
  loaded now reports which packages to load.
- Input validation that used `@assert` now throws `ArgumentError`. The
  `output_dir` check is gone.
- `HPCBackend`s take a `job_timeout`, the number of seconds an iteration waits
  for a running job before giving up (default 24 hours). The clock starts when a
  job leaves the queue, so a long wait for an allocation does not count
  against it.
- `initialize` errors when the `EnsembleKalmanProcess` given to a restart
  describes a different calibration than the stored one: a different ensemble
  size, observations, covariance shape, or process. A restart uses the stored
  process, so the two have to agree.
- `terminated_iteration` reports the iteration at which the scheduler
  terminated a calibration.

### Bug fixes

- **Slurm backends could not report a failed ensemble member.** Every generated
  batch script ended with `exit 0`, so the job always exited successfully, and
  `job_status` queried `squeue`, which stops listing a job once it finishes and
  therefore reported every finished job as `COMPLETED`. A crashed ensemble was
  indistinguishable from a successful one on `CaltechHPCBackend`,
  `ClimaGPUBackend`, and `GCPBackend`: no member log was printed, no warning was
  raised, and the calibration went on to update the ensemble. Job state is now
  read from `sacct` once the job leaves the queue, an unrecognized state is
  treated as a failure rather than a success, and a member that never wrote a
  "completed" checkpoint counts as failed regardless of what the scheduler said.
- `initialize` no longer overwrites the first iteration when restarting. It used
  to rewrite `iteration_001`'s `parameters.toml` and `eki_file.jld2` from the
  `EnsembleKalmanProcess` passed in. Since that object is usually rebuilt on
  restart with a fresh random initial ensemble, the update then paired the
  forward model output of the checkpointed members with parameters those members
  never ran with.
- `load_latest_ekp` returned `nothing` for every directory that `calibrate`
  produces: it probed `iteration_000`, but iterations are numbered from one.
- `analyze_residual` read the observational noise covariance
  from the last iteration instead of the requested one, so with minibatching it
  projected the residual onto the eigenvectors of a different observation.
- `get_observations_for_nth_iteration` and the metadata and index helpers built
  on it wrapped the iteration back into the first epoch, so from the second
  epoch on a shuffling minibatcher gave `analyze_residual` and
  `GEnsembleBuilder` the variables of a different minibatch. They now use the
  minibatch EKP recorded for that iteration, and a minibatcher that repeats each
  epoch still answers for an iteration that has not run yet.
- A `squeue` that could not reach the controller returned no state, which read
  as a job that had left the queue and, with no `sacct` to consult, as a
  completed one. A member still running was then counted as failed by the
  checkpoint cross-check. A failed `squeue` now leaves the job as running.
- The exit hook cancels the jobs that are still running. `scancel` and `qdel`
  report their own failures without raising, so a finished job does not produce
  a `ProcessFailedException` for the user to read.
- `calibrate` returns the `EnsembleKalmanProcess` the run ended with when there
  is nothing left to run. It used to return the stored first iteration, which
  looks like a calibration that never updated.
- A calibration that the scheduler terminated stays terminated. `calibrate`
  records the iteration in `terminated.txt`, so a restart no longer runs the
  remaining iterations on the parameters the terminated update left behind.
- `set_worker_loggers(workers)` ignored `workers` and set the logger on every
  worker.
- `SlurmConfig` directives no longer have underscores rewritten in their
  *values*: `:partition => "gpu_debug"` used to emit `--partition=gpu-debug`.
  Single-letter directives now emit `-t 00:10:00` rather than the `-t=00:10:00`
  that `sbatch` rejects.
- PBS reported queued and held jobs as `RUNNING`, so `ispending` was unreachable;
  `qstat` states `H`, `W`, `T`, `M`, `S`, `B`, and `X` were not recognized at all.
- `show(::GEnsembleBuilder)` printed the table body to `stdout` instead of the
  given `IO`.
- `Checker.check(::SignChecker, ...)` required the `data` keyword argument that
  the interface documents as optional, so omitting it reported
  `"Not yet implemented!"` instead of naming the missing argument. Its two sign
  proportions were also computed with different denominators when the data
  contained `NaN`s.

### Observation pipeline

- `SeasonalDiagonalCovariance` rejects a degenerate result instead of handing it
  to EKP. A zero or `NaN` diagonal entry (a field that does not vary across
  samples, or one that is `NaN` in a sample built by hand) used to surface much
  later as a linear algebra failure with no indication of which observation
  caused it; the error now names the offending index and short name.
- `SVDplusDCovariance` requires at least two samples, as
  `SeasonalDiagonalCovariance` already did, and warns when its D term has an
  entry that is zero or `NaN` while the SVD term is rank deficient, naming the
  variable it belongs to. The SVD term has rank at most `n_samples - 1`, so D is
  what makes the covariance invertible, and D is
  `(model_error_scale * mean)^2 + regularization`: all zeros with the defaults,
  and zero wherever the sample mean is zero if only `model_error_scale` is set.
- `ObservationRecipe.observation` takes a `covariance` keyword argument, so
  building several observations from one `SampleCollection` no longer repeats the
  same covariance estimate (an SVD, for `SVDplusDCovariance`) for each of them.
- `ObservationRecipe.observation` checks a `covariance` passed to it against the
  sample it goes with. EKP checks neither the size nor the shape, so a covariance
  from another `SampleCollection` surfaced as a dimension mismatch inside EKP.
- `SampleBuilder.build_samples_by_times` warns when the time windows overlap,
  since the resulting samples share time slices and bias the covariance they are
  used to estimate.
- `SeasonalDiagonalCovariance` warns when a sample covers fewer than four
  seasons, which is allowed but is what a window shorter than a year produces.
- `QuantileRegularization` reported "Insufficient samples for computing
  quantile" when the constraint is on the number of *entries in the variable*,
  not the number of samples. Its zero check catches a quantile of exactly zero:
  `isapprox` to zero has no tolerance, and a tolerance here would be an absolute
  threshold on a squared quantity, which rejects the small values that a
  variable in SI units legitimately has.
- Documented that `SeasonalDiagonalCovariance` applies latitude weights *after*
  regularization, so the effective regularization varies with latitude, unlike
  `SVDplusDCovariance`. Both were documented as `regularization * I`.
- Corrected the claim that `SeasonalDiagonalCovariance` ignores `NaN`s when
  computing the variance: they are dropped when the samples are built, and a
  `NaN` whose position differs between samples is an error.

### Behavior changes

- `wait_for_jobs` now rethrows exceptions instead of swallowing them, so
  interrupting a calibration aborts it. It also takes a `job_timeout` (default
  24 hours), since a scheduler that stops responding would otherwise block a run
  forever. It queries each job once per poll instead of up to five times.
- `ispending`, `isrunning`, `issuccess`, `isfailed`, and `iscompleted` accept a
  `JobStatus` as well as a `JobInfo`.
- `log_member_error` no longer throws when the scheduler never wrote a log.

### Documentation

- Rewrote `README.md` as a landing page following the CliMA
  [DeveloperGuides](https://github.com/CliMA/DeveloperGuides) documentation
  policy: tagline, badge table, features, installation, a runnable quick
  example, documentation links, how the package fits into the CliMA ecosystem,
  and contribution guidelines.
- Added the `NOTICE` file that Apache-licensed CliMA repositories carry,
  mirroring the copyright line in `LICENSE`.
- Swept docstrings against the documentation policy: `# Fields` sections in
  place of field-level docstrings (which Documenter does not render), single-`#`
  section headings in place of `##` and underlined ones, imperative summaries,
  subtype lists on the abstract types, and `# Examples` on the API a user calls
  directly. `AbstractBackend` had no docstring at all.
- Aligned the sidebar labels in `docs/make.jl` with the page titles, and titled
  the task-oriented pages as verb phrases.
- Replaced the two `jldoctest` blocks with plain `julia` blocks, and the
  metaprogrammed HPC backend definitions with explicit ones, both per the
  policy.

- New "How a calibration works" page: the loop, where your `forward_model` and
  `observation_map` are called relative to the ensemble update, what the output
  directory contains, what is checkpointed on a restart, and a glossary of the
  EKP terms the rest of the docs assume.
- New "Troubleshooting" page: the errors a calibration raises, with causes and
  fixes, and what to check when a run converges to nothing
  ([#205](https://github.com/CliMA/ClimaCalibrate.jl/issues/205)).
- New "Job status" section in Backends, explaining what
  `PENDING`/`RUNNING`/`COMPLETED`/`FAILED` mean and how each is derived from
  `sacct` and `qstat`
  ([#297](https://github.com/CliMA/ClimaCalibrate.jl/issues/297)).
- Rewrote the tutorial. It runs a damped oscillator, so the docs build no
  longer pins `ClimaAtmos = "=0.28.3"` and `ClimaCore = "=0.14.51"` or runs 30
  ensemble members for 7 iterations of a climate model. It keeps its
  configuration in the model interface, and it puts the forward model in its own
  file, which is what a worker or a job script has to load. The parallel variant
  is shown as the three lines that change. It is not executed during the build,
  because Documenter's output capture deadlocks against `Distributed`'s message
  handling.
- Expanded the Observations page to cover what the noise covariance does and
  why observation ordering matters.
- Documented the SVD residual diagnostics, which the README and landing page
  advertise.
- Every submodule now has a docstring, and the `SampleBuilder`,
  `EnsembleBuilder`, and `ObservationRecipe` functions are documented in `src/`
  rather than only in the extension, so `?build_samples` works before
  ClimaAnalysis is loaded. `plot_g`, `plot_g_mean`, and `plot_obs` each had two
  copies of their docstring, which had already drifted apart.
- Corrected: the surface-fluxes example runs 6 iterations, not 8;
  `checkpoint.txt` is written before a member runs as well as after;
  `ignore_dims` exists only on `build_samples(::Matrix)`; `ScalarCovariance`
  takes neither `regularization` nor `model_error_scale`; the ClimaAnalysis
  floor is 0.5.23, not 0.5.19; and the EnsembleKalmanProcesses v2.4.2 metadata
  caveat is dead (the floor is 2.5). Fixed an undefined variable in the
  `GEnsembleBuilder` example and its missing imports.
- Removed a zero-byte `bibliography.bib` and the `DocumenterCitations` plumbing
  around it (the docs have no citations), and two assets orphaned
  when the CES extension was removed.

### Testing

- Slurm and PBS job-state parsing moved into pure functions
  (`_parse_slurm_state`, `_parse_pbs_state`) with unit tests in
  `test/job_status.jl`, so this behavior is covered without a cluster.
- Aqua now runs last, after the tests that load ClimaAnalysis, NaNStatistics,
  and CairoMakie, so its ambiguity, piracy, and unbound-argument checks cover
  `ext/`. `test_undocumented_names` and `test_persistent_tasks` are now run too.
- The end-to-end and initialization tests pass an `rng` to
  `EnsembleKalmanProcess` and check reproducibility and convergence directly.
- The surface-fluxes example calibrates `coefficient_a_m_businger` alone with
  unscented Kalman inversion, against a synthetic observation that carries a 2%
  error, from a prior centered on 3.5 while the observation was generated with
  4.7. One parameter takes a stencil of three members, so the calibration costs
  30 forward model runs over 10 iterations, which the docs build runs to
  generate the two figures on the quickstart page.
- The experiment runs on SurfaceFluxes v1.2, which replaced the thermodynamic
  state objects its forward model was built on with a call that takes the state
  variables. The docs environment pins the same version.
- The forward model adds a 2% error of its own, standing in for the internal
  variability that makes a climate model return a different statistic on every
  run, and the calibration is given it together with the observation error as
  its noise covariance. `loss_landscape_plot` sweeps the parameter and plots the
  loss that produces, which the quickstart page shows alongside what an ensemble
  Kalman method does with it. It also calibrated
  `coefficient_a_h_businger`, which moves the profile-averaged `ustar` in almost
  the same direction, so one observable left the pair unidentified and the
  ensemble free to slide along a ridge. The observation carried no error at all:
  the argument that adds it defaults to `false` and was never passed, and what
  the calibration was given as its noise covariance was the variance of `ustar`
  across profiles that are all the same case.
- `SurfaceFluxModelInterface` holds the output directory and the ensemble size.
  The forward model read a hardcoded path, so the script in the quickstart
  failed for any other `output_dir`.
- `test/visualization.jl` now checks that each recipe plots the data it was
  asked for.
- The docs build now loads both trigger packages for the ClimaAnalysis
  extension, lists both extensions and each submodule in `modules` so that
  `checkdocs = :exports` reaches the APIs that `names(ClimaCalibrate)` does not,
  and runs `linkcheck` on CI as a warning.
- CI tests Julia 1.10 (the declared minimum), the current release, and nightly.
  Buildkite steps that submit scheduler jobs retry on failure instead of being
  marked soft-failing, so a `calibrate`, `submit_job`, or `job_status` failure
  fails the build.

### Other

- Fix a bug where `SampleBuilder.build_samples` did not work with `OutputVar`s
  with no dimensions. `SampleBuilder`, `ObservationRecipe`, and
  `EnsembleBuilder` now support `OutputVar`s with no dimensions. This requires
  ClimaAnalysis v0.5.23 or newer.

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

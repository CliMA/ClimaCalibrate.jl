```@meta
CurrentModule = ClimaCalibrate
```

# Troubleshooting

## Where to look first

When something goes wrong, the output directory has the answer more often than
the driver's log does. For a failing ensemble member:

```julia
# What the member's forward model printed (HPC backends write this)
read(ClimaCalibrate.path_to_model_log(output_dir, iteration, member), String)

# The parameters it was run with
read(ClimaCalibrate.parameter_path(output_dir, iteration, member), String)

# Whether it finished
ClimaCalibrate.model_completed(output_dir, iteration, member)
```

For the calibration as a whole:

```julia
ekp = ClimaCalibrate.load_latest_ekp(output_dir)
EKP.get_error(ekp)                     # covariance-weighted error per iteration
EKP.get_ϕ_mean_final(prior, ekp)       # current mean parameters
ClimaCalibrate.last_completed_iteration(output_dir)
```

## Errors you may hit

**`Execution halted: iteration N had a X% failure rate`**

More than `failure_rate` of the ensemble failed. Read the model log of one of
the members named in the preceding warning. If the failures are expected,
because some parameter draws fall outside the model's domain of validity, raise
the threshold with `JuliaBackend(; failure_rate = 0.8)` or the equivalent on your
backend, or tighten the prior so those draws are not proposed.

**`No workers available for Ns ... Ensure workers were submitted`**

A `WorkerBackend` iteration waited on an empty pool with nothing running and
nothing initializing. Either [`add_workers`](@ref) was never called, or the
workers failed to start. Check the worker logs (`worker_*.log` in the launch
directory) and `Distributed.workers()`.

**`Iteration N did not finish within Ns`**

Scheduler jobs never reached a terminal state. Check whether they are still
queued (`squeue -u $USER`, `qstat -u $USER`) and whether the scheduler is
reachable. PBS reports a job it cannot see as still running, so an unreachable
`qstat` looks like a job that never finishes.

**`OutputVar with the short name X did not match with any of the metadata`**

`GEnsembleBuilder` could not find a place for that variable in the G ensemble
matrix. Either the short name differs from the one in the observation, or one of
the checks failed. Pass `verbose = true` to
`EnsembleBuilder.fill_g_ens_col!` to have each failed check say why, and call
`EnsembleBuilder.missing_short_names(builder, col_idx)` to see what is still
unfilled.

**A `MethodError` on `build_samples`, `covariance`, `GEnsembleBuilder`, or
`plot_g`**

These live in package extensions. Load their trigger packages first:
`import ClimaAnalysis, NaNStatistics` for the observation tooling, and a Makie
backend such as `import CairoMakie` for the plots.

**The covariance has non-positive or NaN diagonal entries**

An observation entry that does not vary across the samples has zero variance,
and one that is `NaN` in every sample gives a `NaN`. Both make the covariance
singular. The error names the offending index and short name; add
`regularization` or `model_error_scale` to the covariance estimator, or fix the
preprocessing.

## The calibration runs but does not converge

**Check that the observation is reachable at all.** Plot the ensemble's forward
map evaluations against the observation:

```julia
import CairoMakie
fig, ax, _ = ClimaCalibrate.Visualization.plot_g(ekp; color = (:grey, 0.3))
ClimaCalibrate.Visualization.plot_obs!(ax, ekp; color = :black)
```

If the observation lies outside the envelope of the ensemble and the residual
has a consistent sign, no parameter combination in the prior can reach it. That
is a model or observation problem, not a calibration one: check units, check for
a missing regridding step, and check that the prior covers plausible values.

**Check the ordering.** The most common silent failure is that the entries of
the G ensemble matrix are not in the same order as the observation. The
calibration then runs without error and converges to nothing meaningful. Using
`ObservationRecipe` together with `GEnsembleBuilder` prevents this, because the
builder places each variable using the observation's own metadata. If you build
the matrix by hand, verify the ordering against
`ObservationRecipe.short_names(obs)`.

**Check the noise covariance.** Too large and the ensemble barely moves; too
small and it collapses in an iteration or two onto whatever the first ensemble
happened to favor. Compare the covariance-weighted error against the number of
observation entries: a well-calibrated run ends with an error of roughly the
same order.

**Check that the ensemble is large enough.** EKP recommends an ensemble size
that grows with the number of parameters, and warns when yours is below it.

**Inspect the residual structure.** For a `SVDplusD` or `Diagonal` covariance,
[`analyze_residual`](@ref) projects the residual onto the leading eigenvectors of
the noise covariance and reports how much of it is structured rather than noise,
broken down by variable.

## Restarting

Re-run the same `calibrate` call with the same `output_dir`. Completed
iterations and completed ensemble members are skipped. See
[Restarts](@ref) for what is checkpointed.

To start over instead, use a fresh output directory. Deleting individual files
from an existing one leaves the two checkpoints inconsistent.

```@meta
CurrentModule = ClimaCalibrate
```

# How a calibration works

ClimaCalibrate runs one loop. What happens in one pass through it, and where
your code is called, is most of what you need to use the package.

## The loop

[`calibrate`](@ref) repeats these steps until it runs out of iterations or
EnsembleKalmanProcesses decides the calibration has converged:

1. **Write the parameters.** EKP proposes an ensemble of `m` parameter vectors.
   ClimaCalibrate writes each one to `iteration_XXX/member_YYY/parameters.toml`
   and saves the EKP object alongside them.
2. **Run the forward model, once per member.** Your `forward_model(interface,
   iteration, member)` reads that member's parameters, runs the model, and
   writes its output wherever you like, usually under the member's own
   directory. This is the only step the backend affects: it decides whether the
   members run one at a time, on Distributed.jl workers, or as separate
   scheduler jobs.
3. **Evaluate the observation map.** Your `observation_map(interface,
   iteration)` reads every member's output and returns the **G ensemble
   matrix**: one column per member, each column comparable entry-for-entry with
   the observation.
4. **Postprocess, optionally.** If you implemented
   [`postprocess_g_ensemble`](@ref), it can transform the matrix using
   information the observation map did not have (the EKP object, the prior, the
   output directory).
5. **Update the ensemble.** EKP compares the G ensemble matrix against the
   observation, weighted by the noise covariance, and proposes the next
   ensemble.
6. **Analyze, optionally.** [`analyze_iteration`](@ref) runs last. The default
   implementation logs the mean parameters and the covariance-weighted error.

```
                  ┌─────────────────────────────────────────┐
                  │                                         │
     parameters.toml                                        │
          │                                                 │
          ▼                                                 │
    forward_model  ×m   ──►  observation_map  ──►  update_ensemble!
     (the backend             (your code)          (EKP, using the
      runs these)                                   noise covariance)
                                                            │
                                                            │
                  └─────────────────────────────────────────┘
```

Steps 2 and 3 are yours; the rest is bookkeeping. Steps 4 and 6 are optional
hooks with defaults.

## What you have to provide

- **A forward model**, as [`forward_model`](@ref).
- **An observation map**, as [`observation_map`](@ref).
- **Observations and a noise covariance**, wrapped in an `EKP.Observation` or
  `EKP.ObservationSeries`. See [Observations](@ref).
- **A prior** for every parameter being calibrated.
- **An `EnsembleKalmanProcess`**, which ties the observations, the prior, and
  the choice of algorithm together.

Both of your functions dispatch on a subtype of
[`AbstractModelInterface`](@ref) that you define. Put the configuration they
need, such as the output directory, ensemble size, and file paths, in its
fields. That object is what gets sent to a worker or serialized into a job
script, so anything not in it will not be there when the forward model runs.

## Where the output goes

The run lives under one output directory, which is also what a restart reads:

```
output_dir/
├── interface.jld2              # the model interface (HPC backends only)
├── iteration_001/
│   ├── eki_file.jld2           # EKP state used to draw this iteration
│   ├── prior.jld2              # the prior, saved once
│   ├── G_ensemble.jld2         # written after the observation map runs
│   └── member_001/
│       ├── parameters.toml     # this member's parameters
│       ├── checkpoint.txt      # "started" or "completed"
│       └── model_log.txt       # the member's stdout (HPC backends only)
└── iteration_002/
    └── ...
```

Use [`path_to_iteration`](@ref), [`path_to_ensemble_member`](@ref),
[`parameter_path`](@ref), and [`ekp_path`](@ref) rather than building these
paths by hand.

## Restarts

Re-running the same `calibrate` call against the same `output_dir` resumes it.
Three things are checkpointed:

- **Iterations.** [`last_completed_iteration`](@ref) reports the last iteration
  that both produced a `G_ensemble.jld2` and got as far as writing the *next*
  iteration's `eki_file.jld2`. The loop starts from the one after that.
- **Ensemble members.** Each member writes `checkpoint.txt` before it runs and
  again when it finishes. On a restart, members marked `completed` are skipped
  and the rest are rerun.
- **Termination.** When the `EnsembleKalmanProcess` scheduler stops the
  calibration, the iteration it stopped at is written to `terminated.txt` and
  [`terminated_iteration`](@ref) reads it back. A restart of a terminated
  calibration returns the stored process and runs nothing: the update that
  terminated left the ensemble as it was, so the remaining iterations would
  repeat the parameters of the iteration before it.

The forward model itself is not restarted mid-run; if your model supports
checkpointing internally, that is up to it.

A restart uses the `EnsembleKalmanProcess` stored in `iteration_001`, not the
one passed to `calibrate`. A freshly constructed object draws a new random
initial ensemble, which would then be paired with forward model output produced
from the old one. The two have to describe the same calibration: `calibrate`
compares the ensemble size, the observations, the shape of their covariances,
and the process, and errors if they disagree. The scheduler, accelerator, and
localizer come from the stored object.

A calibration whose iterations are all complete also runs nothing, and returns
the process it ended with.

## Vocabulary

Terms that come from EnsembleKalmanProcesses and appear throughout these docs:

- **Ensemble member**: one parameter vector and the forward model run that used
  it. An iteration has `m` of them.
- **G ensemble matrix**: the observation map's output for a whole iteration. `G`
  is the composition of the forward model and the observation map, so column `m`
  is `G(θ_m)`.
- **Forward map evaluation**: one column of that matrix.
- **Constrained and unconstrained parameters**: EKP works in an unconstrained
  space and transforms to the constrained (physical) space that your model sees.
  `parameters.toml` holds constrained values; `EKP.get_ϕ` returns constrained
  ones and `EKP.get_u` unconstrained ones.
- **Noise covariance**: how much of the model–observation mismatch to attribute
  to error rather than to the parameters. It sets the weighting of the
  observation entries, and how far the ensemble is willing to move.
- **Minibatch**: a subset of observations used for one iteration, when there are
  more observations than you want to score against at once. A **minibatcher**
  decides the subsets, and an `ObservationSeries` holds them.

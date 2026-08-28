# ClimaCalibrate.jl

ClimaCalibrate runs the calibration loop around your forward model: it launches
an ensemble of models in parallel, collects their output, hands it to
[EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl/)
to produce the next set of parameters, and picks up where it left off if the run
is interrupted. EKP chooses the parameters, while ClimaCalibrate runs your model
with them and returns the results.

The same calibration code runs unchanged on a laptop, across Julia worker
processes, or as one scheduler job per ensemble member on an HPC cluster. You
choose by swapping the backend.

## Installation

```julia
julia> ] add ClimaCalibrate
```

Julia 1.10 or newer is required.

## What it provides

- A backend system covering single-process runs, Distributed.jl workers, and
  the Slurm and PBS job schedulers on several HPC systems
- Checkpointing, so an interrupted calibration resumes without rerunning
  completed forward models
- Recipes for turning [ClimaAnalysis.jl](https://github.com/CliMA/ClimaAnalysis.jl)
  `OutputVar`s into observations with estimated noise covariances
- A builder that assembles the ensemble output matrix from `OutputVar`s, so you
  do not have to track index ranges by hand
- Diagnostics for `SVDplusD` covariance matrices and Makie plots of ensemble
  output against observations

## Where to start

- [Getting Started](quickstart.md) walks through the pieces a calibration needs and how
  to put them together.
- The [Distributed Calibration Tutorial](literate_example.md) is a complete example you can run
  locally.
- [Backends](backends.md) explains how to move a working calibration onto more
  hardware.
- The [how-to guide](howdoi.md) collects answers to common questions.

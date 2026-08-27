<!-- Title -->
<h1 align="center">
  ClimaCalibrate.jl
</h1>

<!-- description -->
<p align="center">
  <strong>A toolkit for building scalable calibration pipelines with minimal boilerplate.</strong>
</p>

[![dev][docs-dev-img]][docs-dev-url]
[![ghaci][gha-ci-img]][gha-ci-url]
[![codecov][codecov-img]][codecov-url]
[![license][license-img]][license-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/ClimaCalibrate.jl/dev/

[gha-ci-img]: https://github.com/CliMA/ClimaCalibrate.jl/actions/workflows/ci.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/ClimaCalibrate.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/CliMA/ClimaCalibrate.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/ClimaCalibrate.jl

[license-img]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[license-url]: https://github.com/CliMA/ClimaCalibrate.jl/blob/main/LICENSE

ClimaCalibrate takes a forward model, a set of observations, and a prior
distribution over the model's parameters, and finds parameter values that make
the model match the observations. The parameter search itself is done by
[EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl);
ClimaCalibrate handles everything around it: running an ensemble of models in
parallel, collecting their output, feeding it back to the solver, and picking up
where it left off if the run is interrupted.

The same calibration code runs unchanged on a laptop, across Julia worker
processes, or as one scheduler job per ensemble member on an HPC cluster. You
choose by swapping the backend.

## Installation

```julia
julia> ] add ClimaCalibrate
```

Julia 1.10 or newer is required.

## Where to run

- `JuliaBackend` runs ensemble members one at a time in the current process.
  Good for small models and for debugging.
- `WorkerBackend` distributes members over
  [Distributed.jl](https://github.com/JuliaLang/Distributed.jl) workers, which
  can be started locally or requested from Slurm or PBS.
- The HPC backends submit one scheduler job per ensemble member. Ready-made
  configurations exist for the
  [Resnick High Performance Computing Center](https://www.hpc.caltech.edu/),
  [NSF NCAR Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/),
  Google Cloud, and CliMA's private GPU server.

## What else is in the box

- Restart handling: completed forward models and iterations are checkpointed and
  skipped on restart.
- Recipes for turning [ClimaAnalysis.jl](https://github.com/CliMA/ClimaAnalysis.jl)
  `OutputVar`s into observations with estimated noise covariances.
- A builder that assembles the ensemble output matrix from `OutputVar`s, so you
  do not have to track index ranges by hand.
- Diagnostics for `SVDplusD` covariance matrices and Makie plots of ensemble
  output against observations.

## Documentation

The [documentation](https://CliMA.github.io/ClimaCalibrate.jl/dev/) covers a
[getting started guide](https://clima.github.io/ClimaCalibrate.jl/dev/quickstart/),
a [worked distributed example](https://clima.github.io/ClimaCalibrate.jl/dev/literate_example/),
and the full API.

## Contributing

Contributions of any size are welcome, and fresh eyes catch errors that regular
developers miss. If you would like to work on a new feature, let us know by
[opening an issue](https://github.com/CliMA/ClimaCalibrate.jl/issues/new).

## License

Apache License 2.0. See [LICENSE](LICENSE).

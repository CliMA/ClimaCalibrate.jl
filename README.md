# ClimaCalibrate.jl

Calibrate model parameters against observations, from a laptop to a supercomputer with the same code.

ClimaCalibrate.jl runs the calibration loop around your forward model: it launches an ensemble of model runs, collects their output, hands it to [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl) to produce the next set of parameters, and repeats. EnsembleKalmanProcesses chooses the parameters; ClimaCalibrate runs your model with them, checkpoints the results, and resumes where it left off if the run is interrupted.

|||
|------------------:|:------------------------------------------------------------|
| **Documentation** | [![stable][docs-stable-img]][docs-stable-url] [![dev][docs-dev-img]][docs-dev-url] |
| **Version**       | [![version][version-img]][version-url]                      |
| **License**       | [![license][license-img]][license-url]                      |
| **Tests**         | [![gha ci][gha-ci-img]][gha-ci-url]                         |
| **Code Coverage** | [![codecov][codecov-img]][codecov-url]                      |
| **Downloads**     | [![Downloads][dlt-img]][dlt-url]                            |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://CliMA.github.io/ClimaCalibrate.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/ClimaCalibrate.jl/dev/

[version-img]: https://juliahub.com/docs/General/ClimaCalibrate/stable/version.svg
[version-url]: https://juliahub.com/ui/Packages/General/ClimaCalibrate

[license-img]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[license-url]: https://github.com/CliMA/ClimaCalibrate.jl/blob/main/LICENSE

[gha-ci-img]: https://github.com/CliMA/ClimaCalibrate.jl/actions/workflows/ci.yml/badge.svg?branch=main
[gha-ci-url]: https://github.com/CliMA/ClimaCalibrate.jl/actions/workflows/ci.yml?query=branch%3Amain

[codecov-img]: https://codecov.io/gh/CliMA/ClimaCalibrate.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/ClimaCalibrate.jl

[dlt-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FClimaCalibrate&query=total_requests&label=Downloads
[dlt-url]: https://juliapkgstats.com/pkg/ClimaCalibrate

## Features

- **One calibration, several places to run it**: the same code runs members one at a time in the current process, across Distributed.jl workers, or as one scheduler job per ensemble member. You choose by swapping the backend.
- **Ready-made HPC backends**: Slurm and PBS support, with configurations for the [Resnick High Performance Computing Center](https://www.hpc.caltech.edu/), [NSF NCAR Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/), Google Cloud, and CliMA's GPU server.
- **Restarts**: completed iterations and completed forward models are checkpointed and skipped, so an interrupted calibration picks up where it stopped.
- **Observations from model output**: recipes that turn [ClimaAnalysis.jl](https://github.com/CliMA/ClimaAnalysis.jl) `OutputVar`s into observations with estimated noise covariances.
- **A G ensemble builder that checks itself**: it places each variable using the observation's own metadata and validates short names, units, dimension names, dimension units, and dimension values, so model output that does not line up with the observation raises an error instead of being calibrated against silently.
- **Diagnostics**: residual analysis that reports how much of the unfitted residual is structured rather than noise-like, and Makie plots of ensemble output against observations.

## Installation

ClimaCalibrate.jl is a registered Julia package. Install it with:

```julia
julia> ] add ClimaCalibrate
```

Julia 1.10 or newer is required.

## Quick Example

A calibration needs three things from you: a struct holding your configuration, a
forward model, and an observation map. Here the "model" is `exp(-rate)`, and the
calibration recovers `rate` from a single observation.

```julia
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions:
    combine_distributions, constrained_gaussian
import TOML

struct Decay <: CAL.AbstractModelInterface
    output_dir::String
    ensemble_size::Int
end

# Run one ensemble member: read the parameters written for it, write its output
function CAL.forward_model(model::Decay, iteration, member)
    (; output_dir) = model
    member_dir = CAL.path_to_ensemble_member(output_dir, iteration, member)
    rate = TOML.parsefile(CAL.parameter_path(output_dir, iteration, member))
    write(joinpath(member_dir, "out"), string(exp(-rate["rate"]["value"])))
end

# Collect the whole ensemble's output, one column per member
function CAL.observation_map(model::Decay, iteration)
    (; output_dir, ensemble_size) = model
    G_ensemble = Matrix{Float64}(undef, 1, ensemble_size)
    for member in 1:ensemble_size
        member_dir = CAL.path_to_ensemble_member(output_dir, iteration, member)
        G_ensemble[1, member] =
            parse(Float64, read(joinpath(member_dir, "out"), String))
    end
    return G_ensemble
end

prior = combine_distributions([constrained_gaussian("rate", 1.0, 0.5, 0, Inf)])
observation = [exp(-0.7)]                # generated with rate = 0.7
noise = reshape([1e-4], 1, 1)
ensemble_size, n_iterations = 10, 5
output_dir = mktempdir()

ekp = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, ensemble_size),
    observation,
    noise,
    EKP.Inversion(),
)

ekp = CAL.calibrate(
    CAL.JuliaBackend(),
    ekp,
    Decay(output_dir, ensemble_size),
    n_iterations,
    prior,
    output_dir,
)

EKP.get_ϕ_mean_final(prior, ekp)          # ≈ [0.7]
```

To run the ensemble in parallel instead, swap `CAL.JuliaBackend()` for
`CAL.WorkerBackend()` or one of the HPC backends. Nothing else changes.

## Documentation

The documentation is at [stable][docs-stable-url] and [dev][docs-dev-url]. Useful entry points:

- [How a calibration works](https://CliMA.github.io/ClimaCalibrate.jl/stable/concepts/): the loop, where your code is called, what lands in the output directory, and how restarts work.
- [Getting Started](https://CliMA.github.io/ClimaCalibrate.jl/stable/quickstart/): the pieces a calibration needs and how to put them together.
- [Calibration Tutorial](https://CliMA.github.io/ClimaCalibrate.jl/stable/literate_example/): a complete worked example you can run locally.
- [Backends](https://CliMA.github.io/ClimaCalibrate.jl/stable/backends/): choosing where the ensemble runs, and moving a working calibration onto a cluster.
- [Observations](https://CliMA.github.io/ClimaCalibrate.jl/stable/observations/): building observations and noise covariances from data.
- [Troubleshooting](https://CliMA.github.io/ClimaCalibrate.jl/stable/troubleshooting/): what the errors mean, and what to check when a calibration does not converge.

## Integration with CliMA models

ClimaCalibrate.jl is the calibration layer of the [CliMA Earth System Model](https://clima.caltech.edu):

- It builds on [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl), which provides the ensemble Kalman algorithms, priors, and observation containers.
- Its observation and G ensemble tooling is a package extension over [ClimaAnalysis.jl](https://github.com/CliMA/ClimaAnalysis.jl), loaded when ClimaAnalysis is available.
- Parameters are exchanged through TOML files in the format [ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl) reads, so a calibrated parameter set drops straight into a model run.
- It is used to calibrate [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl), [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl), [ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl), and [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl) (via [ClimaOceanCalibration.jl](https://github.com/CliMA/ClimaOceanCalibration.jl)). Worked setups live in [calibration-experiments](https://github.com/CliMA/calibration-experiments).

## Contributing

Contributions of any size are welcome, and fresh eyes catch errors that regular developers miss. If you would like to work on a new feature, let us know by [opening an issue](https://github.com/CliMA/ClimaCalibrate.jl/issues/new).

Development follows the shared CliMA [DeveloperGuides](https://github.com/CliMA/DeveloperGuides), which cover code style, the documentation policy, testing, and the changelog conventions this repository uses. Before opening a pull request, run the test suite and the formatter:

```julia
julia --project -e 'using Pkg; Pkg.test()'
julia -e 'using JuliaFormatter; format(".")'
```

User-visible changes go in [NEWS.md](NEWS.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

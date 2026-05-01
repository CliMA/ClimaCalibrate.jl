# ClimaCalibrate.jl

ClimaCalibrate provides a scalable framework for calibrating forward models
using the Ensemble Kalman Process (EKP). It integrates with
[EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl/)
to enable distributed model calibration with minimal boilerplate code.

Key Features

- Distributed computing support for multiple HPC environments via a backend
  system
- Support for Distributed.jl with Slurm and PBS job schedulers
- Integration with EnsembleKalmanProcesses.jl for parameter estimation
- Flexible model interface for different component models
- Reusable recipes for creating observations from ClimaAnalysis.jl `OutputVar`s
- G ensemble builder for simplifying the construction of the G ensemble matrix
- `SVDplusD` covariance matrix analysis

For more information, see our
[Getting Started page](https://clima.github.io/ClimaCalibrate.jl/dev/quickstart/).

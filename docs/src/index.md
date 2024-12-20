# ClimaCalibrate.jl

ClimaCalibrate.jl is a toolkit for developing scalable and reproducible model 
calibration pipelines using [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl/) with minimal boilerplate.

This documentation assumes basical familiarity with inverse problems and [Ensemble Kalman Inversion](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/ensemble_kalman_inversion/#eki) in particular.

To use this framework, component models define their own versions of the functions provided in the interface.
Calibrations can either be run using just Julia, the Caltech central cluster, NCAR Derecho, or CliMA's GPU server.

For more information, see our [Getting Started page](https://clima.github.io/ClimaCalibrate.jl/dev/quickstart/).

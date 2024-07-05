# ClimaCalibrate.jl

ClimaCalibrate.jl is a toolkit for developing scalable and reproducible model 
calibration pipelines using CalibrateEmulateSample.jl with minimal boilerplate.

To use this framework, component models (and the coupler) define their own versions of the functions provided in the interface (`get_config`, `get_forward_model`, and `run_forward_model`).

Calibrations can either be run using pure Julia, the Caltech central cluster, or CliMA's GPU server.

For more information, see our Getting Started page.

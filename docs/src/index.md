# ClimaCalibrate.jl

ClimaCalibrate.jl is a toolkit for developing scalable and reproducible model 
calibration pipelines using CalibrateEmulateSample.jl with minimal boilerplate.

To use this framework, component models (and the coupler) define their own versions of the functions provided in the interface.
Calibrations can either be run using just Julia, the Caltech central cluster, NCAR Derecho, or CliMA's GPU server.

For more information, see our Getting Started page.

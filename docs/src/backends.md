## Backends

ClimaCalibrate can scale calibrations on different distributed computing environments, referred to as backends. Most of these are high-performance computing clusters.

Each backend has an associated `calibrate(::AbstractBackend, ...)`  dispatch, which initializes and runs the calibration on the given backend.

The following backends are currently supported:

- [`JuliaBackend`](@ref)
- [`WorkerBackend`](@ref)
- [`CaltechHPCBackend`](@ref)
- [`ClimaGPUBackend`](@ref)
- [`DerechoBackend`](@ref)


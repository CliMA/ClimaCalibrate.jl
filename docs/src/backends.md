# Backends

ClimaCalibrate can scale calibrations on different distributed computing environments, referred to as backends. Each backend is optimized for specific use cases and computing resources. The backend system is implemented through Julia's multiple dispatch, allowing seamless switching between different computing environments.

## Available Backends

1. [`JuliaBackend`](@ref): The simplest backend that runs everything serially on a single machine. Best for initial testing and small calibrations that do not require parallelization. 

2. [`WorkerBackend`](@ref): Uses Julia's built-in distributed computing capabilities, assigning forward model runs to separate workers using Distributed.jl. Workers can be created using [`SlurmManager`](@ref), [`Distributed.addprocs`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.addprocs), or by initializing julia with the `-p` option: `julia -p 2`. Available workers can be accessed using [`Distributed.workers()`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.workers).

3. HPC Cluster Backends: These backends schedule forward model runs on HPC clusters using Slurm or PBS.
    - [`CaltechHPCBackend`](@ref): Caltech's Resnick HPC cluster
    - [`ClimaGPUBackend`](@ref): CliMA's private GPU server
    - [`DerechoBackend`](@ref): NSF NCAR Derecho supercomputing system.

## Choosing the Right Backend

The right backend is largely determined by the computational cost of your forward model.

If your model is very simple or you are debugging, use the `JuliaBackend`.

If your model requires just one CPU core or GPU, the best backend is the `WorkerBackend`. 

If your forward model requires parallelization across multiple cores or GPUs, choose one of the HPC Cluster backends. These allow you allocate more resources to each forward model using Slurm or PBS.

## Using a Backend

Backends are the first argument to the [`calibrate`](@ref) function, which runs iterations of the forward model, updating model parameter based on observations.

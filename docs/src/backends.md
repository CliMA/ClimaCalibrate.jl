# Backends

ClimaCalibrate can scale calibrations on different distributed computing
environments, referred to as backends. Each backend is optimized for specific
use cases and computing resources. The backend system is implemented through
Julia's multiple dispatch, allowing seamless switching between different
computing environments.

## Available backends

1. [`JuliaBackend`](@ref): The simplest backend that runs everything serially on
   a single machine. Best for initial testing and small calibrations that do not
   require parallelization.

2. [`WorkerBackend`](@ref): Uses Julia's built-in distributed computing
   capabilities, assigning forward model runs to separate workers using
   Distributed.jl. Workers can be created using [`SlurmManager`](@ref),
   [`Distributed.addprocs`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.addprocs),
   or by initializing julia with the `-p` option: `julia -p 2`. Available
   workers can be accessed using
   [`Distributed.workers()`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.workers).

3. HPC Cluster Backends: These backends schedule forward model runs on HPC clusters using Slurm or PBS.
    - [`CaltechHPCBackend`](@ref): Caltech's Resnick HPC cluster,
    - [`ClimaGPUBackend`](@ref): CliMA's private GPU server,
    - [`DerechoBackend`](@ref): NSF NCAR Derecho supercomputing system,
    - [`GCPBackend`](@ref): CliMA's Google cloud platform.

!!! note "What are the differences between the `WorkerBackend` and the `HPCBackend`?"
    The `WorkerBackend` requests for all the resources up front to do all the
    work (typically executing a forward model) for a single iteration while the
    `HPCBackend` submits a job for each ensemble member for a single iteration.
    On a busy cluster, the `WorkerBackend` can stall an entire iteration waiting
    for enough free resources to launch all workers at once. The `HPCBackend`
    avoids this by submitting each ensemble member as an independent job, so
    progress for a calibration is made as soon as any ensemble member completes.
    Because each `HPCBackend` job starts a fresh Julia process, precompilation
    occurs on every job submission. Workers in the `WorkerBackend` are
    long-lived, so precompiled code is reused across all iterations.

## Choosing the right backend for calibration

The right backend is largely determined by the computational cost of your
forward model.

If your model is very simple or you are debugging, use the `JuliaBackend`.

If your model requires just one CPU core or GPU, the best backend is the
`WorkerBackend`.

If your forward model requires parallelization across multiple cores or GPUs,
choose one of the HPC cluster backends. These allow you to allocate more
resources to each forward model using Slurm or PBS.

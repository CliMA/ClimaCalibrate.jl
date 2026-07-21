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
   On a cluster, [`add_workers`](@ref) submits each worker as an individual
   scheduler allocation and adds it to a pool as it connects, so a calibration
   can start before every worker is up and picks up workers as they join.

3. HPC Cluster Backends: These backends schedule forward model runs on HPC clusters using Slurm or PBS.
    - [`CaltechHPCBackend`](@ref): Caltech's Resnick HPC cluster,
    - [`ClimaGPUBackend`](@ref): CliMA's private GPU server,
    - [`DerechoBackend`](@ref): NSF NCAR Derecho supercomputing system,
    - [`GCPBackend`](@ref): CliMA's Google cloud platform.

!!! note "What are the differences between the `WorkerBackend` and the HPC cluster backends?"
    Both distribute ensemble members across a cluster, but in different ways. The
    `WorkerBackend` keeps a pool of long-lived Julia workers and dispatches
    forward-model runs to them, so each worker precompiles once and reuses that
    code across all iterations. The HPC cluster backends instead submit a
    separate scheduler job for every ensemble member on every iteration; each job
    starts a fresh Julia process and precompiles again, but the jobs are
    independent, so an iteration makes progress as soon as any member is
    scheduled. Prefer the `WorkerBackend` when precompilation dominates runtime
    and the workers fit in your allocation; prefer an HPC cluster backend when
    forward models are long-running, need internal parallelism, or the cluster
    cannot hold all workers at once.

!!! note "Loading code on workers"
    For a `WorkerBackend`, load your model code with [`@worker_setup`](@ref)
    rather than `Distributed.@everywhere`. Because workers can join
    asynchronously, `@everywhere` would miss any worker that connects after it
    runs; `@worker_setup` records the setup and replays it on each worker as it
    joins, so late-joining workers are initialized correctly.

## Choosing the right backend for calibration

The right backend is largely determined by the computational cost of your
forward model.

If your model is very simple or you are debugging, use the `JuliaBackend`.

If your model requires just one CPU core or GPU, the best backend is the
`WorkerBackend`.

If your forward model requires parallelization across multiple cores or GPUs,
choose one of the HPC cluster backends. These allow you to allocate more
resources to each forward model using Slurm or PBS.

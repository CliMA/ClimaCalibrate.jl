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
   can start before every worker is up and picks up workers as they join. On
   clusters that charge for whole nodes (e.g. Derecho), pass `workers_per_node` to
   [`add_workers`](@ref) to put several workers, one per GPU, in each
   allocation.

3. HPC Cluster Backends: These backends schedule forward model runs on HPC clusters using Slurm or PBS.
    - [`CaltechHPCBackend`](@ref): Caltech's Resnick HPC cluster,
    - [`ClimaGPUBackend`](@ref): CliMA's private GPU server,
    - [`DerechoBackend`](@ref): NSF NCAR Derecho supercomputing system,
    - [`GCPBackend`](@ref): CliMA's Google cloud platform.

!!! note "What are the differences between the `WorkerBackend` and the HPC cluster backends?"
    The main difference between the two backends is how the work of the ensemble
    members is distributed. The `WorkerBackend` keeps a pool of long-lived Julia
    workers and dispatches forward-model runs to them, so each worker precompiles
    once and reuses that code across all iterations. The HPC cluster backends
    instead submit a
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

## Job status

An `HPCBackend` reports an ensemble member's job as one of four
[`JobStatus`](@ref) values:

| Status | Meaning |
|---|---|
| `PENDING` | Queued, waiting to be scheduled |
| `RUNNING` | Executing |
| `COMPLETED` | Finished successfully |
| `FAILED` | Finished with an error, or was cancelled, timed out, or ran out of memory |

Query one with [`job_status`](@ref), or with the predicates
[`ispending`](@ref), [`isrunning`](@ref), [`issuccess`](@ref),
[`isfailed`](@ref), and [`iscompleted`](@ref), which accept either a
[`JobInfo`](@ref) or a `JobStatus`. Each `JobInfo` query shells out to the
scheduler, so ask for the status once and test the result rather than calling
several predicates on the job.

### How it is determined

On **Slurm**, `squeue` only lists jobs that are still queued or running, so it
answers `PENDING` and `RUNNING`. Once a job leaves the queue, its outcome comes
from `sacct`: `COMPLETED` maps to success, and `FAILED`, `CANCELLED`, `TIMEOUT`,
`OUT_OF_MEMORY`, `NODE_FAIL`, `BOOT_FAIL`, `DEADLINE`, `PREEMPTED`, and
`REVOKED` all map to `FAILED`. A state the package does not recognize is treated
as a failure, not a success.

!!! note "Slurm accounting must be enabled"
    Distinguishing a successful job from a failed one requires `sacct`. If your
    cluster has no accounting database, ClimaCalibrate falls back to the member's
    checkpoint file, which is written by the forward model itself.

On **PBS**, `qstat` reports a `job_state` and a `substate`. `Q`, `H`, `W`, `T`,
and `M` are `PENDING`; `R`, `S`, `B`, and `E` are `RUNNING`; `F` and `X` are
finished, and are `FAILED` rather than `COMPLETED` when the substate is 93,
which is how PBS records a job that exited with an error.

If the scheduler cannot be reached, the job is reported as still running and a
warning is logged. That is right for a transient outage. A permanent one would
block forever, so a member that has been running for longer than the backend's
[`job_timeout`](@ref ClimaCalibrate.Backend.job_timeout) ends the iteration, and
the remaining jobs are cancelled. The default is 24 hours, and a model that runs
longer than that needs a larger one:

```julia
backend = ClimaCalibrate.CaltechHPCBackend(;
    directives = [:time => 60 * 48, :ntasks => 1],
    job_timeout = 60 * 60 * 48,
)
```

The clock starts when a job leaves the queue, so a long wait for an allocation
does not count against it.

### The checkpoint is the final word

Whatever the scheduler says, a member that did not write a `completed`
checkpoint is counted as failed. A batch script can exit successfully while the
model inside it did not, and a scheduler can lose the record of a job, so the
checkpoint the forward model writes itself is the more reliable signal.

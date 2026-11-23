# Backends

ClimaCalibrate runs calibrations on different distributed computing
environments, called backends. Each suits a different set of computing
resources.

## Available backends

1. [`JuliaBackend`](@ref): The simplest backend, which runs all members serially on
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
   can start before all workers are up and picks up workers as they join. On
   clusters that charge for whole nodes (e.g. Derecho), pass `workers_per_node` to
   [`add_workers`](@ref) to put several workers, one per GPU, in each
   allocation.

3. [`SharedFilesBackend`](@ref): The driver posts each member to a directory
   the workers can see, and workers started by any means pull members from it
   with [`run_worker`](@ref). Workers may join or leave at any time, a member
   whose worker dies is handed to another, and the driver can be restarted
   mid-iteration. Use it when workers live outside the driver's scheduler or
   when the driver should not hold an allocation.

4. HPC Cluster Backends: These backends schedule forward model runs on HPC clusters using Slurm or PBS.
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
    separate scheduler job per ensemble member and iteration; each job
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
    joins, so workers that join later get the same setup.

## Choosing the right backend for calibration

The right backend is largely determined by the computational cost of your
forward model.

If your model is simple or you are debugging, use the `JuliaBackend`.

If your model requires one CPU core or GPU, use the `WorkerBackend`.

If your forward model requires parallelization across multiple cores or GPUs,
choose one of the HPC cluster backends. These allow you to allocate more
resources to each forward model using Slurm or PBS.

If your workers live outside the driver's scheduler, on another cluster or on
rented machines, use the `SharedFilesBackend`.

## Worker registry

For a [`WorkerBackend`](@ref), the driver keeps a record of every worker
launched with [`add_workers`](@ref): its scheduler job id, Distributed id,
output file, and [`WorkerState`](@ref ClimaCalibrate.Backend.Workers.WorkerState).
Use it to see which workers are queued, idle, or running a member, why a worker
ended, and where its log is, without querying the scheduler. List the workers
in the driver with [`worker_rows`](@ref):

```julia-repl
julia> worker_rows()
3-element Vector{ClimaCalibrate.Backend.Workers.WorkerRecord}:
 Worker 1 in group 1: busy, job 4182734, pid 2, output /scratch/logs/worker-1.out
 Worker 2 in group 1: ready, job 4182735, pid 3, output /scratch/logs/worker-2.out
 Worker 3 in group 1: ended (scheduler reported FAILED), job 4182736, output /scratch/logs/worker-3.out
```

Pass `states` to filter them, e.g.
`worker_rows(; states = (ClimaCalibrate.Backend.Workers.ENDED,))` for the
workers that have ended. The registry lives in the driver's memory only.

Each `add_workers` call is a launch group with a desired worker count.
Every few seconds during a calibration, the driver checks the registry against
the scheduler: it asks `squeue` or `qstat` about live jobs, ends workers that are
not ready within `startup_timeout` seconds of submission (12 hours by default),
and launches replacements for any group below its desired count, up to that
group's `max_relaunches`. The only other way to grow the pool mid-run is to call
`add_workers` again. A worker that dies while running a member still counts as a
failed member and its replacement picks up later members. A worker that fails to
load its code is stopped and replaced. Once two workers of a group fail to load
their code and none has succeeded, the group launches no more workers, since
the setup code is likely broken.

When the driver exits, an `atexit` hook cancels every job the session
submitted by id and ends the live rows with
`exit_reason = "cancel_worker_jobs"`.

## Shared files backend

Use a [`SharedFilesBackend`](@ref) when something other than the driver starts
the workers: you, on a multi-GPU machine, `qsub` on Derecho, or a cloud
provider. The driver and the workers share only `output_dir`.

1. Write the driver script as for any backend, with a `SharedFilesBackend()`.
   The quickstart script works with that one change:

   ```julia
   eki = ClimaCalibrate.calibrate(
       ClimaCalibrate.SharedFilesBackend(),
       ekp,
       SurfaceFluxModelInterface(output_dir, ensemble_size),
       n_iterations,
       prior,
       output_dir,
   )
   ```

2. Write a worker script that builds the same interface with the same
   `output_dir` and calls [`run_worker`](@ref). For the quickstart, where
   `output_dir` is `my_experiment` and the ensemble has three members:

   ```julia
   import ClimaCalibrate
   include(joinpath(
       pkgdir(ClimaCalibrate),
       "experiments",
       "surface_fluxes_perfect_model",
       "model_interface.jl",
   ))
   ClimaCalibrate.run_worker(
       SurfaceFluxModelInterface("my_experiment", 3),
       "my_experiment",
   )
   ```

   When `model_interface.jl` evaluates to the interface, as the surface
   fluxes one does, the worker is `run_worker(include(path), output_dir)`.

3. Start the workers, one per GPU with two threads each, then the driver:

   ```sh
   for i in 0 1 2 3; do
       CUDA_VISIBLE_DEVICES=$i julia --project=experiment --threads=2 \
           worker.jl > worker-$i.log 2>&1 &
   done
   julia --project=experiment driver.jl
   ```

   The order does not matter. A worker with nothing to run polls every 5
   seconds and exits after 10 minutes, `idle_timeout`. The second thread
   renews the worker's claim while the forward model runs. A worker on
   another machine needs only to mount `output_dir`, so on Derecho each
   `qsub` job runs the same worker command against GLADE.

4. Watch `worker-*.log` and the driver's output. Each member directory under
   `output_dir/iteration_XXX/` gains `task.toml` when posted, `claim.toml`
   while a worker runs it, then `done.toml` or `failed.toml` with the worker
   id and any error text. The driver moves on once every member has one of
   the two.

### On a bucket

Workers that cannot mount `output_dir`, on another cluster or on rented
machines, meet the driver in an S3-compatible bucket instead. Both sides
load `AWSS3` and name the same
[`BucketFiles`](@ref ClimaCalibrate.SharedFiles.BucketFiles), with
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` set in both environments:

```julia
using AWSS3
files = ClimaCalibrate.SharedFiles.BucketFiles(
    "https://<account>.r2.cloudflarestorage.com",
    "clima-calibrations",
    "experiments/my_experiment",
)
```

The driver passes it to the backend with the outputs `observation_map`
reads, as paths relative to the member directory, and keeps its
`output_dir` local:

```julia
backend = SharedFilesBackend(; files, outputs = ["output_active"])
```

Each calibration gets its own prefix in the bucket, made on the first
iteration and kept in `output_dir/shared_files_prefix`, so a rerun resumes
where it stopped and a new `output_dir` starts clean. The driver logs the
prefixed bucket when it posts an iteration. A worker started by hand passes
that to `run_worker` with a local `output_dir` of its own:

```julia
run_worker(MyModel("/output", 10), "/output"; files = BucketFiles("https://<account>.r2.cloudflarestorage.com/clima-calibrations/experiments/my_experiment/my_experiment-1795458247"))
```

The driver uploads each member's `parameters.toml` before posting it. The
worker downloads that, runs the member locally, uploads `outputs`, and lists
them in `done.toml`. The driver downloads them into its `output_dir` once the
member is done. Transfers pass through memory, so keep `outputs` to what the
observation map needs. Bucket copies are leftovers once downloaded, so give
the bucket a lifecycle rule that deletes objects after a few days.

On Vast.ai, the backend rents the instances itself and destroys them when
the driver exits:

```julia
import ClimaCalibrate.Backend: Vast
backend = SharedFilesBackend(;
    files,
    outputs = ["output_active"],
    workers = Vast.Workers(4; gpu = "L40S", disk = 200),
)
```

Each instance runs the `clima-base` image, whose entrypoint clones the
repository holding `model_interface_filepath(interface)` at the branch
checked out on the driver, instantiates the driver's active project, and
runs `run_worker()`, which includes that file and runs the interface it
evaluates to. So the experiment's `model_interface.jl` ends with the
interface, the branch is pushed before the driver starts, the project must
hold `AWSS3`, and the project, the interface file, and `output_dir` all live
inside the repository, since the worker resolves them against its clone.
The `vastai` tool must be installed, with an API key holding
`instance_read`, `instance_write`, and `misc`.

When something dies:

- A worker that dies or hangs stops renewing its claim. After `lease`
  seconds, 300 by default, the driver hands the member to another worker.
  After three attempts the member is failed.
- A forward model that raises writes `failed.toml` and is not retried within
  the iteration. `failure_rate` decides whether the iteration halts. Rerunning
  the calibration retries failed members.
- A driver that dies can be restarted with the same script. It reposts only
  what is not done. Restart within the workers' `idle_timeout` or start new
  workers.
- With members open and no worker claiming any for `wait_timeout` seconds, 6
  hours by default, the iteration halts.

The lease compares the driver's clock with modification times stamped by the
worker's host, so the hosts' clocks must agree to well within `lease`.

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
`OUT_OF_MEMORY`, `NODE_FAIL`, `BOOT_FAIL`, `DEADLINE`, `PREEMPTED`, `REVOKED`,
`SPECIAL_EXIT`, and `LAUNCH_FAILED` all map to `FAILED`. A state the package does not recognize is treated
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
warning is logged. A member that has been running for longer than the backend's
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

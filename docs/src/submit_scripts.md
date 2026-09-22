# Writing submission scripts

This page provides concrete examples and best practices for running calibrations
on HPC clusters using ClimaCalibrate.jl. The examples assume basic familiarity
with either Slurm or PBS job schedulers.

## Overview

ClimaCalibrate.jl supports two main approaches for running calibrations on HPC
clusters:

1. **WorkerBackend**: Uses Julia's distributed computing capabilities
   with workers managed by the job scheduler
2. **HPCBackends**: Directly submits individual model runs as
   separate jobs to the scheduler

The choice between these approaches depends on your cluster's resource
allocation policies and your model's computational requirements. For more
information, see the [Backends](@ref Backends) page.

## WorkerBackend on a Slurm cluster

When using [`WorkerBackend`](@ref) on a Slurm cluster, request minimal
resources for the top-level script. Each worker is submitted as its own batch
job by the [`SlurmManager`](@ref), with the resources given to
[`add_workers`](@ref).

```bash
#!/bin/bash
#SBATCH --job-name=slurm_calibration
#SBATCH --output=calibration_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Set environment variables for CliMA
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="SINGLETON"

# Load required modules
module load climacommon

# Build and run the Julia code
julia --project=calibration -e 'using Pkg; Pkg.instantiate(;verbose=true)'
julia --project=calibration calibration_script.jl
```

**Key points:**
- Requests only 1 CPU core for the main script
- Workers are launched as separate Slurm jobs. Their walltime, CPUs, and GPUs
  come from `add_workers`, e.g. `add_workers(5; time = 120, device = :gpu)`
- Uses `%j` in output/error file names to interpolate the job ID
- Run as many workers as ensemble members to parallelize across all members

## WorkerBackend on a PBS cluster

As on Slurm, request minimal resources for the top-level script. Each worker
acquires its own resource allocation through the [`PBSManager`](@ref).

```bash
#!/bin/bash
#PBS -N pbs_calibration
#PBS -o calibration_${PBS_JOBID}.out
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=2GB

# Set environment variables for CliMA
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="SINGLETON"

# Set temporary directory
export TMPDIR=$SCRATCH/tmp && mkdir -p $TMPDIR

# Load required modules
module load climacommon

# Build and run the Julia code
julia --project=calibration -e 'using Pkg; Pkg.instantiate(;verbose=true)'
julia --project=calibration calibration_script.jl
```

**Key points:**
- Requests only 1 CPU core for the main script
- Workers will be launched as separate PBS jobs with their own resource
  allocations
- Uses `${PBS_JOBID}` to include the job ID in output file names

## HPC Backend Approach

The [`HPCBackend`](@ref)s directly submit individual forward model runs as
separate jobs to the scheduler. This approach is ideal when:
- Your forward model requires multiple CPU cores or GPUs
- You need fine-grained control over resource allocation per model run

Since each model run consists of an independent resource allocation, minimal
resources are needed to run the top-level calibration script. For a Slurm
cluster, here is a minimal submission script:

```bash
#!/bin/bash
#SBATCH --job-name=slurm_calibration
#SBATCH --output=calibration_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Load required modules
module load climacommon

# Build and run the Julia code
julia --project=calibration -e 'using Pkg; Pkg.instantiate(;verbose=true)'
julia --project=calibration calibration_script.jl
```
The [`WorkerBackend`](@ref) scripts above request the same minimal allocation
and can be reused for either scheduler.

## Resource Configuration

Configurations of the jobs submitted by the HPC backends are set by the
[`SlurmConfig`](@ref) for clusters using Slurm or [`PBSConfig`](@ref) for
clusters using PBS.

```@setup config
import ClimaCalibrate
```

```@example config
ClimaCalibrate.SlurmConfig(;
    directives = [
        :ntasks => 1,
        :gpus_per_task => 1,
        :cpus_per_task => 12,
        :time => 720,
    ],
    modules = ["climacommon"],
    env_vars = [
        "CLIMACOMMS_CONTEXT" => "SINGLETON",
        "CLIMACOMMS_DEVICE" => "CUDA",
    ],
)

nothing # hide
```

This example creates a Slurm configuration for a job with a single task, using
12 CPUs and 1 GPU, and a runtime of 720 minutes. It loads the latest version of
climacommon and explicitly sets environment variables for ClimaComms. The same
keyword arguments can also be passed to the `PBSConfig`.

!!! note "Backend constructors"
    To simplify the process of constructing a backend, you can pass
    `directives`, `modules`, and `env_vars` as keyword arguments to the backend
    constructor.

    ```@example config
    ClimaCalibrate.ClimaGPUBackend(;
         directives = [
            :ntasks => 1,
            :gpus_per_task => 1,
            :cpus_per_task => 12,
            :time => 720,
         ],
         modules = ["climacommon"],
         env_vars = [
            "CLIMACOMMS_CONTEXT" => "SINGLETON",
            "CLIMACOMMS_DEVICE" => "CUDA",
         ],
    )

    nothing # hide
    ```

## Environment Variables

Set these environment variables in your submission script:

- `CLIMACOMMS_DEVICE`: Set to `"CUDA"` for GPU runs or `"CPU"` for CPU-only runs
- `CLIMACOMMS_CONTEXT`: Set to `"SINGLETON"` for [`WorkerBackend`](@ref). The
  context is automatically set to `"MPI"` for HPC backends.

## Scheduler troubleshooting

### Common Issues

1. **Worker Timeout**: Increase `ENV["JULIA_WORKER_TIMEOUT"]` in your Julia
   session if workers are timing out
2. **Memory Issues**: Monitor memory usage and adjust `--mem` or `-l mem`
   accordingly.
3. **GPU Allocation**: Ensure `--gpus-per-task` or `-l select` matches the
   GPUs the model needs
4. **Module Conflicts**: Use `module purge` and ensure your MODULEPATH is set
   before loading required modules

### Debugging Commands

```bash
# Check job status (Slurm)
squeue -u $USER

# Check job status (PBS)
qstat -u $USER

# View job logs
tail -f calibration_<jobid>.out

# Check resource usage
seff <jobid>  # Slurm
qstat -f <jobid>  # PBS
```

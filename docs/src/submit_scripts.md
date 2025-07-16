# Job Submission Scripts for HPC Clusters

This page provides concrete examples and best practices for running calibrations on HPC clusters using ClimaCalibrate.jl. The examples assume basic familiarity with either Slurm or PBS job schedulers.

## Overview

ClimaCalibrate.jl supports two main approaches for running calibrations on HPC clusters:

1. **WorkerBackend**: Uses Julia's distributed computing capabilities with workers managed by the job scheduler
2. **HPC Backends**: Directly submits individual model runs as separate jobs to the scheduler

The choice between these approaches depends on your cluster's resource allocation policies and your model's computational requirements.
For more information, see the Backends page.

## WorkerBackend on a Slurm cluster

When using `WorkerBackend` on a Slurm cluster, allocate resources at the top level since Slurm allows nested resource allocations. Each worker will inherit one task from the Slurm allocation.

```bash
#!/bin/bash
#SBATCH --job-name=slurm_calibration
#SBATCH --output=calibration_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G

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
- `--ntasks=5`: Requests 5 tasks, each worker gets one task
- `--cpus-per-task=4`: Each worker gets 4 CPU cores
- `--gpus-per-task=1`: Each worker gets 1 GPU
- Uses `%j` in output/error file names to interpolate the job ID

## WorkerBackend on a PBS cluster

Since PBS does not support nested resource allocations, request minimal resources for the top-level script. Each worker will acquire its own resource allocation through the `PBSManager`.

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
- Workers will be launched as separate PBS jobs with their own resource allocations
- Uses `${PBS_JOBID}` to include the job ID in output file names

## HPC Backend Approach

HPC backends directly submit individual forward model runs as separate jobs to the scheduler. This approach is ideal when:
- Your forward model requires multiple CPU cores or GPUs
- You need fine-grained control over resource allocation per model run
- Your cluster doesn't support nested allocations

Since each model run consists of an independent resource allocation, minimal resources are needed to run the top-level calibration script.
For a slurm cluster, here is a minimal submission script:
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
For a PBS cluster, the script in the WorkerBackend section can be reused since it already specifies a minimal resource allocation.

## Resource Configuration

### CPU-Only Jobs

For CPU-only forward models:

```julia
hpc_kwargs = Dict(
    :time => 30,
    :ntasks => 1,
    :cpus_per_task => 8,
    :mem => "16G"
)
```

### GPU Jobs

For GPU-accelerated forward models:

```julia
hpc_kwargs = Dict(
    :time => 60,
    :ntasks => 1,
    :cpus_per_task => 4,
    :gpus_per_task => 1,
    :mem => "32G"
)
```

### Multi-Node Jobs

For models requiring multiple nodes:

```julia
hpc_kwargs = Dict(
    :time => 120,
    :ntasks => 16,
    :cpus_per_task => 4,
    :nodes => 4,
    :mem => "64G"
)
```

## Environment Variables

Set these environment variables in your submission script:

- `CLIMACOMMS_DEVICE`: Set to `"CUDA"` for GPU runs or `"CPU"` for CPU-only runs
- `CLIMACOMMS_CONTEXT`: Set to `"SINGLETON"` for WorkerBackend. The context is automatically set to `"MPI"` for HPC backends

## Troubleshooting

### Common Issues

1. **Worker Timeout**: Increase `ENV["JULIA_WORKER_TIMEOUT"]` in your Julia session if workers are timing out
2. **Memory Issues**: Monitor memory usage and adjust `--mem` or `-l mem` accordingly. 
3. **GPU Allocation**: Ensure `--gpus-per-task` or `-l select` is set correctly
4. **Module Conflicts**: Use `module purge` and ensure your MODULEPATH is set before loading required modules

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

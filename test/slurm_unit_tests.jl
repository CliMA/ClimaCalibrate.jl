# Unit tests for slurm job control functionality
using Test
import ClimaCalibrate as CAL

const OUTPUT_DIR = "test"
const ITER = 1
const MEMBER = 1
const TIME_LIMIT = 90
const NTASKS = 1
const CPUS_PER_TASK = 16
const GPUS_PER_TASK = 1
const EXPERIMENT_DIR = "exp/dir"
const MODEL_INTERFACE = "model_interface.jl"
const MODULE_LOAD_STR = CAL.module_load_string(CAL.CaltechHPCBackend())
const hpc_kwargs = CAL.kwargs(
    time = TIME_LIMIT,
    cpus_per_task = CPUS_PER_TASK,
    gpus_per_task = GPUS_PER_TASK,
)

# Time formatting tests
@test CAL.format_slurm_time(TIME_LIMIT) == "01:30:00"
@test CAL.format_slurm_time(1) == "00:01:00"
@test CAL.format_slurm_time(60) == "01:00:00"
@test CAL.format_slurm_time(1440) == "1-00:00:00"

# Generate and validate sbatch file contents
sbatch_file = CAL.generate_sbatch_script(
    ITER,
    MEMBER,
    OUTPUT_DIR,
    EXPERIMENT_DIR,
    MODEL_INTERFACE,
    MODULE_LOAD_STR,
    hpc_kwargs,
)

expected_sbatch_contents = """
#!/bin/bash
#SBATCH --job-name=run_1_1
#SBATCH --output=test/iteration_001/member_001/model_log.txt
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00

export MODULEPATH="/resnick/groups/esm/modules:\$MODULEPATH"
module purge
module load climacommon/2024_10_09
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="MPI"

srun --output=test/iteration_001/member_001/model_log.txt --open-mode=append julia  --project=exp/dir -e '

    import ClimaCalibrate as CAL
    iteration = 1; member = 1
    model_interface = "model_interface.jl"; include(model_interface)
    experiment_dir = "exp/dir"
    CAL.forward_model(iteration, member)
    CAL.write_model_completed("test", iteration, member)
'
exit 0
"""

for (generated_str, test_str) in
    zip(split(sbatch_file, "\n"), split(expected_sbatch_contents, "\n"))
    # Test one line at a time to see discrepancies
    @test generated_str == test_str
end

# Helper function for submitting commands and checking job status
function submit_cmd_helper(cmd)
    sbatch_filepath, io = mktemp()
    write(io, cmd)
    close(io)
    jobid = CAL.submit_slurm_job(sbatch_filepath)
    sleep(1)  # Allow time for the job to start
    return jobid
end

# Test job lifecycle
test_cmd = """
#!/bin/bash
#SBATCH --time=00:01:00
sleep 30
"""

jobid = submit_cmd_helper(test_cmd)
@test CAL.job_running(jobid) || CAL.job_pending(jobid)

sleep(480)  # Ensure job finishes. To debug, lower sleep time or comment it out
@test CAL.job_status(jobid) == :COMPLETED
@test CAL.job_completed(jobid)
@test CAL.job_success(jobid)

# Test job cancellation
jobid = submit_cmd_helper(test_cmd)
CAL.kill_job(jobid)
sleep(5)
@test CAL.job_status(jobid) == :COMPLETED
@test CAL.job_completed(jobid)

# Test batch cancellation
jobids = ntuple(x -> submit_cmd_helper(test_cmd), 5)

CAL.kill_job.(jobids)
sleep(5)
for jobid in jobids
    @test CAL.job_completed(jobid)
end

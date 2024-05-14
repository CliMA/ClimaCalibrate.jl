# Unit tests for slurm job control functionality
using Test
import ClimaCalibrate as CAL

const OUTPUT_DIR = "test"
const ITER = 1
const MEMBER = 1
const TIME_LIMIT = 90
const NTASKS = 1
const PARTITION = "expansion"
const CPUS_PER_TASK = 16
const GPUS_PER_TASK = 1
const EXPERIMENT_DIR = "exp/dir"
const MODEL_INTERFACE = "model_interface.jl"

const slurm_kwargs = CAL.kwargs(
    time = TIME_LIMIT,
    partition = PARTITION,
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
    MODEL_INTERFACE;
    slurm_kwargs,
)

expected_sbatch_contents = """
#!/bin/bash
#SBATCH --job-name=run_1_1
#SBATCH --output=test/iteration_001/member_001/model_log.txt
#SBATCH --partition=expansion
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00

export MODULEPATH=/groups/esm/modules:\$MODULEPATH
module purge
module load climacommon/2024_04_30


srun --output=test/iteration_001/member_001/model_log.txt --open-mode=append julia --project=exp/dir -e '
import ClimaCalibrate as CAL
iteration = 1; member = 1
model_interface = "model_interface.jl"; include(model_interface)

experiment_dir = "exp/dir"
experiment_config = CAL.ExperimentConfig(experiment_dir)
experiment_id = experiment_config.id
physical_model = CAL.get_forward_model(Val(Symbol(experiment_id)))
CAL.run_forward_model(physical_model, CAL.get_config(physical_model, member, iteration, experiment_dir))
@info "Forward Model Run Completed" experiment_id physical_model iteration member'
"""

for (generated_str, test_str) in
    zip(split(sbatch_file, "\n"), split(expected_sbatch_contents, "\n"))
    @test generated_str == test_str
end

# Helper function for submitting commands and checking job status
function submit_cmd_helper(cmd)
    sbatch_filepath, io = mktemp()
    write(io, cmd)
    close(io)
    jobid = CAL.submit_sbatch_job(sbatch_filepath)
    sleep(1)  # Allow time for the job to start
    return jobid
end

# Test job lifecycle
test_cmd = """
#!/bin/bash
#SBATCH --time=00:00:10
#SBATCH --partition=expansion
sleep 10
"""

jobid = submit_cmd_helper(test_cmd)
@test CAL.job_status(jobid) == "RUNNING"
@test CAL.job_running(CAL.job_status(jobid))

sleep(180)  # Ensure job finishes. To debug, lower sleep time or comment out the code block
@test CAL.job_status(jobid) == "COMPLETED"
@test CAL.job_completed(CAL.job_status(jobid))
@test CAL.job_success(CAL.job_status(jobid))

# Test job cancellation
jobid = submit_cmd_helper(test_cmd)
CAL.kill_slurm_job(jobid)
sleep(1)
@test CAL.job_status(jobid) == "FAILED"
@test CAL.job_completed(CAL.job_status(jobid)) &&
      CAL.job_failed(CAL.job_status(jobid))

# Test batch cancellation
jobids = ntuple(x -> submit_cmd_helper(test_cmd), 5)
CAL.kill_all_jobs(jobids)
for jobid in jobids
    @test CAL.job_completed(CAL.job_status(jobid))
    @test CAL.job_failed(CAL.job_status(jobid))
end

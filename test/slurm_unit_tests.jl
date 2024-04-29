# Unit tests for slurm job control functionality

using Test

import CalibrateAtmos as CAL

output_dir = "test"
iter = 1
member = 1
time_limit = 90
ntasks = 1
partition = "expansion"
cpus_per_task = 16
gpus_per_task = 1
experiment_dir = "exp/dir"
model_interface = "model_interface.jl"

@test CAL.format_slurm_time(time_limit) == "01:30:00"
@test CAL.format_slurm_time(1) == "00:01:00"
@test CAL.format_slurm_time(60) == "01:00:00"
@test CAL.format_slurm_time(1440) == "1-00:00:00"

sbatch_file = CAL.generate_sbatch_file_contents(;
    output_dir,
    iter,
    member,
    time_limit,
    ntasks,
    partition,
    cpus_per_task,
    gpus_per_task,
    experiment_dir,
    model_interface,
)

test_string = """
#!/bin/bash
#SBATCH --job-name=run_1_1
#SBATCH --time=90
#SBATCH --ntasks=1
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=test/iteration_001/member_001/model_log.txt

export MODULEPATH=/groups/esm/modules:\$MODULEPATH
module purge
module load climacommon/2024_04_05

srun --output=test/iteration_001/member_001/model_log.txt --open-mode=append julia --project=exp/dir -e '
import CalibrateAtmos as CAL
iteration = 1; member = 1
model_interface = "model_interface.jl"; include(model_interface)

experiment_dir = "exp/dir"
experiment_config = CAL.ExperimentConfig(experiment_dir)
experiment_id = experiment_config.id
physical_model = CAL.get_forward_model(Val(Symbol(experiment_id)))
CAL.run_forward_model(physical_model, CAL.get_config(physical_model, member, iteration, experiment_dir))
@info "Forward Model Run Completed" experiment_id physical_model iteration member'
"""

for (generated, test_str) in zip(split(sbatch_file, "\n"), split(test_string, "\n"))
    @test generated == test_str
end

# Test job status
test_cmd = """
#!/bin/bash
#SBATCH --time=00:00:10
sleep 1"""

function submit_cmd_helper()
    sbatch_filepath, io = mktemp()
    write(io, test_cmd)
    close(io)
    return CAL.submit_job(sbatch_filepath)
end

jobid = submit_cmd_helper()
_status = CAL.job_status(jobid)
@test _status == "RUNNING"
@test CAL.job_running(_status)
# Ensure job finishes
sleep(60)
_status = CAL.job_status(jobid)
@test _status == "COMPLETED"
@test CAL.job_completed(_status)
@test CAL.job_success(_status)

# Test cancellation
jobid = submit_cmd_helper()
@test CAL.job_status(jobid) == "RUNNING"
CAL.kill_slurm_job(jobid)
_status = CAL.job_status(jobid)
@test _status == "FAILED"
@test CAL.job_completed(_status) && CAL.job_failed(_status)

jobids = ntuple(x -> submit_cmd_helper(), 5)

CAL.kill_all_jobs(jobids)

for jobid in jobids
    job_status = CAL.job_status(jobid)
    @test CAL.job_completed(job_status) && CAL.job_failed(job_status)
end

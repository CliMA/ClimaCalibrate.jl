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
const MODULE_LOAD_STR = CAL.module_load_string(CAL.DerechoBackend)
const hpc_kwargs =
    CAL.kwargs(walltime = TIME_LIMIT, select = "2:ncpus=128:ngpus=2")

# Time formatting tests
@test CAL.format_pbs_time(TIME_LIMIT) == "01:30:00"
@test CAL.format_pbs_time(1) == "00:01:00"
@test CAL.format_pbs_time(60) == "01:00:00"
@test CAL.format_pbs_time(1440) == "24:00:00"

# Generate and validate sbatch file contents
pbs_file = CAL.generate_pbs_script(
    ITER,
    MEMBER,
    OUTPUT_DIR,
    EXPERIMENT_DIR,
    MODEL_INTERFACE,
    MODULE_LOAD_STR;
    hpc_kwargs,
)

expected_pbs_contents = """
#!/bin/bash
#PBS -N run_1_1
#PBS -j oe
#PBS -A UCIT0011
#PBS -q preempt
#PBS -o test/iteration_001/member_001/model_log.txt
#PBS -l walltime=01:30:00
#PBS -l select=2:ncpus=128:ngpus=2
set -euo pipefail
export PBS_ACCOUNT="UCIT0011"
export MODULEPATH="/glade/campaign/univ/ucit0011/ClimaModules-Derecho:\$MODULEPATH" 
module purge
module load climacommon

\$MPITRAMPOLINE_EXEC -n 4 -ppn 2 set_gpu_rank julia --project=exp/dir -e '
import ClimaCalibrate as CAL
iteration = 1; member = 1
model_interface = "model_interface.jl"; include(model_interface)

experiment_dir = "exp/dir"
CAL.run_forward_model(CAL.set_up_forward_model(member, iteration, experiment_dir))'

exit 0
"""

for (generated_str, test_str) in
    zip(split(pbs_file, "\n"), split(expected_pbs_contents, "\n"))
    @test generated_str == test_str
end

# Helper function for submitting commands and checking job status
function submit_cmd_helper(cmd)
    sbatch_filepath, io = mktemp()
    write(io, cmd)
    close(io)
    jobid = CAL.submit_pbs_job(sbatch_filepath)
    sleep(1)  # Allow time for the job to start
    return jobid
end

# Test job lifecycle
test_cmd = """
#!/bin/bash
#PBS -j oe
#PBS -A UCIT0011
#PBS -q develop
#PBS -l walltime=00:00:12
#PBS -l select=1:ncpus=1

sleep 10
"""

jobid = submit_cmd_helper(test_cmd)
@test CAL.job_status(jobid) == :RUNNING
@test CAL.job_running(CAL.job_status(jobid))

sleep(180)  # Ensure job finishes. To debug, lower sleep time or comment out the code block
@test CAL.job_status(jobid) == :COMPLETED
@test CAL.job_completed(CAL.job_status(jobid))
@test CAL.job_success(CAL.job_status(jobid))

# Test job cancellation
jobid = submit_cmd_helper(test_cmd)
CAL.kill_slurm_job(jobid)
sleep(1)
@test CAL.job_status(jobid) == :FAILED
@test CAL.job_completed(CAL.job_status(jobid)) &&
      CAL.job_failed(CAL.job_status(jobid))

# Test batch cancellation
jobids = ntuple(x -> submit_cmd_helper(test_cmd), 5)

CAL.kill_job.(jobids)
for jobid in jobids
    @test CAL.job_completed(CAL.job_status(jobid))
    @test CAL.job_failed(CAL.job_status(jobid))
end

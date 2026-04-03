# Unit tests for slurm job control functionality
using Test
import ClimaCalibrate

@testset "Format slurm time" begin
    @test ClimaCalibrate.Backend.format_slurm_time(1) == "00:01:00"
    @test ClimaCalibrate.Backend.format_slurm_time(60) == "01:00:00"
    @test ClimaCalibrate.Backend.format_slurm_time(90) == "01:30:00"
    @test ClimaCalibrate.Backend.format_slurm_time(1440) == "1-00:00:00"
end

@testset "Generate slurm script" begin
    backend_type = ClimaCalibrate.get_backend()
    if !(backend_type <: ClimaCalibrate.SlurmBackend)
        @info "Backend identified is $backend_type which is not a SlurmBackend. Skipping generating a slurm script"
        return
    end

    time_limit = 1
    cpus_per_task = 16
    gpus_per_task = 1
    hpc_kwargs = Dict(
        :time => time_limit,
        :cpus_per_task => cpus_per_task,
        :gpus_per_task => gpus_per_task,
    )
    backend = backend_type(; hpc_kwargs)

    script = """
    sleep(30)
    """

    experiment_dir = ClimaCalibrate.project_dir()
    job_body = """
    julia --project=$experiment_dir -e '$script'
    """

    job_name = "slurm_job"
    output = "output.txt"

    sbatch_string =
        ClimaCalibrate.make_job_script(backend, job_body; job_name, output)

    module_str = ClimaCalibrate.Backend.module_load_string(backend)

    expected_sbatch_contents = """
    #!/bin/bash
    #SBATCH --job-name=$job_name
    #SBATCH --output=$output
    #SBATCH --gpus-per-task=$gpus_per_task
    #SBATCH --cpus-per-task=$cpus_per_task
    #SBATCH --time=$(ClimaCalibrate.Backend.format_slurm_time(time_limit))

    $module_str
    export CLIMACOMMS_DEVICE="CUDA"
    export CLIMACOMMS_CONTEXT="MPI"

    srun --output=$output --open-mode=append julia --project=$experiment_dir -e 'sleep(30)
    '

    exit 0
    """

    for (generated_str, test_str) in
        zip(split(sbatch_string, "\n"), split(expected_sbatch_contents, "\n"))
        # Test one line at a time to see discrepancies
        @test generated_str == test_str
    end
end

@testset "Submit slurm job" begin
    backend_type = ClimaCalibrate.get_backend()
    if !(backend_type <: ClimaCalibrate.SlurmBackend)
        @info "Backend identified is $backend_type which is not a SlurmBackend. Skipping submitting a slurm job"
        return
    end

    # Need the backend to submit the job
    backend = backend_type()

    job_script = """
    #!/bin/bash
    #SBATCH --time=00:01:00
    sleep 30
    """

    # Helper function to wait for a job to complete
    function wait_for(job, t)
        curr_time = time()
        while time() - curr_time < t
            ClimaCalibrate.job_status(job) == :COMPLETED && break
            sleep(3)
        end
        return nothing
    end

    @testset "Submit a slurm job and cancel it" begin
        job = ClimaCalibrate.submit_job(backend, job_script)
        ClimaCalibrate.cancel_job(job)
        # Cancelling is quicker than waiting for a job to complete so we wait
        # for only one minute
        wait_for(job, 60)
        @test ClimaCalibrate.job_status(job) == :COMPLETED
        @test ClimaCalibrate.iscompleted(job)
    end

    @testset "Submit multiple slurm jobs and cancel them" begin
        jobs = ntuple(x -> ClimaCalibrate.submit_job(backend, job_script), 5)
        ClimaCalibrate.cancel_job.(jobs)
        wait_for.(jobs, 60)
        for job in jobs
            @test ClimaCalibrate.iscompleted(job)
        end
    end

    @testset "Submit a single slurm job and wait for completion" begin
        job = ClimaCalibrate.submit_job(backend, job_script)

        # Test for job status and completion
        @test ClimaCalibrate.isrunning(job) || ClimaCalibrate.ispending(job)
        wait_for(job, 480)
        @test ClimaCalibrate.job_status(job) == :COMPLETED
        @test ClimaCalibrate.iscompleted(job)
        @test ClimaCalibrate.issuccess(job)
    end
end

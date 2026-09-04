# Unit tests for slurm job control functionality
using Test
import ClimaCalibrate

@testset "Submit slurm job" begin
    cluster_backend = ClimaCalibrate.backend_type()
    if !(cluster_backend <: ClimaCalibrate.SlurmBackend)
        @info "Backend identified is $cluster_backend, which is not a \
               SlurmBackend. Skipping submitting a slurm job"
        return
    end

    # Need the backend to submit the job
    # Note that the config is used to generate the job script but we will
    # manually make one ourself
    config = ClimaCalibrate.SlurmConfig(; directives = [:time => 60])
    backend = cluster_backend(config)

    job_script = """
    #!/bin/bash
    #SBATCH --time=00:01:00
    sleep 30
    """

    # Helper function to wait for a job to complete
    function wait_for(job, t)
        curr_time = time()
        while time() - curr_time < t
            ClimaCalibrate.iscompleted(job) && break
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
        # A cancelled job is not a successful one. Reporting it as COMPLETED
        # would make a crashed ensemble member indistinguishable from one that
        # succeeded
        @test ClimaCalibrate.job_status(job) == ClimaCalibrate.Backend.FAILED
        @test ClimaCalibrate.isfailed(job)
        @test ClimaCalibrate.iscompleted(job)
        @test !ClimaCalibrate.issuccess(job)
    end

    @testset "A slurm job that fails is reported as failed" begin
        failing_script = """
        #!/bin/bash
        #SBATCH --time=00:01:00
        exit 1
        """
        job = ClimaCalibrate.submit_job(backend, failing_script)
        wait_for(job, 480)
        @test ClimaCalibrate.isfailed(job)
        @test !ClimaCalibrate.issuccess(job)
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
        @test ClimaCalibrate.job_status(job) == ClimaCalibrate.Backend.COMPLETED
        @test ClimaCalibrate.iscompleted(job)
        @test ClimaCalibrate.issuccess(job)
    end
end

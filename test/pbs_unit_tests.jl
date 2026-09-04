# Unit tests for PBS job control functionality
using Test
import ClimaCalibrate

@testset "Submit PBS job" begin
    cluster_backend = ClimaCalibrate.backend_type()
    if !(cluster_backend <: ClimaCalibrate.DerechoBackend)
        @info "Backend identified is $cluster_backend, which is not a \
               DerechoBackend. Skipping submitting a PBS job"
        return
    end

    # Need the backend to submit the job
    backend = cluster_backend(; directives = [:time => 1])

    job_script = """
    #!/bin/bash
    #PBS -j oe
    #PBS -A UCIT0011
    #PBS -q preempt
    #PBS -l walltime=00:01:00
    #PBS -l select=1:ncpus=1:ngpus=1

    sleep 10
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

    @testset "Submit a PBS job and cancel it" begin
        job = ClimaCalibrate.submit_job(backend, job_script)
        ClimaCalibrate.cancel_job(job)
        # Derecho can be slow when updating their database when querying with
        # qstat, so we need to wait longer
        # If the test is flaky, increase the time to wait for
        wait_for(job, 180)
        # Whether PBS records a qdel'd job with the error substate depends on
        # how far it got, so only assert that it is no longer pending or running
        @test ClimaCalibrate.iscompleted(job)
    end

    @testset "A PBS job that fails is reported as failed" begin
        failing_script = """
        #!/bin/bash
        #PBS -l walltime=00:01:00
        exit 1
        """
        job = ClimaCalibrate.submit_job(backend, failing_script)
        wait_for(job, 240)
        @test ClimaCalibrate.isfailed(job)
        @test !ClimaCalibrate.issuccess(job)
    end

    @testset "Submit multiple PBS jobs and cancel them" begin
        jobs = ntuple(x -> ClimaCalibrate.submit_job(backend, job_script), 3)
        ClimaCalibrate.cancel_job.(jobs)
        wait_for.(jobs, 180)
        for job in jobs
            @test ClimaCalibrate.iscompleted(job)
        end
    end

    @testset "Submit a single PBS job and wait for completion" begin
        job = ClimaCalibrate.Backend.submit_job(backend, job_script)

        # Test for job status and completion
        @test ClimaCalibrate.isrunning(job) || ClimaCalibrate.ispending(job)
        wait_for(job, 240)
        @test ClimaCalibrate.job_status(job) == ClimaCalibrate.Backend.COMPLETED
        @test ClimaCalibrate.iscompleted(job)
        @test ClimaCalibrate.issuccess(job)
    end
end

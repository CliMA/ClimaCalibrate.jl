# Unit tests for PBS job control functionality
using Test
import ClimaCalibrate

@testset "Format PBS time" begin
    @test ClimaCalibrate.BetterBackend.format_pbs_time(1) == "00:01:00"
    @test ClimaCalibrate.BetterBackend.format_pbs_time(60) == "01:00:00"
    @test ClimaCalibrate.BetterBackend.format_pbs_time(90) == "01:30:00"
    @test ClimaCalibrate.BetterBackend.format_pbs_time(1440) == "24:00:00"
    @test ClimaCalibrate.BetterBackend.format_pbs_time(2880) == "48:00:00"
end

@testset "Generate PBS script" begin
    time_limit = 1
    cpus_per_task = 16
    gpus_per_task = 1
    ntasks = 2
    hpc_kwargs = Dict(
        :time => time_limit,
        :ntasks => ntasks,
        :cpus_per_task => cpus_per_task,
        :gpus_per_task => gpus_per_task,
    )
    backend = ClimaCalibrate.DerechoBackend(; hpc_kwargs)

    script = """
    sleep(30)
    """

    experiment_dir = ClimaCalibrate.project_dir()
    job_body = """
    julia --project=$experiment_dir -e '$script'
    """

    job_name = "pbs_job"
    output = "output.txt"
    pbs_string = ClimaCalibrate.BetterBackend.make_job_script(
        backend,
        job_body;
        job_name,
        output,
    )

    # TODO: climacommon version will fail depending on the backend!
    expected_pbs_contents = """
#!/bin/bash
#PBS -N $job_name
#PBS -j oe
#PBS -A UCIT0011
#PBS -q main
#PBS -o $output
#PBS -l job_priority=regular
#PBS -l walltime=$(ClimaCalibrate.BetterBackend.format_pbs_time(time_limit))
#PBS -l select=$ntasks:ncpus=$cpus_per_task:ngpus=$gpus_per_task:mpiprocs=1

export MODULEPATH="/glade/campaign/univ/ucit0011/ClimaModules-Derecho:\$MODULEPATH"
module purge
module load climacommon

export JULIA_MPI_HAS_CUDA=true
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="MPI"
\$MPITRAMPOLINE_MPIEXEC -n 2 -ppn 1 set_gpu_rank julia --project=$experiment_dir -e 'sleep(30)
'


    """

    for (generated_str, test_str) in
        zip(split(pbs_string, "\n"), split(expected_pbs_contents, "\n"))
        # Test one line at a time to see discrepancies
        @test generated_str == test_str
    end
end

@testset "Submit PBS job" begin
    backend_type = ClimaCalibrate.get_backend()
    if !(backend_type <: ClimaCalibrate.DerechoBackend)
        @info "Backend identified is $backend_type which is not a DerechoBackend. Skipping submitting a PBS job"
        return
    end

    # Need the backend to submit the job
    backend = backend_type()

    job_script = """
    #!/bin/bash
    #PBS -j oe
    #PBS -A UCIT0011
    #PBS -q preempt
    #PBS -l walltime=00:01:00
    #PBS -l select=1:ncpus=1

    sleep 10
    """

    job = ClimaCalibrate.BetterBackend.submit_job(backend, job_script)

    # Test for job status and completion
    @test ClimaCalibrate.BetterBackend.isrunning(job) ||
          ClimaCalibrate.BetterBackend.ispending(job)

    # Helper function to wait for a job to complete
    function wait_for(job, t)
        curr_time = time()
        while time() - curr_time < t
            ClimaCalibrate.BetterBackend.job_status(job) == :COMPLETED && break
            sleep(3)
        end
        return nothing
    end

    @testset "Submit a single PBS job and wait for completion" begin
        wait_for(job, 180)
        @test ClimaCalibrate.BetterBackend.job_status(job) == :COMPLETED
        @test ClimaCalibrate.BetterBackend.iscompleted(job)
        @test ClimaCalibrate.BetterBackend.issuccess(job)
    end

    @testset "Submit a PBS job and cancel it" begin
        job = ClimaCalibrate.BetterBackend.submit_job(backend, job_script)
        ClimaCalibrate.BetterBackend.kill_job(job)
        # Derecho can be slow when updating their database when querying with qstat,
        # so we need to wait longer
        # If the test is flaky, increase the time to wait for
        wait_for(job, 180)
        @test ClimaCalibrate.BetterBackend.job_status(job) == :COMPLETED
        @test ClimaCalibrate.BetterBackend.iscompleted(job)
    end

    @testset "Submit multiple PBS jobs and cancel them" begin
        jobs = ntuple(
            x -> ClimaCalibrate.BetterBackend.submit_job(backend, job_script),
            3,
        )
        ClimaCalibrate.BetterBackend.kill_job.(jobs)
        wait_for.(jobs, 180)
        for job in jobs
            @test ClimaCalibrate.BetterBackend.iscompleted(job)
        end
    end
end

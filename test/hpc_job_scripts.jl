using Test
import ClimaCalibrate

@testset "Format slurm time" begin
    @test ClimaCalibrate.Backend.format_slurm_time(1) == "00:01:00"
    @test ClimaCalibrate.Backend.format_slurm_time(60) == "01:00:00"
    @test ClimaCalibrate.Backend.format_slurm_time(90) == "01:30:00"
    @test ClimaCalibrate.Backend.format_slurm_time(1440) == "1-00:00:00"
end

@testset "Format PBS time" begin
    @test ClimaCalibrate.Backend.format_pbs_time(1) == "00:01:00"
    @test ClimaCalibrate.Backend.format_pbs_time(60) == "01:00:00"
    @test ClimaCalibrate.Backend.format_pbs_time(90) == "01:30:00"
    @test ClimaCalibrate.Backend.format_pbs_time(1440) == "24:00:00"
    @test ClimaCalibrate.Backend.format_pbs_time(2880) == "48:00:00"
end

@testset "Generate slurm script" begin
    backend_types = [
        ClimaCalibrate.GCPBackend,
        ClimaCalibrate.ClimaGPUBackend,
        ClimaCalibrate.CaltechHPCBackend,
    ]

    for backend_type in backend_types
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
        mpiexec_string =
            ClimaCalibrate.Backend._generate_mpiexec_string(backend, 1, output)

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

        $mpiexec_string julia --project=$experiment_dir -e 'sleep(30)
        '

        exit 0
        """

        for (generated_str, test_str) in zip(
            split(sbatch_string, "\n"),
            split(expected_sbatch_contents, "\n"),
        )
            # Test one line at a time to see discrepancies
            @test generated_str == test_str
        end
    end
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
    pbs_string =
        ClimaCalibrate.make_job_script(backend, job_body; job_name, output)

    module_str = ClimaCalibrate.Backend.module_load_string(backend)
    expected_pbs_contents = """
#!/bin/bash
#PBS -N $job_name
#PBS -j oe
#PBS -A UCIT0011
#PBS -q main
#PBS -o $output
#PBS -l job_priority=regular
#PBS -l walltime=$(ClimaCalibrate.Backend.format_pbs_time(time_limit))
#PBS -l select=$ntasks:ncpus=$cpus_per_task:ngpus=$gpus_per_task:mpiprocs=1

$module_str

export JULIA_MPI_HAS_CUDA=true
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="MPI"

cd \$PBS_O_WORKDIR
\$MPITRAMPOLINE_MPIEXEC -n 2 -ppn 1 set_gpu_rank julia --project=$experiment_dir -e 'sleep(30)
'


    """

    for (generated_str, test_str) in
        zip(split(pbs_string, "\n"), split(expected_pbs_contents, "\n"))
        # Test one line at a time to see discrepancies
        @test generated_str == test_str
    end
end

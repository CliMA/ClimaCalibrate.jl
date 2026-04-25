using Test
import ClimaCalibrate

@testset "Format slurm time" begin
    @test ClimaCalibrate.Backend.format_slurm_time(1) == "00:01:00"
    @test ClimaCalibrate.Backend.format_slurm_time(60) == "01:00:00"
    @test ClimaCalibrate.Backend.format_slurm_time(90) == "01:30:00"
    @test ClimaCalibrate.Backend.format_slurm_time(1440) == "1-00:00:00"
    @test ClimaCalibrate.Backend.format_slurm_time("00:01:00") == "00:01:00"
end

@testset "Format PBS time" begin
    @test ClimaCalibrate.Backend.format_pbs_time(1) == "00:01:00"
    @test ClimaCalibrate.Backend.format_pbs_time(60) == "01:00:00"
    @test ClimaCalibrate.Backend.format_pbs_time(90) == "01:30:00"
    @test ClimaCalibrate.Backend.format_pbs_time(1440) == "24:00:00"
    @test ClimaCalibrate.Backend.format_pbs_time(2880) == "48:00:00"
    @test ClimaCalibrate.Backend.format_pbs_time("00:01:00") == "00:01:00"
end

@testset "SlurmConfig" begin
    # No default directive for time
    @test_throws r"key :time" ClimaCalibrate.SlurmConfig(;
        directives = [],
        modules = [],
        env_vars = [],
    )

    # Minimal config
    config = ClimaCalibrate.SlurmConfig(;
        directives = [:time => 1],
        modules = [],
        env_vars = [],
    )

    @test collect(config.directives) ==
          [:time => "00:01:00", :gpus_per_task => 0]
    @test isempty(config.modules)
    @test collect(config.env_vars) ==
          ["CLIMACOMMS_DEVICE" => "CPU", "CLIMACOMMS_CONTEXT" => "MPI"]

    # Override default
    config = ClimaCalibrate.SlurmConfig(;
        directives = [:time => 1, :gpus_per_task => 10],
        modules = [],
        env_vars = [
            "CLIMACOMMS_CONTEXT" => "SINGLETON",
            "CLIMACOMMS_DEVICE" => "GPU",
        ],
    )

    @test collect(config.directives) ==
          [:time => "00:01:00", :gpus_per_task => 10]
    @test isempty(config.modules)
    @test collect(config.env_vars) ==
          ["CLIMACOMMS_CONTEXT" => "SINGLETON", "CLIMACOMMS_DEVICE" => "GPU"]
end

@testset "PBSConfig" begin
    # No default directive for time
    @test_throws r"key :time" config =
        ClimaCalibrate.PBSConfig(; directives = [], modules = [], env_vars = [])

    # Minimal config
    config = ClimaCalibrate.PBSConfig(;
        directives = [:time => 1],
        modules = [],
        env_vars = [],
    )

    @test collect(config.directives) == [
        :time => "00:01:00",
        :queue => "main",
        :ntasks => 1,
        :cpus_per_task => 1,
        :gpus_per_task => 0,
        :job_priority => "regular",
    ]
    @test isempty(config.modules)
    @test collect(config.env_vars) == [
        "JULIA_MPI_HAS_CUDA" => true
        "CLIMACOMMS_DEVICE" => "CPU"
        "CLIMACOMMS_CONTEXT" => "MPI"
    ]

    # Override default
    config = ClimaCalibrate.PBSConfig(;
        directives = [
            :time => 1,
            :queue => "idk",
            :ntasks => 2,
            :cpus_per_task => 10,
            :gpus_per_task => 42,
            :job_priority => "preempt",
        ],
        modules = [],
        env_vars = [
            "CLIMACOMMS_DEVICE" => "GPU"
            "CLIMACOMMS_CONTEXT" => "SINGLETON"
            "JULIA_MPI_HAS_CUDA" => false
        ],
    )

    @test collect(config.directives) == [
        :time => "00:01:00",
        :queue => "idk",
        :ntasks => 2,
        :cpus_per_task => 10,
        :gpus_per_task => 42,
        :job_priority => "preempt",
    ]
    @test isempty(config.modules)
    @test collect(config.env_vars) == [
        "CLIMACOMMS_DEVICE" => "GPU"
        "CLIMACOMMS_CONTEXT" => "SINGLETON"
        "JULIA_MPI_HAS_CUDA" => false
    ]

    # Warn when directives won't be used
    @test_logs (:warn, r"following directives are not supported") match_mode =
        :any ClimaCalibrate.PBSConfig(;
        directives = [:time => 1, :checkpoint => 10],
    )
end

@testset "Error handling for configs" begin
    # Error handling
    directives = [:time => 1]
    repeated_directives = [:time => 1, :time => 2]
    @test_throws r"Not all directives" ClimaCalibrate.SlurmConfig(;
        directives = repeated_directives,
    )
    @test_throws r"Not all directives" ClimaCalibrate.PBSConfig(;
        directives = repeated_directives,
    )

    repeated_env_vars =
        ["CLIMACOMMS_CONTEXT" => "SINGLETON", "CLIMACOMMS_CONTEXT" => "MPI"]
    @test_throws r"Not all environment" ClimaCalibrate.SlurmConfig(;
        env_vars = repeated_env_vars,
    )
    @test_throws r"Not all environment" ClimaCalibrate.PBSConfig(;
        env_vars = repeated_env_vars,
    )

    repeated_modules = ["climacommon", "climacommon"]
    @test_throws r"Not all modules" ClimaCalibrate.SlurmConfig(;
        modules = repeated_modules,
    )
    @test_throws r"Not all modules" ClimaCalibrate.PBSConfig(;
        modules = repeated_modules,
    )
end

@testset "Check climacommon" begin
    # This is difficult to check when constructing the SlurmConfig and PBSConfig
    # since we need to modify the ENV, so we test it separately
    modules = ["climacommon/2026_02_18"]
    env = Dict("LOADEDMODULES" => "julia/1.12.5:climacommon/2026_02_18")
    @test_nowarn ClimaCalibrate.Backend._check_climacommon(modules; env)

    modules = ["climacommon/2030_02_18"]
    env = Dict("LOADEDMODULES" => "julia/1.12.5:climacommon/2026_02_18")
    @test_warn r"are not the same" ClimaCalibrate.Backend._check_climacommon(
        modules;
        env,
    )

    modules = []
    env = Dict("LOADEDMODULES" => "julia/1.12.5:climacommon/2026_02_18")
    @test_warn r"most likely want to load climacommon in the config" ClimaCalibrate.Backend._check_climacommon(
        modules;
        env,
    )

    modules = ["climacommon"]
    env = Dict("LOADEDMODULES" => "julia/1.12.5:climacommon/2026_02_18")
    @test_nowarn ClimaCalibrate.Backend._check_climacommon(modules; env)

    modules = []
    env = Dict()
    @test_nowarn ClimaCalibrate.Backend._check_climacommon(modules; env)
end

@testset "Generate directives, modules, and environment variables" begin
    slurm_config = ClimaCalibrate.SlurmConfig(;
        directives = [
            :time => 1,
            :queue => "idk",
            :ntasks => 2,
            :cpus_per_task => 10,
            :gpus_per_task => 42,
            :job_priority => "preempt",
        ],
        modules = ["climacommon"],
        env_vars = [
            "CLIMACOMMS_DEVICE" => "GPU"
            "CLIMACOMMS_CONTEXT" => "SINGLETON"
            "JULIA_MPI_HAS_CUDA" => false
        ],
    )

    directives = ClimaCalibrate.Backend.generate_directives(slurm_config)
    modules = ClimaCalibrate.Backend.generate_modules(slurm_config)
    env_vars = ClimaCalibrate.Backend.generate_env_vars(slurm_config)

    @test directives == """#SBATCH --time=00:01:00
       #SBATCH --queue=idk
       #SBATCH --ntasks=2
       #SBATCH --cpus-per-task=10
       #SBATCH --gpus-per-task=42
       #SBATCH --job-priority=preempt"""

    @test modules == "module load climacommon"
    # Remove leading and trailing whitespaces to avoid using \" for quotation
    # marks
    @test env_vars == strip("""
export CLIMACOMMS_DEVICE="GPU"
export CLIMACOMMS_CONTEXT="SINGLETON"
export JULIA_MPI_HAS_CUDA="false"
""")

    pbs_config = ClimaCalibrate.PBSConfig(;
        directives = [
            :time => 1,
            :queue => "idk",
            :ntasks => 2,
            :cpus_per_task => 10,
            :gpus_per_task => 42,
            :job_priority => "preempt",
        ],
        modules = ["climacommon/2024_04_05"],
        env_vars = [
            "CLIMACOMMS_DEVICE" => "GPU"
            "CLIMACOMMS_CONTEXT" => "SINGLETON"
            "JULIA_MPI_HAS_CUDA" => false
        ],
    )

    directives = ClimaCalibrate.Backend.generate_directives(pbs_config)
    modules = ClimaCalibrate.Backend.generate_modules(pbs_config)
    env_vars = ClimaCalibrate.Backend.generate_env_vars(pbs_config)

    @test directives == """#PBS -j oe
       #PBS -A UCIT0011
       #PBS -q idk
       #PBS -l job_priority=preempt
       #PBS -l walltime=00:01:00
       #PBS -l select=2:ncpus=10:ngpus=42:mpiprocs=42"""

    @test modules == "module load climacommon/2024_04_05"
    # Remove leading and trailing whitespaces to avoid having to use \"
    # everywhere
    @test env_vars == strip("""
export CLIMACOMMS_DEVICE="GPU"
export CLIMACOMMS_CONTEXT="SINGLETON"
export JULIA_MPI_HAS_CUDA="false"
""")
end

@testset "Construct backends from configs" begin
    directives =
        [:ntasks => 1, :gpus_per_task => 1, :cpus_per_task => 12, :time => 720]
    modules = ["climacommon"]
    env_vars =
        ["CLIMACOMMS_CONTEXT" => "SINGLETON", "CLIMACOMMS_DEVICE" => "CUDA"]
    pbs_config = ClimaCalibrate.PBSConfig(; directives, modules, env_vars)
    slurm_config = ClimaCalibrate.SlurmConfig(; directives, modules, env_vars)

    function is_equal_backend(x, y)
        typeof(x) === typeof(y) || return false
        return all(
            name -> getfield(x, name) == getfield(y, name),
            fieldnames(typeof(x)),
        )
    end

    derecho_backend =
        ClimaCalibrate.DerechoBackend(; directives, modules, env_vars)
    @test is_equal_backend(derecho_backend.hpc_config, pbs_config)

    clima_backend =
        ClimaCalibrate.ClimaGPUBackend(; directives, modules, env_vars)
    @test is_equal_backend(clima_backend.hpc_config, slurm_config)

    caltech_hpc_backend =
        ClimaCalibrate.CaltechHPCBackend(; directives, modules, env_vars)
    @test is_equal_backend(caltech_hpc_backend.hpc_config, slurm_config)

    gcp_backend = ClimaCalibrate.GCPBackend(; directives, modules, env_vars)
    @test is_equal_backend(gcp_backend.hpc_config, slurm_config)
end

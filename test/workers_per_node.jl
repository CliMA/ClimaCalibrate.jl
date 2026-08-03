using Test
import Distributed
import ClimaCalibrate
import ClimaCalibrate.Backend:
    multi_worker_script,
    shell_quote,
    workers_per_allocation,
    allocation_resource_kwargs,
    parse_pbs_worker_params,
    parse_slurm_worker_params,
    worker_cookie,
    PBSManager,
    SlurmManager

@testset "multi_worker_script" begin
    outputs = ["/tmp/w-1.out", "/tmp/w-2.out", "/tmp/w-3.out"]
    script = multi_worker_script(
        "julia",
        Cmd(["--project=@temp proj", "--threads=2"]),
        outputs,
    )
    lines = split(script, '\n'; keepempty = false)

    @test first(lines) == "#!/bin/bash"
    @test last(lines) == "wait"
    worker_lines = lines[2:(end - 1)]
    @test length(worker_lines) == 3

    for (g, line) in enumerate(worker_lines)
        @test startswith(line, "CUDA_VISIBLE_DEVICES=$(g - 1) ")
        @test endswith(line, "> '$(outputs[g])' 2>&1 &")
        @test occursin("--worker=$(worker_cookie())", line)
        # Arguments with spaces must be shell escaped
        @test occursin("'--project=@temp proj'", line)
        @test occursin("'--threads=2'", line)
    end
end

@testset "shell_quote" begin
    @test shell_quote("--threads=2") == "'--threads=2'"
    @test shell_quote("/my proj") == "'/my proj'"
    @test shell_quote("a\$b;c&d") == "'a\$b;c&d'"
    # An embedded quote ends the quoted run, adds a literal quote, then restarts
    @test shell_quote("it's") == "'it'\\''s'"
end

@testset "workers_per_allocation" begin
    @test workers_per_allocation(8, 4) == [4, 4]
    @test workers_per_allocation(9, 4) == [4, 4, 1]
    @test workers_per_allocation(3, 4) == [3]
    # max_per_node = 1 so workers get one allocation each
    @test workers_per_allocation(3, 1) == [1, 1, 1]
end

@testset "allocation_resource_kwargs" begin
    # PBS requests resources per allocation, so the request scales
    @test allocation_resource_kwargs(PBSManager(8), :gpu, 4) ==
          Dict(:l_select => "ngpus=4:ncpus=16")
    @test allocation_resource_kwargs(PBSManager(8), :cpu, 4) ==
          Dict(:l_select => "ncpus=4")
    # Slurm runs the workers as background tasks in one allocation, so that
    # task needs all the workers' resources
    @test allocation_resource_kwargs(SlurmManager(8), :gpu, 4) ==
          Dict(:gpus_per_task => 4, :cpus_per_task => 16)
    @test allocation_resource_kwargs(SlurmManager(8), :cpu, 4) ==
          Dict(:cpus_per_task => 4)
end

@testset "worker param parsing ignores packing key" begin
    pbs_params = Dict{Symbol, Any}(
        :workers_per_node => 4,
        :l_select => "ngpus=4:ncpus=16",
        :q => "main",
    )
    pbs_args = parse_pbs_worker_params(pbs_params)
    @test !any(a -> occursin("workers", string(a)), pbs_args)
    @test "select=ngpus=4:ncpus=16" in pbs_args

    slurm_params = Dict{Symbol, Any}(
        :workers_per_node => 4,
        :gpus_per_task => 1,
        :cpus_per_task => 4,
    )
    slurm_args = parse_slurm_worker_params(slurm_params)
    @test !any(a -> occursin("workers", string(a)), slurm_args)
    @test "--gpus-per-task=1" in slurm_args
end

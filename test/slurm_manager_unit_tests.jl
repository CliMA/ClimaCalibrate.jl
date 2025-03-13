using Test, ClimaCalibrate, Distributed, Logging

@testset "SlurmManager Unit Tests" begin
    out_file = "my_slurm_job.out"
    initial_expr = quote
        @info "Running worker $(myid())"
        using Test
        using LinearAlgebra
    end
    p = addprocs(SlurmManager(1; expr = initial_expr); o = out_file)

    @test nprocs() == 2
    @test workers() == p
    @test fetch(@spawnat :any myid()) == p[1]
    @test remotecall_fetch(+, p[1], 1, 1) == 2
    # Test that the worker is configured correctly
    @test remotecall_fetch(Base.active_project, p[1]) == Base.active_project()
    @test remotecall_fetch(global_logger, p[1]) isa
          Base.CoreLogging.SimpleLogger
    rmprocs(p)
    @test nprocs() == 1
    @test workers() == [1]
    # Check output file creation
    @test isfile(out_file)
    rm(out_file)

    # Test incorrect generic arguments
    @test_throws TaskFailedException p = addprocs(SlurmManager(1), time = "w")
end

@testset "SlurmManager Initialization Expressions" begin
    p = addprocs(SlurmManager(1; expr = :(@info "test")))
    rmprocs(p)
    test_logger = TestLogger();
    with_logger(test_logger) do
        addprocs(SlurmManager(1; expr = :(w + 2)))
    end
    @test test_logger.logs[end].message == "Initial worker expression errored:"
end


@testset "Test remotecall utilities" begin
    out_file = "pbs_unit_test.out"
    p = addprocs(
        PBSManager(1),
    )
    @test nprocs() == length(p) + 1
    @test workers() == p
    @test remotecall_fetch(+, p[1], 1, 1) == 2

    @everywhere using ClimaCalibrate
    # Test function with no arguments
    p = workers()
    @test ClimaCalibrate.map_remotecall_fetch(myid) == p

    # single argument 
    x = rand(5)
    @test ClimaCalibrate.map_remotecall_fetch(identity, x) == fill(x, length(p))

    # multiple arguments
    @test ClimaCalibrate.map_remotecall_fetch(+, 2, 3) == fill(5, length(p))

    # Test specified workers list
    @test length(ClimaCalibrate.map_remotecall_fetch(myid; workers = p[1:2])) ==
          2

    # Test with more complex data structure
    d = Dict("a" => 1, "b" => 2)
    @test ClimaCalibrate.map_remotecall_fetch(identity, d) == fill(d, length(p))

    loggers = ClimaCalibrate.set_worker_loggers()
    @test length(loggers) == length(p)
    @test typeof(loggers) == Vector{Base.CoreLogging.SimpleLogger}

    rmprocs(p)
    @test nprocs() == 1
    @test workers() == [1]
end

using Test, ClimaCalibrate, Distributed, Logging

@testset "PBSManager Unit Tests" begin
    p = addprocs(
        PBSManager(1),
        q = "main",
        A = "UCIT0011",
        l_select = "ngpus=1",
        l_walltime = "00:05:00",
    )
    @test nprocs() == length(p) + 1
    @test workers() == p
    @test remotecall_fetch(myid, 2) == 2
    @test remotecall_fetch(+, p[1], 1, 1) == 2
    # Test that the worker is configured correctly
    @test remotecall_fetch(pwd, p[1]) == pwd()
    @test remotecall_fetch(Base.active_project, p[1]) == Base.active_project()
    @test remotecall_fetch(global_logger, p[1]) isa
          Base.CoreLogging.SimpleLogger

    rmprocs(p)
    @test nprocs() == 1
    @test workers() == [1]
end

@testset "PBSManager - multiple processes" begin
    out_file = "pbs_unit_test.out"
    p = addprocs(
        PBSManager(2),
        o = out_file,
        q = "main",
        A = "UCIT0011",
        l_select = "ngpus=1",
        l_walltime = "00:05:00",
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
    @test length(
        ClimaCalibrate.map_remotecall_fetch(myid; workers = workers()[1:2]),
    ) == 2

    # Test with more complex data structure
    d = Dict("a" => 1, "b" => 2)
    @test ClimaCalibrate.map_remotecall_fetch(identity, d) == fill(d, length(p))

    loggers = ClimaCalibrate.set_worker_loggers()
    @test length(loggers) == length(p)
    @test typeof(loggers) == Vector{Base.CoreLogging.SimpleLogger}

    rmprocs(p)
    @test nprocs() == 1
    @test workers() == [1]

    @test isfile(out_file)
    rm(out_file)
end

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

    # Test broken arguments
    @test_throws TaskFailedException p = addprocs(PBSManager(1), time = "w")
end

@testset "Test PBSManager multiple tasks, output file" begin
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

    @test isfile(out_file)
    rm(out_file)
end

using Test, ClimaCalibrate, Distributed, Logging

@testset "SlurmManager Unit Tests" begin
    @test ClimaCalibrate.get_manager() == SlurmManager(1)
    out_file = tempname()
    p = add_workers(1; device = :cpu, o = out_file, time = 5)
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

    # Test incorrect generic arguments
    @test_throws TaskFailedException p = addprocs(SlurmManager(1), time = "w")
end

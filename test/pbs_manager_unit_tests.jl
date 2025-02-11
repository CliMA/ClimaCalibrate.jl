using Test, ClimaCalibrate, Distributed

@testset "PBSManager Unit Tests" begin
    p = addprocs(
        PBSManager(1),
        q = "main",
        A = "UCIT0011",
        l_select = "ngpus=1",
        l_walltime = "00:05:00",
    )
    @test nprocs() == 2
    @test workers() == p
    @test fetch(@spawnat :any myid()) == p[1]
    @test remotecall_fetch(+, p[1], 1, 1) == 2
    rmprocs(p)
    @test nprocs() == 1
    @test workers() == [1]
end

@testset "PBSManager Output file" begin
    out_file = "pbs_unit_test.out"
    p = addprocs(
        PBSManager(2),
        o = out_file,
        q = "main",
        A = "UCIT0011",
        l_select = "ngpus=1",
        l_walltime = "00:05:00",
    )
    @test nprocs() == 2
    @test workers() == p
    @test fetch(@spawnat :any myid()) == p[1]
    @test remotecall_fetch(+, p[1], 1, 1) == 2
    rmprocs(p)
    @test nprocs() == 1
    @test workers() == [1]

    @test isfile(out_file)
    rm(out_file)
end

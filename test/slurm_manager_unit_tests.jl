using Test, Distributed, Logging
import ClimaCalibrate
# Blocking version of add_workers for test
function add_workers_and_wait(n; timeout = 1200, kwargs...)
    ids = fetch(ClimaCalibrate.add_workers(n; kwargs...))
    pool = ClimaCalibrate.Backend.GLOBAL_WORKER_POOL
    tstart = time()
    while !all(id -> id in pool.workers, ids)
        (time() - tstart) > timeout &&
            error("Workers did not initialize within $(timeout)s")
        sleep(0.5)
    end
    return ids
end

@testset "SlurmManager Unit Tests" begin
    @test ClimaCalibrate.get_manager() == ClimaCalibrate.SlurmManager(1)
    out_file = tempname()
    p = add_workers_and_wait(1; device = :cpu, o = out_file, time = 5)
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
    # Each worker is submitted individually and writes to `<o>-<i>.out`
    @test isfile("$out_file-1.out")

    # Test incorrect generic arguments
    @test_throws TaskFailedException p =
        addprocs(ClimaCalibrate.SlurmManager(1), time = "w")
end

@testset "SlurmManager - two workers per node" begin
    # Two workers in one allocation must land on the same node and join the
    # pool like individually-submitted workers
    out_file = tempname()
    kwargs = (; device = :cpu, o = out_file, time = 5, workers_per_node = 2)
    p = add_workers_and_wait(2; kwargs...)
    @test workers() == p
    @test length(unique(map(w -> remotecall_fetch(gethostname, w), p))) == 1
    @test ClimaCalibrate.map_remotecall_fetch(myid) == p
    rmprocs(p)
    @test nprocs() == 1
    # Each worker writes to `<o>-<job>-<worker>.out`
    for g in 1:2
        @test isfile("$out_file-1-$g.out")
        rm("$out_file-1-$g.out")
    end
end

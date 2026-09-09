using Test
using Distributed
import ClimaCalibrate

# Initialization runs in the background, so workers join the pool over the next
# few seconds
function wait_for_pool(pool, ids; timeout = 300)
    t0 = time()
    while !issubset(ids, pool.workers) && time() - t0 < timeout
        sleep(1)
    end
    return nothing
end

# A worker enters `workers()` when `Distributed` registers it, which is before
# it has loaded any code. The invariant is that `calibration_worker_pool` only
# ever hands out a worker that has been through `initialize_worker`, since a
# member dispatched to a worker without `ClimaCalibrate` dies on the first
# thing it is sent.
@testset "A pooled worker has the code to run a forward model" begin
    project = dirname(Base.active_project())
    ids = addprocs(2; exeflags = "--project=$project")
    try
        ClimaCalibrate.@worker_setup const WORKER_POOL_PROBE = 1

        pool = ClimaCalibrate.calibration_worker_pool()
        wait_for_pool(pool, ids)
        @test issubset(ids, pool.workers)
        @test ClimaCalibrate.Backend.n_initializing_workers() == 0

        # `isdefined` rather than a closure: this test runs in a module of its
        # own, which a worker cannot deserialize
        for id in pool.workers
            @test remotecall_fetch(isdefined, id, Main, :ClimaCalibrate)
            @test remotecall_fetch(isdefined, id, Main, :WORKER_POOL_PROBE)
        end

        # A pooled worker is claimed by neither path a second time. Replaying
        # the setup expressions would redefine `WORKER_POOL_PROBE`
        @test !ClimaCalibrate.Backend._claim_worker(first(ids))
        ClimaCalibrate.calibration_worker_pool()
        @test ClimaCalibrate.Backend.n_initializing_workers() == 0
    finally
        rmprocs(ids)
    end
end

# A worker that exits while running a member is deregistered by `Distributed`,
# which drops it from the pool. The invariant is that `run_iteration` does not
# put it back: `take!(::WorkerPool)` throws once the dead id it prunes was the
# pool's last, so a calibration started on a pool holding one aborted instead
# of waiting for the workers still joining.
@testset "A worker that dies running a member stays out of the pool" begin
    project = dirname(Base.active_project())
    # The setup expressions run in `Main`, which under `Pkg.test` has not
    # imported the package
    ClimaCalibrate.@worker_setup import ClimaCalibrate
    ClimaCalibrate.@worker_setup struct ExitingModelInterface <:
                                        ClimaCalibrate.AbstractModelInterface end
    # Every member of iteration 1 takes its worker down; iteration 2 succeeds
    ClimaCalibrate.@worker_setup ClimaCalibrate.forward_model(
        ::ExitingModelInterface,
        iter,
        member,
    ) = iter == 1 && exit()
    interface = Main.ExitingModelInterface()
    backend = ClimaCalibrate.WorkerBackend(;
        failure_rate = 1.0,
        empty_pool_timeout = 300,
    )
    pool = backend.worker_pool
    output_dir = mktempdir()

    doomed = addprocs(1; exeflags = "--project=$project")
    ClimaCalibrate.calibration_worker_pool()
    wait_for_pool(pool, doomed)
    @test issubset(doomed, pool.workers)

    ClimaCalibrate.Calibration.run_iteration(
        backend,
        interface,
        1,
        1,
        output_dir,
    )
    @test isempty(intersect(doomed, procs()))
    @test !ClimaCalibrate.model_completed(output_dir, 1, 1)
    @test ClimaCalibrate.Calibration.take_live_worker!(pool) === nothing

    # Dispatched while the pool holds nothing but the dead worker, as when the
    # rest of an allocation's workers are still starting
    second = @async ClimaCalibrate.Calibration.run_iteration(
        backend,
        interface,
        2,
        1,
        output_dir,
    )
    fresh = addprocs(1; exeflags = "--project=$project")
    try
        ClimaCalibrate.calibration_worker_pool()
        @test timedwait(() -> istaskdone(second), 300) == :ok
        @test !istaskfailed(second)
        @test ClimaCalibrate.model_completed(output_dir, 2, 1)
    finally
        rmprocs(fresh)
    end
end

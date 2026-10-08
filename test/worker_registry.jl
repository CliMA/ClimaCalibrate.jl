using Test
using Distributed
import TOML
import ClimaCalibrate
import ClimaCalibrate.Backend
import ClimaCalibrate.Backend.Workers
import ClimaCalibrate.Backend.Workers:
    WorkerRegistry,
    insert_group!,
    insert_worker!,
    set_job_id!,
    set_pid!,
    set_state!,
    worker_row,
    worker_rows,
    group_row,
    reconcile_workers!,
    STARTING,
    READY,
    BUSY,
    ENDED

# Every test builds its own registry so the fakes below are the only writers.
open_registry() = WorkerRegistry(joinpath(mktempdir(), "workers.toml"))

# Stands in for a scheduler. `states(ids)` answers `query_scheduler_states`,
# where `nothing` is a failed query. Queries and cancellations are recorded.
Base.@kwdef mutable struct FakeManager
    states::Any = ids -> nothing
    queries::Vector{Any} = []
    cancelled::Vector{Any} = []
end
Workers.query_scheduler_states(m::FakeManager, ids) =
    (push!(m.queries, ids); m.states(ids))
Workers.cancel_scheduler_jobs(m::FakeManager, ids) =
    (push!(m.cancelled, ids); nothing)

# A group this session can launch for through `manager`, with `n` rows
# submitted at time 0.
function scheduler_group(
    reg;
    desired,
    max_relaunches,
    manager = FakeManager(),
    startup_timeout = 1e6,
    n = 0,
)
    launcher = (; manager, kwargs = (;))
    gid = insert_group!(
        reg;
        cluster = :slurm,
        desired,
        max_relaunches,
        startup_timeout,
        launcher,
    )
    ids = [insert_worker!(reg; group_id = gid, now = 0.0) for _ in 1:n]
    foreach(i -> set_job_id!(reg, ids[i], "j$i"; now = 0.0), 1:n)
    return gid, ids
end

# Reconcile with every external effect replaced. Callers space `now` by more
# than RECONCILE_INTERVAL.
reconcile(reg; now, alive = Int[], submit = (g, n) -> nothing) =
    reconcile_workers!(reg; now, alive_pids = alive, submit)

@testset "Schema and transitions" begin
    reg = open_registry()
    t0 = 1000.0
    gid = insert_group!(
        reg;
        cluster = :slurm,
        desired = 2,
        max_relaunches = 3,
        startup_timeout = 60.0,
        now = t0,
    )
    id = insert_worker!(reg; group_id = gid, now = t0)
    @test worker_row(reg, id).state == STARTING

    @test set_state!(reg, id, READY; now = t0 + 1)
    @test worker_row(reg, id).ready_at == t0 + 1
    @test set_state!(reg, id, BUSY)
    @test set_state!(reg, id, ENDED; reason = "done", now = t0 + 2)
    row = worker_row(reg, id)
    @test row.state == ENDED
    @test row.ended_at == t0 + 2
    @test row.exit_reason == "done"
    # A worker that never made the pool ends with no ready_at
    never = insert_worker!(reg; group_id = gid, now = t0)
    @test set_state!(reg, never, ENDED; reason = "lost")
    @test isnothing(worker_row(reg, never).ready_at)
    # Ended is never left
    @test !set_state!(reg, id, READY)
    @test !set_state!(reg, id, ENDED; reason = "again")
    @test worker_row(reg, id).exit_reason == "done"
    # Unknown rows
    @test_throws BoundsError set_state!(reg, 999, READY)

    @test length(worker_rows(reg; states = (ENDED,))) == 2
    @test isempty(worker_rows(reg; states = (READY,)))
    @test length(worker_rows(reg; group_id = gid)) == 2

    # The file holds the same records, without the in-memory launcher
    Workers.write_registry_file(reg)
    file = TOML.parsefile(reg.path)
    @test length(file["groups"]) == 1
    @test file["groups"][1]["cluster"] == "slurm"
    @test !haskey(file["groups"][1], "launcher")
    @test length(file["workers"]) == 2
    @test file["workers"][1]["state"] == "ended"
    @test file["workers"][1]["exit_reason"] == "done"
    @test !haskey(file["workers"][2], "ready_at")

    # The path is fixed once the registry has groups
    @test_throws ErrorException Workers.set_registry_path!("other.toml", reg)
end

@testset "cancel_worker_jobs" begin
    reg = open_registry()
    m = FakeManager()
    gid, ids = scheduler_group(
        reg;
        desired = 3,
        max_relaunches = 0,
        manager = m,
        n = 3,
    )
    a, b, c = ids
    set_job_id!(reg, b, "shared")
    set_job_id!(reg, c, "shared")
    set_state!(reg, c, READY)
    # A group without a launcher is not this session's to cancel
    foreign = insert_group!(
        reg;
        cluster = :slurm,
        desired = 1,
        max_relaunches = 0,
        startup_timeout = 1e6,
    )
    f = insert_worker!(reg; group_id = foreign)
    set_job_id!(reg, f, "foreign")

    ClimaCalibrate.cancel_worker_jobs(; reg)
    @test length(m.cancelled) == 1
    @test sort(m.cancelled[1]) == ["j1", "shared"]
    for id in ids
        @test worker_row(reg, id).state == ENDED
        @test worker_row(reg, id).exit_reason == "cancel_worker_jobs"
    end
    @test worker_row(reg, f).state == STARTING
    @test !Workers.owned(group_row(reg, gid))
end

@testset "Reconcile: scheduler states" begin
    reg = open_registry()
    # j1 queued, j2 running, j3 failed, j4 and j5 gone from the scheduler
    m = FakeManager(;
        states = ids -> Dict(
            "j1" => Backend.PENDING,
            "j2" => Backend.RUNNING,
            "j3" => Backend.FAILED,
        ),
    )
    gid, ids = scheduler_group(
        reg;
        desired = 0,
        max_relaunches = 0,
        manager = m,
        n = 5,
    )
    set_state!(reg, ids[3], READY)
    set_pid!(reg, ids[3], 3)
    set_state!(reg, ids[5], READY)
    set_pid!(reg, ids[5], 5)
    # A job submitted moments ago that the scheduler does not list yet
    young = insert_worker!(reg; group_id = gid, now = 60.0)
    set_job_id!(reg, young, "young"; now = 60.0)

    # Worker 5 is still connected, so its missing job is not believed yet
    reconcile(reg; now = 100.0, alive = [5])
    @test length(m.queries) == 1
    @test sort(m.queries[1]) == ["j1", "j2", "j3", "j4", "j5", "young"]
    @test worker_row(reg, ids[1]).state == STARTING
    @test worker_row(reg, ids[2]).state == STARTING
    @test worker_row(reg, ids[3]).state == ENDED
    @test worker_row(reg, ids[3]).exit_reason == "scheduler reported FAILED"
    @test worker_row(reg, ids[4]).state == ENDED
    @test worker_row(reg, ids[4]).exit_reason == "job left the queue"
    @test worker_row(reg, ids[5]).state == READY
    @test worker_row(reg, young).state == STARTING

    # Rate limited: a second call within the interval does not query
    reconcile(reg; now = 115.0, alive = [5])
    @test length(m.queries) == 1
    reconcile(reg; now = 130.0, alive = Int[])
    @test length(m.queries) == 2
    # Only live jobs are asked about, the grace has passed, and worker 5 is
    # gone from Distributed too
    @test sort(m.queries[2]) == ["j1", "j2", "j5", "young"]
    @test worker_row(reg, young).state == ENDED
    @test worker_row(reg, ids[5]).state == ENDED
end

@testset "Reconcile: local workers, startup timeout, cancellation" begin
    reg = open_registry()
    # A local worker whose process is gone
    lgid = insert_group!(
        reg;
        cluster = :local,
        desired = 0,
        max_relaunches = 0,
        startup_timeout = 1e6,
        launcher = (; manager = nothing, kwargs = (;)),
    )
    local_id = insert_worker!(reg; group_id = lgid)
    set_pid!(reg, local_id, 42)
    # Two workers sharing one allocation, one already connected, and a
    # lone queued worker
    m = FakeManager()
    gid, ids = scheduler_group(
        reg;
        desired = 0,
        max_relaunches = 0,
        manager = m,
        startup_timeout = 50.0,
        n = 3,
    )
    a, b, c = ids
    set_job_id!(reg, a, "shared")
    set_job_id!(reg, b, "shared")
    set_state!(reg, b, READY)
    set_pid!(reg, b, 43)
    set_job_id!(reg, c, "lone")

    # The scheduler query fails, so only the local check and the timeouts act
    reconcile(reg; now = 51.0, alive = [1, 43])
    @test worker_row(reg, local_id).state == ENDED
    @test worker_row(reg, local_id).exit_reason == "disconnected"
    # `shared` is still live through `b`, so only `lone` is cancelled
    @test worker_row(reg, a).state == ENDED
    @test occursin("not ready after 51s", worker_row(reg, a).exit_reason)
    @test worker_row(reg, b).state == READY
    @test worker_row(reg, c).state == ENDED
    @test m.cancelled == [["lone"]]
end

@testset "Reconcile: relaunch budget, rate limit, desired count" begin
    reg = open_registry()
    m = FakeManager()
    gid, _ = scheduler_group(
        reg;
        desired = 3,
        max_relaunches = 2,
        manager = m,
        n = 3,
    )
    launched = []
    # Like `launch_group!`, insert the rows before returning
    submit = (g, n) -> begin
        push!(launched, (g, n))
        for _ in 1:n
            id = insert_worker!(reg; group_id = g, now = 0.0)
            set_job_id!(reg, id, "new-$id"; now = 0.0)
        end
    end

    # Nothing to do at full strength
    reconcile(reg; now = 0.0, submit)
    @test isempty(launched)

    # Two die: both replaced, budget exhausted
    m.states = ids -> Dict("j3" => Backend.RUNNING)
    reconcile(reg; now = 100.0, submit)
    @test launched == [(gid, 2)]
    @test group_row(reg, gid).relaunches_used == 2
    @test length(
        worker_rows(reg; states = Workers.LIVE_STATES, group_id = gid),
    ) == 3

    # A third death cannot be replaced
    m.states = ids -> Dict{String, Any}()
    @test_logs (:warn, r"used all 2 relaunches") match_mode = :any reconcile(
        reg;
        now = 200.0,
        submit,
    )
    @test launched == [(gid, 2)]

    # Calls inside RECONCILE_INTERVAL are skipped
    reconcile(reg; now = 201.0, submit)
    @test launched == [(gid, 2)]

    # Raising the budget and the desired count launches the difference. The
    # scheduler query fails from here on, so the new rows stay live
    m.states = ids -> nothing
    group_row(reg, gid).max_relaunches = 10
    group_row(reg, gid).desired = 5
    reconcile(reg; now = 300.0, submit)
    @test last(launched) == (gid, 5)
    @test group_row(reg, gid).relaunches_used == 7

    # A group without a launcher is left alone
    orphan = insert_group!(
        reg;
        cluster = :slurm,
        desired = 2,
        max_relaunches = 2,
        startup_timeout = 1e6,
    )
    n_before = length(launched)
    reconcile(reg; now = 400.0, submit)
    @test length(launched) == n_before
    @test group_row(reg, orphan).relaunches_used == 0
end

# Initialization runs in the background, so rows reach `READY` over the next
# few seconds
function wait_for_rows(reg, state, n; timeout = 300)
    t0 = time()
    while length(worker_rows(reg; states = (state,))) < n &&
          time() - t0 < timeout
        sleep(0.5)
    end
    return length(worker_rows(reg; states = (state,)))
end

@testset "Local workers end to end" begin
    project = dirname(Base.active_project())
    exeflags = "--project=$project"
    ClimaCalibrate.@worker_setup import ClimaCalibrate
    ClimaCalibrate.@worker_setup struct RegistryExitingInterface <:
                                        ClimaCalibrate.AbstractModelInterface end
    ClimaCalibrate.@worker_setup ClimaCalibrate.forward_model(
        ::RegistryExitingInterface,
        iter,
        member,
    ) = iter == 1 && exit()
    interface = Main.RegistryExitingInterface()
    backend = ClimaCalibrate.WorkerBackend(;
        failure_rate = 1.0,
        empty_pool_timeout = 300,
    )
    output_dir = mktempdir()
    registry_path = joinpath(mktempdir(), "workers.toml")
    reg = Workers.set_registry_path!(registry_path)

    ids = fetch(
        ClimaCalibrate.add_workers(
            1;
            cluster = :local,
            exeflags,
            max_relaunches = 1,
            startup_timeout = 300,
        ),
    )
    @test Workers.worker_registry() === reg
    try
        @test length(ids) == 1
        @test wait_for_rows(reg, READY, 1) == 1
        row = only(worker_rows(reg; states = (READY,)))
        @test row.pid == only(ids)
        g = group_row(reg, row.group_id)
        @test g.desired == 1
        @test g.cluster == :local

        # The member takes its worker down and the row records the exit
        ClimaCalibrate.Calibration.run_iteration(
            backend,
            interface,
            1,
            1,
            output_dir,
        )
        @test isempty(intersect(ids, procs()))
        @test worker_row(reg, row.id).state == ENDED
        @test !isnothing(worker_row(reg, row.id).ready_at)
        @test isempty(worker_rows(reg; states = Workers.LIVE_STATES))

        # Iteration 2 starts with an empty pool. Its dispatch loop launches a
        # replacement, runs the member on it, and returns it to the pool
        ClimaCalibrate.Calibration.run_iteration(
            backend,
            interface,
            2,
            1,
            output_dir,
        )
        @test ClimaCalibrate.model_completed(output_dir, 2, 1)
        fresh = only(worker_rows(reg; states = (READY,)))
        @test fresh.id != row.id
        @test fresh.pid in workers()
        @test group_row(reg, g.id).relaunches_used == 1
        @test length(worker_rows(reg)) == 2

        # Teardown ends live rows, forgets the launchers, and keeps the file
        ClimaCalibrate.cancel_worker_jobs()
        @test worker_row(reg, fresh.id).state == ENDED
        @test worker_row(reg, fresh.id).exit_reason == "cancel_worker_jobs"
        @test isnothing(group_row(reg, g.id).launcher)
        @test isempty(worker_rows(reg; states = Workers.LIVE_STATES))
        @test isfile(registry_path)
    finally
        rmprocs(workers())
    end
end

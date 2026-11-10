using Test
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
import Random
import ClimaCalibrate as CAL
import ClimaCalibrate.SharedFiles: LocalFiles
import ClimaCalibrate.TaskBoard as TB

# The model lives in its own file so the worker processes `start_worker`
# launches can include it too
const MODEL_FILE = joinpath(@__DIR__, "shared_files_model.jl")
Base.include(Main, MODEL_FILE)
import Main: FileModel, FILE_MODEL_OUTPUT

prior = constrained_gaussian("test_param", 10, 5, 0, Inf)
observations = [20.0]
noise = [0.01;;]

function make_ekp(ensemble_size)
    rng = Random.MersenneTwister(1234)
    return EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng, prior, ensemble_size),
        observations,
        noise,
        EKP.Inversion();
        rng,
    )
end

# A worker in its own Julia process, logging to <output_dir>/<worker_id>.log.
function start_worker(
    output_dir,
    ensemble_size;
    worker_id,
    delay = 0.0,
    idle_timeout = 60,
)
    code = """
    include($(repr(MODEL_FILE)))
    import ClimaCalibrate
    ClimaCalibrate.run_worker(
        FileModel($(repr(output_dir)), $ensemble_size, $delay),
        $(repr(output_dir));
        worker_id = $(repr(worker_id)),
        idle_timeout = $idle_timeout,
        poll_interval = 0.2,
    )
    """
    log = joinpath(output_dir, "$worker_id.log")
    project = Base.active_project()
    julia = Base.julia_cmd()
    cmd = `$julia --project=$project --startup-file=no --threads=2 -e $code`
    return run(pipeline(cmd; stdout = log, stderr = log); wait = false)
end

function stop(proc)
    process_running(proc) && kill(proc)
    wait(proc)
    return nothing
end

post_iteration!(files, iteration, ensemble_size; lease = 10.0) = begin
    foreach(m -> TB.publish!(files, iteration, m), 1:ensemble_size)
    TB.set_current!(files, iteration, ensemble_size; lease)
end

@testset "run_worker runs what is posted" begin
    output_dir = mktempdir()
    n = 3
    CAL.Calibration.initialize(make_ekp(n), prior, output_dir)
    files = LocalFiles(output_dir)
    post_iteration!(files, 1, n)

    ran = CAL.run_worker(
        FileModel(output_dir, n, 0.0),
        output_dir;
        worker_id = "inproc",
        idle_timeout = 0.5,
        poll_interval = 0.1,
    )
    @test ran == n
    @test TB.summary(files, 1, n).done == 1:n
    for m in 1:n
        member_path = CAL.path_to_ensemble_member(output_dir, 1, m)
        @test isfile(joinpath(member_path, FILE_MODEL_OUTPUT))
        done = TB.read_toml(files, TB.member_key(1, m, "done.toml"))
        @test done["worker_id"] == "inproc"
    end
end

@testset "a worker with nothing posted exits after idle_timeout" begin
    output_dir = mktempdir()
    t = @elapsed ran = CAL.run_worker(
        FileModel(output_dir, 1, 0.0),
        output_dir;
        idle_timeout = 0.3,
        poll_interval = 0.1,
    )
    @test ran == 0
    @test t >= 0.3
end

struct FailingModel <: CAL.AbstractModelInterface end
CAL.forward_model(::FailingModel, iteration, member) =
    error("member $member exploded")

@testset "a model failure is recorded and not retried" begin
    output_dir = mktempdir()
    n = 2
    files = LocalFiles(output_dir)
    post_iteration!(files, 1, n)
    ran = CAL.run_worker(
        FailingModel(),
        output_dir;
        worker_id = "inproc",
        idle_timeout = 0.5,
        poll_interval = 0.1,
    )
    @test ran == n
    @test TB.summary(files, 1, n).failed == 1:n
    failed = TB.read_toml(files, TB.member_key(1, 1, "failed.toml"))
    @test occursin("member 1 exploded", failed["error"])
end

# The driver in a task and one worker in this process. The driver's result
# or error comes back through `fetch`
function drive_with_worker(backend, interface, n, output_dir; worker_id)
    driver = @async CAL.Calibration.run_iteration(
        backend,
        interface,
        1,
        n,
        output_dir,
    )
    CAL.run_worker(
        interface,
        output_dir;
        worker_id,
        idle_timeout = 1.0,
        poll_interval = 0.1,
    )
    return fetch(driver)
end

@testset "the driver halts on failures and a rerun retries them" begin
    output_dir = mktempdir()
    n = 2
    files = LocalFiles(output_dir)
    backend = CAL.SharedFilesBackend(; poll_interval = 0.1)
    @test_throws TaskFailedException drive_with_worker(
        backend,
        FailingModel(),
        n,
        output_dir;
        worker_id = "first",
    )
    @test TB.summary(files, 1, n).failed == 1:n

    tolerant = CAL.SharedFilesBackend(; poll_interval = 0.1, failure_rate = 1.0)
    @test isnothing(
        drive_with_worker(
            tolerant,
            FailingModel(),
            n,
            output_dir;
            worker_id = "second",
        ),
    )
    @test TB.summary(files, 1, n).failed == 1:n
    failed = TB.read_toml(files, TB.member_key(1, 1, "failed.toml"))
    @test failed["worker_id"] == "second"
end

@testset "calibrate over worker processes" begin
    output_dir = mktempdir()
    n = 4
    workers = [start_worker(output_dir, n; worker_id = "w$i") for i in 1:2]
    try
        backend = CAL.SharedFilesBackend(; poll_interval = 0.2, lease = 10.0)
        ekp = CAL.calibrate(
            backend,
            make_ekp(n),
            FileModel(output_dir, n, 0.0),
            2,
            prior,
            output_dir,
        )
        @test CAL.last_completed_iteration(output_dir) == 2
        @test EKP.get_N_iterations(ekp) == 2
        files = LocalFiles(output_dir)
        for iter in 1:2
            @test TB.summary(files, iter, n).done == 1:n
            owners = [
                TB.read_toml(files, TB.member_key(iter, m, "done.toml"))["worker_id"]
                for m in 1:n
            ]
            @test all(in(("w1", "w2")), owners)
            @test all(m -> CAL.model_completed(output_dir, iter, m), 1:n)
        end
        # The model echoes its parameter, so the update moves toward the data
        initial = abs(EKP.get_ϕ_mean(prior, ekp, 1)[1] - only(observations))
        final = abs(EKP.get_ϕ_mean_final(prior, ekp)[1] - only(observations))
        @test final < initial
    finally
        foreach(stop, workers)
    end
end

@testset "an iteration with no workers halts after wait_timeout" begin
    output_dir = mktempdir()
    n = 2
    CAL.Calibration.initialize(make_ekp(n), prior, output_dir)
    backend = CAL.SharedFilesBackend(; poll_interval = 0.05, wait_timeout = 0.3)
    t = @elapsed @test_throws ErrorException CAL.Calibration.run_iteration(
        backend,
        FileModel(output_dir, n, 0.0),
        1,
        n,
        output_dir,
    )
    @test t >= 0.3
end

@testset "a dead worker's member is handed to another" begin
    output_dir = mktempdir()
    n = 2
    files = LocalFiles(output_dir)
    CAL.Calibration.initialize(make_ekp(n), prior, output_dir)

    slow = start_worker(output_dir, n; worker_id = "slow", delay = 600.0)
    backend = CAL.SharedFilesBackend(; poll_interval = 0.2, lease = 2.0)
    driver = @async CAL.Calibration.run_iteration(
        backend,
        FileModel(output_dir, n, 0.0),
        1,
        n,
        output_dir,
    )
    claimed() = TB.summary(files, 1, n).claimed
    @test timedwait(() -> !isempty(claimed()), 180.0) == :ok
    victim = only(claimed())
    stop(slow)

    fast = start_worker(output_dir, n; worker_id = "fast")
    try
        wait(driver)
        @test TB.summary(files, 1, n).done == 1:n
        done = TB.read_toml(files, TB.member_key(1, victim, "done.toml"))
        @test done["worker_id"] == "fast"
        @test done["attempt"] == 2
        @test TB.attempt(files, 1, victim) == 2
    finally
        stop(fast)
    end
end

@testset "a restarted driver picks up from the markers" begin
    output_dir = mktempdir()
    n = 2
    files = LocalFiles(output_dir)
    ekp = make_ekp(n)
    CAL.Calibration.initialize(ekp, prior, output_dir)
    post_iteration!(files, 1, n)

    # The iteration finishes while no driver is around
    CAL.run_worker(
        FileModel(output_dir, n, 0.0),
        output_dir;
        worker_id = "w",
        idle_timeout = 0.5,
        poll_interval = 0.1,
    )
    @test TB.summary(files, 1, n).done == 1:n
    @test CAL.last_completed_iteration(output_dir) == 0

    # The restarted driver has nothing to wait for and moves on to the update
    backend = CAL.SharedFilesBackend(; poll_interval = 0.1)
    CAL.calibrate(
        backend,
        ekp,
        FileModel(output_dir, n, 0.0),
        1,
        prior,
        output_dir,
    )
    @test CAL.last_completed_iteration(output_dir) == 1
end

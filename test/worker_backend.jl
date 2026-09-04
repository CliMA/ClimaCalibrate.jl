import ClimaCalibrate
using Distributed
import Random
import EnsembleKalmanProcesses as EKP

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)

nprocs = 3
# Submit workers asynchronously: each worker is an individual allocation that
# adds itself to the global pool once started. The calibration begins with an
# empty pool and picks up workers as they join
if nworkers() == 1
    if ClimaCalibrate.backend_type() == ClimaCalibrate.DerechoBackend
        ClimaCalibrate.add_workers(
            nprocs;
            cluster = :pbs,
            q = "develop@desched1",
            A = "UCIT0011",
            l_select = "1:ncpus=1:ngpus=1",
            l_walltime = "00:30:00",
        )
    else
        ClimaCalibrate.add_workers(nprocs; cluster = :slurm, device = :cpu)
    end
end

# Use `@worker_setup` (not `@everywhere`) so workers that join later are
# initialized with the model code before they run the forward model
ClimaCalibrate.@worker_setup using ClimaCalibrate
ClimaCalibrate.@worker_setup struct CancelModelInterface <:
                                    ClimaCalibrate.AbstractModelInterface end
ClimaCalibrate.@worker_setup ClimaCalibrate.forward_model(
    ::CancelModelInterface,
    i,
    m,
) = m == 1 && exit()

# Each phase below writes to a directory of its own, created fresh. Reusing one
# directory meant deleting and recreating it under the processes the previous
# phase still had open: on a cluster filesystem `initialize` then read a
# truncated `eki_file.jld2` out of the directory it had removed a moment
# earlier. `cleanup = false` keeps the directories for the CI artifact upload
mkpath(output_dir)
interrupted_dir =
    mktempdir(output_dir; prefix = "interrupted_", cleanup = false)
calibration_dir =
    mktempdir(output_dir; prefix = "calibration_", cleanup = false)

eki = make_ekp(prior, observation, variance; verbose = true)

ClimaCalibrate.initialize(eki, prior, interrupted_dir)

ClimaCalibrate.Calibration.run_iteration(
    ClimaCalibrate.WorkerBackend(),
    CancelModelInterface(),
    1,
    ensemble_size,
    interrupted_dir,
)

# Member 1 exits, which takes down the worker running it along with the other
# members that worker had in flight. How many that is depends on the machine, so
# the test is that each member is left with a checkpoint a restart can read, and
# that the member which exited is not marked complete.
@testset "Test model checkpoints with interruptions" begin
    @test ClimaCalibrate.model_started(interrupted_dir, 1, 1)
    for m in 1:ensemble_size
        @test ClimaCalibrate.model_started(interrupted_dir, 1, m) ||
              ClimaCalibrate.model_completed(interrupted_dir, 1, m)
        rm(ClimaCalibrate.checkpoint_path(interrupted_dir, 1, m))
    end
end

ClimaCalibrate.@worker_setup include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "model_interface.jl",
    ),
)

ekp = make_ekp(prior, observation, variance)
eki = ClimaCalibrate.Calibration.calibrate(
    ClimaCalibrate.WorkerBackend(),
    ekp,
    SurfaceFluxModelInterface(calibration_dir, ensemble_size),
    n_iterations,
    prior,
    calibration_dir,
)

@test ClimaCalibrate.last_completed_iteration(calibration_dir) == n_iterations

test_sf_calibration_output(eki, prior, observation, variance)

theta_star_vec = (; coefficient_a_m_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger"],
    calibration_dir,
)

g_vs_iter_plot(eki, calibration_dir)

@testset "Restarts" begin
    last_iter = ClimaCalibrate.last_completed_iteration(calibration_dir)
    @test last_iter == n_iterations
    ClimaCalibrate.Calibration.run_iteration(
        ClimaCalibrate.WorkerBackend(),
        SurfaceFluxModelInterface(calibration_dir, ensemble_size),
        last_iter + 1,
        ensemble_size,
        calibration_dir,
    )
    G_ensemble = ClimaCalibrate.observation_map(
        SurfaceFluxModelInterface(calibration_dir, ensemble_size),
        last_iter + 1,
    )
    ClimaCalibrate.save_G_ensemble(calibration_dir, last_iter + 1, G_ensemble)
    ClimaCalibrate.update_ensemble(calibration_dir, last_iter + 1, prior)

    @test ClimaCalibrate.last_completed_iteration(calibration_dir) ==
          n_iterations + 1
end

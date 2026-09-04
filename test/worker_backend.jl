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

eki = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, ensemble_size),
    observation,
    variance,
    EKP.Inversion(),
    verbose = true,
)

ClimaCalibrate.initialize(eki, prior, output_dir)

ClimaCalibrate.Calibration.run_iteration(
    ClimaCalibrate.WorkerBackend(),
    CancelModelInterface(),
    1,
    ensemble_size,
    output_dir,
)

# Member 1 exits, which takes down the worker running it along with the other
# members that worker had in flight. How many that is depends on the machine, so
# the test is that each member is left with a checkpoint a restart can read, and
# that the member which exited is not marked complete.
@testset "Test model checkpoints with interruptions" begin
    @test ClimaCalibrate.model_started(output_dir, 1, 1)
    for m in 1:ensemble_size
        @test ClimaCalibrate.model_started(output_dir, 1, m) ||
              ClimaCalibrate.model_completed(output_dir, 1, m)
        rm(ClimaCalibrate.checkpoint_path(output_dir, 1, m))
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

# The interrupted iteration above wrote a first iteration of its own, and
# `initialize` keeps the stored one. The calibration below is checked against
# the ensemble it is given, so it starts from a directory of its own
rm(output_dir, recursive = true)

ekp = EKP.EnsembleKalmanProcess(
    observation,
    variance,
    EKP.Unscented(prior);
    # `Unscented` defaults to a scheduler that stops once the misfit target is
    # reached, which would end the run before `n_iterations`
    scheduler = EKP.DefaultScheduler(),
)
eki = ClimaCalibrate.Calibration.calibrate(
    ClimaCalibrate.WorkerBackend(),
    ekp,
    SurfaceFluxModelInterface(output_dir, ensemble_size),
    n_iterations,
    prior,
    output_dir,
)

@test ClimaCalibrate.last_completed_iteration(output_dir) == n_iterations

test_sf_calibration_output(eki, prior, observation)

theta_star_vec = (; coefficient_a_m_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger"],
    output_dir,
)

g_vs_iter_plot(eki, output_dir)

@testset "Restarts" begin
    last_iter = ClimaCalibrate.last_completed_iteration(output_dir)
    @test last_iter == n_iterations
    ClimaCalibrate.Calibration.run_iteration(
        ClimaCalibrate.WorkerBackend(),
        SurfaceFluxModelInterface(output_dir, ensemble_size),
        last_iter + 1,
        ensemble_size,
        output_dir,
    )
    G_ensemble = ClimaCalibrate.observation_map(
        SurfaceFluxModelInterface(output_dir, ensemble_size),
        last_iter + 1,
    )
    ClimaCalibrate.save_G_ensemble(output_dir, last_iter + 1, G_ensemble)
    ClimaCalibrate.update_ensemble(output_dir, last_iter + 1, prior)

    @test ClimaCalibrate.last_completed_iteration(output_dir) ==
          n_iterations + 1
end

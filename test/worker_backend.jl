using ClimaCalibrate, Distributed
import EnsembleKalmanProcesses as EKP

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)

nprocs = 5
if nworkers() == 1
    if get_backend() == ClimaCalibrate.DerechoBackend
        addprocs(
            PBSManager(nprocs),
            q = "main",
            A = "UCIT0011",
            l_select = "1:ncpus=1",
            l_walltime = "00:30:00",
        )
    else
        addprocs(SlurmManager(nprocs))
    end
end

@everywhere using ClimaCalibrate
@everywhere ClimaCalibrate.forward_model(i, m) = m == 1 && exit()

eki = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, ensemble_size),
    observation,
    variance,
    EKP.Inversion(),
    verbose = true,
)

ClimaCalibrate.initialize(eki, prior, output_dir)

ClimaCalibrate.run_worker_iteration(0, ensemble_size, output_dir)

@testset "Test model checkpoints with interruptions" begin
    for m in 1:ensemble_size
        @test m == 1 ? ClimaCalibrate.model_started(output_dir, 0, m) :
              ClimaCalibrate.model_completed(output_dir, 0, m)
        rm(ClimaCalibrate.checkpoint_path(output_dir, 0, m))
    end
end

@everywhere include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "model_interface.jl",
    ),
)

user_initial_ensemble = EKP.construct_initial_ensemble(prior, ensemble_size)
ekp = EKP.EnsembleKalmanProcess(
    user_initial_ensemble,
    observation,
    variance,
    EKP.Inversion();
    localization_method = EKP.Localizers.NoLocalization(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DefaultScheduler(),
)
eki = ClimaCalibrate.calibrate(
    WorkerBackend(),
    ekp,
    n_iterations,
    prior,
    output_dir,
)

@test ClimaCalibrate.last_completed_iteration(output_dir) == n_iterations - 1

test_sf_calibration_output(eki, prior, observation)

theta_star_vec =
    (; coefficient_a_m_businger = 4.7, coefficient_a_h_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger", "coefficient_a_h_businger"],
)

g_vs_iter_plot(eki)

@testset "Restarts" begin
    last_iter = ClimaCalibrate.last_completed_iteration(output_dir)
    @test last_iter == n_iterations - 1
    ClimaCalibrate.run_worker_iteration(
        last_iter + 1,
        ensemble_size,
        output_dir,
    )
    G_ensemble = observation_map(last_iter + 1)
    save_G_ensemble(output_dir, last_iter + 1, G_ensemble)
    update_ensemble(output_dir, last_iter + 1, prior)

    @test ClimaCalibrate.last_completed_iteration(output_dir) == n_iterations
end

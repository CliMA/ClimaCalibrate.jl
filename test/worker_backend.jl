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

nprocs = 5
if nworkers() == 1
    if ClimaCalibrate.get_backend() == ClimaCalibrate.DerechoBackend
        addprocs(
            PBSManager(nprocs),
            q = "main",
            A = "UCIT0011",
            l_select = "1:ncpus=1",
            l_walltime = "00:30:00",
        )
    else
        addprocs(ClimaCalibrate.SlurmManager(nprocs))
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

ClimaCalibrate.Calibration.run_iteration(
    ClimaCalibrate.WorkerBackend(),
    1,
    ensemble_size,
    output_dir,
)

@testset "Test model checkpoints with interruptions" begin
    for m in 1:ensemble_size
        @test m == 1 ? ClimaCalibrate.model_started(output_dir, 1, m) :
              ClimaCalibrate.model_completed(output_dir, 1, m)
        rm(ClimaCalibrate.checkpoint_path(output_dir, 1, m))
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

rng_seed = 1234
Random.seed!(rng_seed)
rng_ekp = Random.MersenneTwister(rng_seed)
user_initial_ensemble = EKP.construct_initial_ensemble(prior, ensemble_size)
ekp = EKP.EnsembleKalmanProcess(
    user_initial_ensemble,
    observation,
    variance,
    EKP.Inversion();
    rng = rng_ekp,
    localization_method = EKP.Localizers.NoLocalization(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DefaultScheduler(),
)
eki = ClimaCalibrate.Calibration.calibrate(
    ClimaCalibrate.WorkerBackend(),
    ekp,
    n_iterations,
    prior,
    output_dir,
)

@test ClimaCalibrate.last_completed_iteration(output_dir) == n_iterations

test_sf_calibration_output(eki, prior, observation)

theta_star_vec =
    (; coefficient_a_m_businger = 4.7, coefficient_a_h_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger", "coefficient_a_h_businger"],
    output_dir,
)

g_vs_iter_plot(eki, output_dir)

@testset "Restarts" begin
    last_iter = ClimaCalibrate.last_completed_iteration(output_dir)
    @test last_iter == n_iterations
    ClimaCalibrate.Calibration.run_iteration(
        ClimaCalibrate.WorkerBackend(),
        last_iter + 1,
        ensemble_size,
        output_dir,
    )
    G_ensemble = ClimaCalibrate.observation_map(last_iter + 1)
    ClimaCalibrate.save_G_ensemble(output_dir, last_iter + 1, G_ensemble)
    ClimaCalibrate.update_ensemble(output_dir, last_iter + 1, prior)

    @test ClimaCalibrate.last_completed_iteration(output_dir) ==
          n_iterations + 1
end

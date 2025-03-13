using ClimaCalibrate, Distributed

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)
# Expression to run on worker initialization, used instead of @everywhere
expr = quote
    using ClimaCalibrate
    include(
        joinpath(
            pkgdir(ClimaCalibrate),
            "experiments",
            "surface_fluxes_perfect_model",
            "model_interface.jl",
        ),
    )
end

if nworkers() == 1
    if get_backend() == ClimaCalibrate.DerechoBackend
        @async addprocs(
            PBSManager(5; expr),
            q = "preempt",
            A = "UCIT0011",
            l_select = "1:ncpus=1:ngpus=1",
            l_walltime = "00:30:00",
        )
    else
        @async addprocs(SlurmManager(5; expr))
    end
end

eki = calibrate(
    WorkerBackend,
    ensemble_size,
    n_iterations,
    observation,
    variance,
    prior,
    output_dir;
    localization_method = EKP.Localizers.NoLocalization(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DefaultScheduler(),
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
    initialize(ensemble_size, observation, variance, prior, output_dir)

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

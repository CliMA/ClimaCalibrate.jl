using ClimaCalibrate

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)
experiment_config = ExperimentConfig(
    n_iterations,
    ensemble_size,
    observation,
    variance,
    prior,
    output_dir,
)

@assert get_backend() <: HPCBackend
hpc_kwargs = kwargs(time = 5, ntasks = 1, cpus_per_task = 1)
if get_backend() == DerechoBackend
    hpc_kwargs[:queue] = "preempt"
    hpc_kwargs[:gpus_per_task] = 1
end

@testset "Restarts" begin
    initialize(ensemble_size, observation, variance, prior, output_dir)

    last_iter = ClimaCalibrate.last_completed_iteration(output_dir)
    @test last_iter == -1
    ClimaCalibrate.run_worker_iteration(
        last_iter + 1,
        ensemble_size,
        output_dir,
    )
    G_ensemble = observation_map(last_iter + 1)
    save_G_ensemble(output_dir, last_iter + 1, G_ensemble)
    update_ensemble(output_dir, last_iter + 1, prior)

    @test ClimaCalibrate.last_completed_iteration(output_dir) == 0
end

eki = calibrate(experiment_config; model_interface, hpc_kwargs, verbose = true)

@test ClimaCalibrate.last_completed_iteration(output_dir) == n_iterations - 1
test_sf_calibration_output(eki, prior, experiment_config.observations)

# Remove previous output - this is not necessary but safe for tests
rm(output_dir, recursive = true)
# Pure Julia calibration, this should run anywhere
julia_eki = calibrate(JuliaBackend, experiment_config)
test_sf_calibration_output(julia_eki, prior, experiment_config.observations)

compare_g_ensemble(eki, julia_eki)

theta_star_vec =
    (; coefficient_a_m_businger = 4.7, coefficient_a_h_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger", "coefficient_a_h_businger"],
)

g_vs_iter_plot(eki)

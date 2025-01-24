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
eki = calibrate(experiment_config; model_interface, hpc_kwargs, verbose = true)
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

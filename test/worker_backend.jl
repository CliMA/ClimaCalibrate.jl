using ClimaCalibrate, Distributed

if nworkers() == 1
    addprocs(SlurmManager(5))
end

include(joinpath(pkgdir(ClimaCalibrate), "test", "sf_calibration_utils.jl"))

eki = calibrate(
    WorkerBackend,
    ensemble_size,
    n_iterations,
    observation,
    variance,
    prior,
    output_dir,
)

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

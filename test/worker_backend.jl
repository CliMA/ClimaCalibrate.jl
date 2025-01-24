using ClimaCalibrate, Distributed

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)

if nworkers() == 1
    if get_backend() == ClimaCalibrate.DerechoBackend
        addprocs(
            PBSManager(5),
            q = "preempt",
            A = "UCIT0011",
            l_select = "1:ncpus=1:ngpus=1",
            l_walltime = "00:30:00",
        )
    else
        addprocs(SlurmManager(5))
    end
end

@everywhere using ClimaCalibrate
@everywhere include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "model_interface.jl",
    ),
)

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

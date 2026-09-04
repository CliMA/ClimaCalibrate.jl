# Runs the surface-fluxes example and writes the two figures the quickstart page
# shows. `make.jl` includes this before `makedocs`, so the figures are the ones
# the code in the page produces.
#
# This runs in `Main` rather than in an `@example` block: `utils.jl` sets its
# definitions up with `@everywhere`, which evaluates them in `Main`, while
# Documenter runs each block in a module of its own.

import ClimaCalibrate
import EnsembleKalmanProcesses as EKP

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)

let
    ekp = EKP.EnsembleKalmanProcess(observation, variance, EKP.Unscented(prior))
    figure_dir = mktempdir()
    eki = ClimaCalibrate.calibrate(
        ClimaCalibrate.JuliaBackend(),
        ekp,
        SurfaceFluxModelInterface(figure_dir, ensemble_size),
        n_iterations,
        prior,
        figure_dir,
    )

    theta_star_vec = (; coefficient_a_m_businger = 4.7)
    convergence_plot(
        eki,
        prior,
        theta_star_vec,
        ["coefficient_a_m_businger"],
        figure_dir,
    )
    g_vs_iter_plot(eki, figure_dir)
    loss_landscape_plot(
        observation,
        variance,
        figure_dir;
        calibrated = only(EKP.get_ϕ_mean_final(prior, eki)),
    )

    assets = joinpath(@__DIR__, "src", "assets")
    for (from, to) in (
        "convergence_coefficient_a_m_businger.png" => "sf_convergence_coefficient_a_m_businger.png",
        "scatter_iter.png" => "sf_scatter_iter.png",
        "loss_landscape.png" => "sf_loss_landscape.png",
    )
        cp(joinpath(figure_dir, from), joinpath(assets, to); force = true)
    end
    @info "Wrote the surface-fluxes figures to $assets"
end

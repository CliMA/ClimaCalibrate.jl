using Test
using Distributed
import ClimaCalibrate
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import Statistics
import JLD2
import CairoMakie

experiment_dir = joinpath(
    pkgdir(ClimaCalibrate),
    "experiments",
    "surface_fluxes_perfect_model",
)
include(joinpath(experiment_dir, "generate_data.jl"))

@everywhere begin
    using ClimaCalibrate
    import EnsembleKalmanProcesses.ParameterDistributions as PD
    output_dir = joinpath("output", "surface_fluxes_perfect_model")
    prior_vec = [
        PD.constrained_gaussian("coefficient_a_m_businger", 4.7, 0.5, 2, 6),
        PD.constrained_gaussian("coefficient_a_h_businger", 4.6, 3, 0, 10),
    ]
    prior = PD.combine_distributions(prior_vec)
    ensemble_size = 20
    n_iterations = 6

    experiment_dir = $experiment_dir
    include(joinpath(experiment_dir, "observation_map.jl"))
    ustar = JLD2.load_object(
        joinpath(experiment_dir, "data", "synthetic_ustar_array_noisy.jld2"),
    )
    (; observation, variance) =
        process_member_data(ustar; output_variance = true)

    model_interface = joinpath(experiment_dir, "model_interface.jl")
    include(model_interface)
end


"""
    test_sf_calibration_output(eki, prior, observation)

Test that the surface fluxes perfect model calibration converges.

The final forward model output should approximately be equal to the
observations, and parameter spread should decrease from the first to last
iteration.
"""
function test_sf_calibration_output(eki, prior, observation)
    @testset "SurfaceFluxes calibration reproducibility tests" begin
        params = EKP.get_ϕ(prior, eki)
        spread = map(Statistics.var, params)

        # Spread should be heavily decreased the ensemble has converged
        @test last(spread) / first(spread) < 0.3

        forward_model_output = EKP.get_g_mean_final(eki)
        @show forward_model_output
        @test all(isapprox.(forward_model_output, observation; rtol = 1e-2))
    end
end

"""
    compare_g_ensemble(eki1, eki2)

Test that the forward model output (`g`) of two `EKP.EnsembleKalmanProcess`
objects is approximately equal across all iterations.
"""
function compare_g_ensemble(eki1, eki2)
    @testset "Compare g_ensemble between two EKP objects" begin
        for (g1, g2) in zip(eki1.g, eki2.g)
            @test g1.data ≈ g2.data rtol = 1e-5
        end
    end
end

"""
    convergence_plot(eki, prior, theta_star_vec, param_names, output_dir)

Generate a 2x2 figure for each parameter showing: error versus iteration,
ensemble spread versus iteration, unconstrained parameters versus iteration, and
constrained parameters versus iteration.

Plots are saved to `output_dir`.
"""
function convergence_plot(eki, prior, theta_star_vec, param_names, output_dir)
    for (param_idx, param_name) in enumerate(param_names)

        u_vec_all_params = EKP.get_u(eki) # vector of iterations that contain parameters from the ensemble (per parameter)
        u_vec = [u[param_idx, :] for u in u_vec_all_params]
        theta_star = getproperty(theta_star_vec, Symbol(param_name))

        n_iterations = length(u_vec)
        ensemble_size = u_vec[1] |> length

        phi_vec_all =
            transform_unconstrained_to_constrained(prior, u_vec_all_params)
        phi_vec = [phi[param_idx, :] for phi in phi_vec_all]

        spread_vec = Statistics.var.(u_vec; corrected = false)

        u_series = [getindex.(u_vec, i) for i in 1:ensemble_size]
        phi_series = [getindex.(phi_vec, i) for i in 1:ensemble_size]

        f = CairoMakie.Figure(size = (800, 800))

        ax = CairoMakie.Axis(
            f[1, 1],
            xlabel = "Iteration",
            ylabel = "Error",
            xticks = 0:50,
            title = "Error for $param_name",
        )
        CairoMakie.lines!(
            ax,
            1.0:length(EKP.get_error(eki)),
            EKP.get_error(eki),
        )

        ax = CairoMakie.Axis(
            f[1, 2],
            xlabel = "Iteration",
            ylabel = "Spread",
            xticks = 0:50,
        )
        CairoMakie.lines!(ax, 1.0:length(spread_vec), spread_vec)

        ax = CairoMakie.Axis(
            f[2, 1],
            xlabel = "Iteration",
            ylabel = "Unconstrained Parameters",
            xticks = 0:50,
        )
        CairoMakie.lines!.(ax, tuple(1.0:length(u_series[1])), u_series)

        ax = CairoMakie.Axis(
            f[2, 2],
            xlabel = "Iteration",
            ylabel = "Constrained Parameters ($param_name)",
            xticks = 0:50,
        )
        CairoMakie.lines!.(ax, tuple(1.0:length(phi_series[1])), phi_series)
        CairoMakie.hlines!(ax, [theta_star], color = :red, linestyle = :dash)

        CairoMakie.save(joinpath(output_dir, "convergence_$param_name.png"), f)
        @info "Convergence plot path: $(joinpath(output_dir, "convergence_$param_name.png"))"
    end
end

"""
    g_vs_iter_plot(eki, output_dir)

Plot ensemble `ustar` values against iteration, with dashed reference lines for
the true value (blue line) and calibrated value (red line) of `ustar`.

The plot is saved to `output_dir`.
"""
function g_vs_iter_plot(eki, output_dir)
    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1, 1], xlabel = "Iteration", ylabel = "Model Ustar")
    ustar_obs = JLD2.load_object(
        joinpath(experiment_dir, "data", "synthetic_ustar_array_noisy.jld2"),
    )

    pkg_dir = pkgdir(ClimaCalibrate)
    x_data_file = joinpath(
        pkg_dir,
        "experiments",
        "surface_fluxes_perfect_model",
        "data",
        "synthetic_profile_data.jld2",
    )
    x_inputs = load_profiles(x_data_file)

    FT = Float32
    n_iterations = EKP.get_N_iterations(eki)
    ensemble_size = EKP.get_N_ens(eki)
    model_config = Dict()
    model_config["output_dir"] = output_dir
    for iter in 1:(n_iterations + 1)
        for i in 1:ensemble_size
            model_config["toml"] = [
                joinpath(
                    pkg_dir,
                    ClimaCalibrate.parameter_path(output_dir, iter, i),
                ),
            ]
            ustar_mod =
                obtain_ustar(FT, x_inputs, model_config, return_ustar = true)

            CairoMakie.scatter!(iter, nanmean(ustar_mod[:]))

        end
    end
    CairoMakie.lines!(
        ax,
        [1, n_iterations + 1],
        [nanmean(ustar_obs), nanmean(ustar_obs)],
        color = :red,
        linestyle = :dash,
    )
    model_config["toml"] = []
    ustar_mod_perfect_params =
        obtain_ustar(FT, x_inputs, model_config, return_ustar = true)
    CairoMakie.lines!(
        ax,
        [1, n_iterations + 1],
        [
            nanmean(ustar_mod_perfect_params[:]),
            nanmean(ustar_mod_perfect_params[:]),
        ],
        color = :blue,
        linestyle = :dash,
    )

    CairoMakie.save(joinpath(output_dir, "scatter_iter.png"), f)
    @info "Scatter plot path: $(joinpath(output_dir, "scatter_iter.png"))"
end

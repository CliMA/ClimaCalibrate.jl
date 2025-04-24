using Test
import EnsembleKalmanProcesses: get_ϕ, get_g_mean_final
import Statistics: var
using Distributed

function test_sf_calibration_output(eki, prior, observation)
    @testset "SurfaceFluxes calibration reproducibility tests" begin
        params = get_ϕ(prior, eki)
        spread = map(var, params)

        # Spread should be heavily decreased the ensemble has converged
        @test last(spread) / first(spread) < 0.15

        forward_model_output = get_g_mean_final(eki)
        @show forward_model_output
        @test all(isapprox.(forward_model_output, observation; rtol = 1e-2))
    end
end

function compare_g_ensemble(eki1, eki2)
    @testset "Compare g_ensemble between two EKP objects" begin
        for (g1, g2) in zip(eki1.g, eki2.g)
            @test g1.stored_data == g2.stored_data
        end
    end
end

experiment_dir = project_dir()
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
    n_iterations = 8
end

@everywhere begin
    experiment_dir = project_dir()
    include(joinpath(experiment_dir, "observation_map.jl"))
    ustar = JLD2.load_object(
        joinpath(experiment_dir, "data", "synthetic_ustar_array_noisy.jld2"),
    )
    (; observation, variance) =
        process_member_data(ustar; output_variance = true)

    model_interface = joinpath(experiment_dir, "model_interface.jl")
    include(model_interface)
end

# Postprocessing
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
using Distributions
import JLD2
import Statistics: mean
import YAML
import TOML
import CairoMakie: Makie
using Statistics

function convergence_plot(
    eki,
    prior,
    theta_star_vec,
    param_names,
    output_dir = output_dir,
)
    for (param_idx, param_name) in enumerate(param_names)

        u_vec_all_params = EKP.get_u(eki) # vector of iterations that contain parameters from the ensemble (per parameter)
        u_vec = [u[param_idx, :] for u in u_vec_all_params]
        theta_star = getproperty(theta_star_vec, Symbol(param_name))

        n_iterations = length(u_vec)
        ensemble_size = u_vec[1] |> length

        phi_vec_all =
            transform_unconstrained_to_constrained(prior, u_vec_all_params)
        phi_vec = [phi[param_idx, :] for phi in phi_vec_all]

        spread_vec = Float64[]
        for ensemble in u_vec # iterate over the iterations
            ensemble_spread = 0
            ensemble_mean = mean(ensemble)
            for i in ensemble # ! not generalized for multiple params (here param 1)
                ensemble_spread += abs(i - ensemble_mean)^2

            end
            ensemble_spread /= length(ensemble)

            push!(spread_vec, ensemble_spread)
        end

        u_series = [getindex.(u_vec, i) for i in 1:ensemble_size]
        phi_series = [getindex.(phi_vec, i) for i in 1:ensemble_size]

        f = Makie.Figure(resolution = (800, 800))

        ax = Makie.Axis(
            f[1, 1],
            xlabel = "Iteration",
            ylabel = "Error",
            xticks = 0:50,
            title = "Error for $param_name",
        )
        Makie.lines!(ax, 0.0:(length(eki.error) - 1), eki.error)

        ax = Makie.Axis(
            f[1, 2],
            xlabel = "Iteration",
            ylabel = "Spread",
            xticks = 0:50,
        )
        Makie.lines!(ax, 0.0:(length(spread_vec) - 1), spread_vec)

        ax = Makie.Axis(
            f[2, 1],
            xlabel = "Iteration",
            ylabel = "Unconstrained Parameters",
            xticks = 0:50,
        )
        Makie.lines!.(ax, tuple(0.0:(length(u_series[1]) - 1)), u_series)

        ax = Makie.Axis(
            f[2, 2],
            xlabel = "Iteration",
            ylabel = "Constrained Parameters ($param_name)",
            xticks = 0:50,
        )
        Makie.lines!.(ax, tuple(0.0:(length(phi_series[1]) - 1)), phi_series)
        Makie.hlines!(ax, [theta_star], color = :red, linestyle = :dash)

        Makie.save(joinpath(output_dir, "convergence_$param_name.png"), f)
        println(joinpath(output_dir, "convergence_$param_name.png"))
    end
end

# Plot the convergence of the model observable: ustar
function g_vs_iter_plot(eki)
    pkg_dir = pkgdir(ClimaCalibrate)
    FT = Float32

    f = Makie.Figure()
    ax = Makie.Axis(f[1, 1], xlabel = "Iteration", ylabel = "Model Ustar")
    ustar_obs = JLD2.load_object(
        joinpath(experiment_dir, "data", "synthetic_ustar_array_noisy.jld2"),
    )

    x_data_file = joinpath(
        pkg_dir,
        "experiments",
        "surface_fluxes_perfect_model",
        "data",
        "synthetic_profile_data.jld2",
    )
    x_inputs = load_profiles(x_data_file)

    ustar_mod = 0
    model_config = Dict()
    model_config["output_dir"] = output_dir
    for iter in 0:n_iterations
        for i in 1:ensemble_size
            model_config["toml"] = [
                joinpath(
                    pkg_dir,
                    ClimaCalibrate.parameter_path(output_dir, iter, i),
                ),
            ]
            ustar_mod =
                obtain_ustar(FT, x_inputs, model_config, return_ustar = true)

            Makie.scatter!(iter, nanmean(ustar_mod[:]))

        end
    end
    Makie.lines!(
        ax,
        [0, n_iterations],
        [nanmean(ustar_obs), nanmean(ustar_obs)],
        color = :red,
        linestyle = :dash,
    )
    model_config["toml"] = []
    ustar_mod_perfect_params =
        obtain_ustar(FT, x_inputs, model_config, return_ustar = true)
    Makie.lines!(
        ax,
        [0, n_iterations],
        [
            nanmean(ustar_mod_perfect_params[:]),
            nanmean(ustar_mod_perfect_params[:]),
        ],
        color = :blue,
        linestyle = :dash,
    )

    Makie.save(joinpath(output_dir, "scatter_iter.png"), f)
    println(joinpath(output_dir, "scatter_iter.png"))
end

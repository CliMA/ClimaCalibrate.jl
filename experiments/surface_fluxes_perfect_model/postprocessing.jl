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

using CalibrateAtmos

experiment_dir = dirname(Base.active_project())
experiment_config = CalibrateAtmos.ExperimentConfig(experiment_dir)
output_dir = experiment_config.output_dir
experiment_id = experiment_config.id
N_iter = experiment_config.n_iterations
N_mem = experiment_config.ensemble_size

function convergence_plot(
    eki,
    prior,
    theta_star_vec,
    param_names,
    output_dir = experiment_config.output_dir,
)

    # per parameter
    for (param_idx, param_name) in enumerate(param_names)

        u_vec_all_params = EKP.get_u(eki) # vector of iterations that contain parameters from the ensemble (per parameter)
        u_vec = [u[param_idx, :] for u in u_vec_all_params]
        theta_star = getproperty(theta_star_vec, Symbol(param_name))

        N_iter = length(u_vec)
        N_mem = u_vec[1] |> length

        phi_vec_all =
            transform_unconstrained_to_constrained(prior, u_vec_all_params)
        phi_vec = [phi[param_idx, :] for phi in phi_vec_all]

        error_vec = Float64[]
        spread_vec = Float64[]
        for ensemble in u_vec # iterate over the iterations
            ensemble_error = 0
            ensemble_spread = 0
            ensemble_mean = mean(ensemble)
            for i in ensemble # ! not generalized for multiple params (here param 1)
                ensemble_error += abs(i - theta_star)^2
                ensemble_spread += abs(i - ensemble_mean)^2

            end
            ensemble_error /= length(ensemble)
            ensemble_spread /= length(ensemble)

            push!(error_vec, ensemble_error)
            push!(spread_vec, ensemble_spread)
        end

        u_series = [getindex.(u_vec, i) for i in 1:N_mem]
        phi_series = [getindex.(phi_vec, i) for i in 1:N_mem]

        f = Makie.Figure(resolution = (800, 800))

        ax = Makie.Axis(
            f[1, 1],
            xlabel = "Iteration",
            ylabel = "Error",
            xticks = 0:50,
            title = "Error for $param_name",
        )
        Makie.lines!(ax, 0.0:(length(error_vec) - 1), error_vec)

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
    end
end


pkg_dir = pkgdir(CalibrateAtmos)
model_config = YAML.load_file(
    joinpath(pkg_dir, "experiments", experiment_id, "model_config.yml"),
)

eki_path = joinpath(
    CalibrateAtmos.path_to_iteration(output_dir, N_iter),
    "eki_file.jld2",
);
eki = JLD2.load_object(eki_path);
EKP.get_u(eki)
prior = experiment_config.prior

theta_star_vec =
    (; coefficient_a_m_businger = 4.7, coefficient_a_h_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger", "coefficient_a_h_businger"],
)

# Plot the convergence of the model observable: ustar
using Pkg
FT = Float32
include(joinpath(experiment_dir, "model_interface.jl"))

f = Makie.Figure()
ax = Makie.Axis(f[1, 1], xlabel = "Iteration", ylabel = "Model Ustar")
ustar_obs = JLD2.load_object(
    joinpath(
        pkg_dir,
        "experiments/$experiment_id/data/synthetic_ustar_array_noisy.jld2",
    ),
)
x_inputs = load_profiles(model_config["x_data_file"])

ustar_mod = 0
for iter in 0:N_iter
    for i in 1:N_mem
        model_config["toml"] = [
            joinpath(
                pkg_dir,
                CalibrateAtmos.path_to_ensemble_member(output_dir, iter, i),
                "parameters.toml",
            ),
        ]
        ustar_mod =
            obtain_ustar(FT, x_inputs, model_config, return_ustar = true)

        Makie.scatter!(iter, nanmean(ustar_mod[:]))

    end
end
Makie.lines!(
    ax,
    [0, N_iter],
    [nanmean(ustar_obs), nanmean(ustar_obs)],
    color = :red,
    linestyle = :dash,
)
model_config["toml"] = []
ustar_mod_perfect_params =
    obtain_ustar(FT, x_inputs, model_config, return_ustar = true)
Makie.lines!(
    ax,
    [0, N_iter],
    [
        nanmean(ustar_mod_perfect_params[:]),
        nanmean(ustar_mod_perfect_params[:]),
    ],
    color = :blue,
    linestyle = :dash,
)

Makie.save(joinpath(output_dir, "scatter_iter.png"), f)

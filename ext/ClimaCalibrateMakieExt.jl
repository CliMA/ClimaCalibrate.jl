module ClimaCalibrateMakieExt

import Makie
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate

import ClimaCalibrate.Visualization:
    plot_g, plot_g!, plot_g_mean, plot_g_mean!, plot_obs, plot_obs!

"""
    plot_g

Plot members of the G ensemble matrix as line plots.

If the `iter` keyword argument is not passed, then this plots the last completed
G ensemble matrix. Otherwise, it plots the `iter`th G ensemble matrix.
"""
Makie.@recipe Plot_G (ekp,) begin
    "Iteration of G ensemble matrix to plot. If `nothing`, then use the last
    iteration"
    iter = nothing
    Makie.documented_attributes(Makie.Lines)...
end

"""
    Makie.plot!(g_ens_plot::Plot_G)

Plot members of the G ensemble matrix as line plots.

This function is called when using `Visualization.plot_g` or
`Visualization.plot_g!`.
"""
function Makie.plot!(g_ens_plot::Plot_G)
    ekp = g_ens_plot.ekp[]
    iter = _iter_or_total(ekp, g_ens_plot.iter[])
    g_ens_matrix = EKP.get_g(ekp, iter)
    for j in axes(g_ens_matrix, 2)
        ensemble_member = view(g_ens_matrix, :, j)
        Makie.lines!(
            g_ens_plot,
            g_ens_plot.attributes,
            1:length(ensemble_member),
            ensemble_member,
        )
    end
    return g_ens_plot
end

"""
    plot_g_mean

Plot mean forward map evaluation as a line plot.

If the `iter` keyword argument is not passed, then this plots the last mean
forward map evaluation. Otherwise, it plots the `iter`th mean forward map
evaluation.
"""
Makie.@recipe Plot_G_Mean (ekp,) begin
    "Iteration of the G ensemble matrix to plot. If `nothing`, then use the last
    iteration"
    iter = nothing
    Makie.documented_attributes(Makie.Lines)...
end

"""
    Makie.plot!(g_ens_plot::Plot_G_Mean)

Plot mean forward map evaluation as a line plot.

This function is called when using `Visualization.plot_g_mean` or
`Visualization.plot_g_mean!`.
"""
function Makie.plot!(g_mean_plot::Plot_G_Mean)
    ekp = g_mean_plot.ekp[]
    iter = _iter_or_total(ekp, g_mean_plot.iter[])
    ensemble_member = EKP.get_g_mean(ekp, iter)
    Makie.lines!(
        g_mean_plot,
        g_mean_plot.attributes,
        1:length(ensemble_member),
        ensemble_member,
    )
    return g_mean_plot
end


"""
    plot_obs

Plot the observations as a line plot.

If the `iter` keyword argument is not passed, then this plots the observations
for the last iteration. Otherwise, it plots the observations for the `iter`th
iteration.
"""
Makie.@recipe Plot_Obs (ekp,) begin
    "Iteration of the G ensemble matrix to plot. If `nothing`, then use the last
    iteration"
    iter = nothing
    Makie.documented_attributes(Makie.Lines)...
end

"""
    Makie.plot!(obs_plot::Plot_Obs)

Plot the observations as a line plot.

This function is called when using `Visualization.plot_obs` or
`Visualization.plot_obs!`.
"""
function Makie.plot!(obs_plot::Plot_Obs)
    ekp = obs_plot.ekp[]
    iter = _iter_or_total(ekp, obs_plot.iter[])
    obs_series = EKP.get_observation_series(ekp)
    obs = ClimaCalibrate.get_observations_for_nth_iteration(obs_series, iter)
    stacked_obs = mapreduce(EKP.get_obs, vcat, obs)
    Makie.lines!(
        obs_plot,
        obs_plot.attributes,
        1:length(stacked_obs),
        stacked_obs,
    )
    return obs_plot
end

@static if pkgversion(Makie) >= v"0.24.11"
    function Makie.preferred_axis_attributes(
        ::Type{Makie.Axis},
        g_ens_plot::Plot_G,
    )
        iter = _iter_or_total(g_ens_plot.ekp[], g_ens_plot.iter[])
        xlabel = "Index"
        ylabel = "Value"
        title = "G ensemble members for iteration $iter"
        return (; title, ylabel, xlabel)
    end

    function Makie.preferred_axis_attributes(
        ::Type{Makie.Axis},
        g_mean_plot::Plot_G_Mean,
    )
        iter = _iter_or_total(g_mean_plot.ekp[], g_mean_plot.iter[])
        xlabel = "Index"
        ylabel = "Value"
        title = "Mean G for iteration $iter"
        return (; title, ylabel, xlabel)
    end

    function Makie.preferred_axis_attributes(
        ::Type{Makie.Axis},
        obs_plot::Plot_Obs,
    )
        iter = _iter_or_total(obs_plot.ekp[], obs_plot.iter[])
        xlabel = "Index"
        ylabel = "Value"
        title = "Observations for iteration $iter"
        return (; title, ylabel, xlabel)
    end
end

"""
    _iter_or_total(ekp::EKP.EnsembleKalmanProcess, iter::Union{Int, Nothing})

Return `iter` if `iter` is an `Int` or the total number of iterations if `iter`
is `nothing`.
"""
function _iter_or_total(::EKP.EnsembleKalmanProcess, iter::Int)
    return iter
end
function _iter_or_total(ekp::EKP.EnsembleKalmanProcess, ::Nothing)
    return EKP.get_N_iterations(ekp)
end

end

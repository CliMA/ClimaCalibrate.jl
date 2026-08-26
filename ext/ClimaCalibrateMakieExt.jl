module ClimaCalibrateMakieExt

import Makie
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate

import ClimaCalibrate.Visualization:
    plot_g,
    plot_g!,
    plot_g_mean,
    plot_g_mean!,
    plot_obs,
    plot_obs!,
    plot_residual,
    plot_residual!

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

"""
    plot_residual

Plot the normalized residual `(mean(G) - obs) / σ` as a scatter plot, where `σ`
is the square root of the diagonal of the observation noise covariance.
Reference lines are drawn at zero and at plus and minus one and two `σ`.

If the `iter` keyword argument is not passed, then this plots the residual of
the last iteration. Otherwise, it plots the residual of the `iter`th iteration.

If the keyword argument `reference_lines = false`, then the reference lines are
not drawn.

If the keyword argument `ignore_nan = true`, then the mean of the G ensemble at
each index is computed over the ensemble members that are not `NaN`.
"""
Makie.@recipe Plot_Residual (ekp,) begin
    "Iteration of the G ensemble matrix to plot. If `nothing`, then use the last
    iteration"
    iter = nothing
    "Draw reference lines at zero and at plus and minus one and two `σ`. Set to
    `false` to draw your own with `Makie.hlines!`"
    reference_lines = true
    "If `true`, then the mean of the G ensemble at each index is computed over
    the ensemble members that are not `NaN`"
    ignore_nan = true
    Makie.documented_attributes(Makie.Scatter)...
end

"""
    Makie.plot!(residual_plot::Plot_Residual)

Plot the normalized residual `(mean(G) - obs) / σ` as a scatter plot.

This function is called when using `Visualization.plot_residual` or
`Visualization.plot_residual!`.
"""
function Makie.plot!(residual_plot::Plot_Residual)
    ekp = residual_plot.ekp[]
    iter = _iter_or_total(ekp, residual_plot.iter[])
    residual = ClimaCalibrate.residual(
        ekp;
        N = iter,
        ignore_nan = residual_plot.ignore_nan[],
    )
    if residual_plot.reference_lines[]
        Makie.hlines!(residual_plot, [0.0], color = :blue)
        Makie.hlines!(
            residual_plot,
            [-1.0, 1.0],
            color = :gray,
            linestyle = :dash,
        )
        Makie.hlines!(
            residual_plot,
            [-2.0, 2.0],
            color = :red,
            linestyle = :dot,
        )
    end
    Makie.scatter!(
        residual_plot,
        residual_plot.attributes,
        1:length(residual),
        residual,
    )
    return residual_plot
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

    function Makie.preferred_axis_attributes(
        ::Type{Makie.Axis},
        residual_plot::Plot_Residual,
    )
        ekp = residual_plot.ekp[]
        iter = _iter_or_total(ekp, residual_plot.iter[])
        residual = ClimaCalibrate.residual(
            ekp;
            N = iter,
            ignore_nan = residual_plot.ignore_nan[],
        )
        rms = round(sqrt(sum(abs2, residual) / length(residual)), sigdigits = 2)
        xlabel = "Index"
        ylabel = "Residual [σ]"
        title = "Residual for iteration $iter, RMS = $(rms)σ"
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

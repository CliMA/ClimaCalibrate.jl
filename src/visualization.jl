"""
    ClimaCalibrate.Visualization

Makie plots of a calibration's forward map output against its observations.

[`plot_g`](@ref) draws every ensemble member, [`plot_g_mean`](@ref) the ensemble
mean, and [`plot_obs`](@ref) the observations, so the three can be overlaid to
see whether the ensemble is bracketing the data.

Requires a Makie backend such as CairoMakie to be loaded.
"""
module Visualization

export plot_g, plot_g!, plot_g_mean, plot_g_mean!, plot_obs, plot_obs!

"""
    plot_g

Plot members of the G ensemble matrix as line plots.

If the `iter` keyword argument is not passed, then this plots the last completed
G ensemble matrix. Otherwise, it plots the `iter`th G ensemble matrix.

All `Makie` keyword arguments compatible with `Makie.lines` are also compatible
with this function.

# Examples
```julia
import CairoMakie
fig, ax, _ = ClimaCalibrate.Visualization.plot_g(ekp; color = (:grey, 0.3))
ClimaCalibrate.Visualization.plot_obs!(ax, ekp; color = :black)
```

See also [`plot_g_mean`](@ref), [`plot_obs`](@ref).
"""
function plot_g end

"""
    plot_g!

This is the mutating variant of the plotting function [`plot_g`](@ref).
"""
function plot_g! end

"""
    plot_g_mean

Plot mean forward map evaluation as a line plot.

If the `iter` keyword argument is not passed, then this plots the last mean
forward map evaluation. Otherwise, it plots the `iter`th mean forward map
evaluation.

All `Makie` keyword arguments compatible with `Makie.lines` are also compatible
with this function.
"""
function plot_g_mean end

"""
    plot_g_mean!

This is the mutating variant of the plotting function [`plot_g_mean`](@ref).
"""
function plot_g_mean! end


"""
    plot_obs

Plot the observations as a line plot.

If the `iter` keyword argument is not passed, then this plots the observations
of the last iteration. Otherwise, it plots the observations of the `iter`th
iteration.

All `Makie` keyword arguments compatible with `Makie.lines` are also compatible
with this function.
"""
function plot_obs end

"""
    plot_obs!

This is the mutating variant of the plotting function [`plot_obs`](@ref).
"""
function plot_obs! end

end

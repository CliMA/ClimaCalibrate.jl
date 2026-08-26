module Visualization

export plot_g,
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

All `Makie` keyword arguments compatible with `Makie.lines` are also compatible
with this function.
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

"""
    plot_residual

Plot the normalized residual `(mean(G) - obs) / σ` as a scatter plot, where `σ`
is the square root of the diagonal of the observation noise covariance.
Reference lines are drawn at zero and at plus and minus one and two `σ`; pass
`reference_lines = false` to not draw them.

If the `iter` keyword argument is not passed, then this plots the residual of
the last iteration. Otherwise, it plots the residual of the `iter`th iteration.

If the keyword argument `ignore_nan = true`, then the mean of the G ensemble at
each index is computed over the ensemble members that are not `NaN`.

All `Makie` keyword arguments compatible with `Makie.scatter` are also
compatible with this function.
"""
function plot_residual end

"""
    plot_residual!

This is the mutating variant of the plotting function [`plot_residual`](@ref).
"""
function plot_residual! end

end

```@meta
CurrentModule = ClimaCalibrate
```

# Visualization

ClimaCalibrate provides primitive plotting utilities for plotting the mean
forward map evaluation, columns of the G ensemble matrix, and the true
observation via a `Makie` extension.

!!! note "Scope of plotting utilities"
    Since the plotting utilities are general, they may be insufficient for your
    use case. The plotting functions do not use metadata in the
    `EKP.EnsembleKalmanProcess` object, since the metadata are specific to the
    calibration that you are conducting. Hence, if these plotting utilities are
    insufficient, you should use the metadata to transform the data in the
    `EKP.EnsembleKalmanProcess` object to data that is more suitable for
    plotting.

To plot the mean forward map evaluation, columns of the G ensemble matrix, and
the true observation, you can use [`Visualization.plot_g_mean`](@ref),
[`Visualization.plot_g`](@ref), and [`Visualization.plot_obs`](@ref)
respectively. The mutating versions also exist as
[`Visualization.plot_g_mean!`](@ref), [`Visualization.plot_g!`](@ref), and
[`Visualization.plot_obs!`](@ref). All plotting functions takes an
`EKP.EnsembleKalmanProcess` object to plot from. Additionally, the plotting
function accept an `iter` keyword argument for plotting from a specific
iteration. If the keyword argument is not provided, then the last iteration is
used for plotting. You can expect all keyword arguments that work with
`Makie.Lines` to also work with these plotting functions and that the plotting
functions behave like `Makie` plotting functions.

!!! tip "Keyword arguments"
    You can enter `help?> ClimaCalibrate.Visualization.plot_g` in the Julia REPL
    to get a list of keyword arguments that work with `Visualization.plot_g`.
    You can do the same with the other plotting functions.

## Example

Here is a complete example where we use the plotting functions to plot the
ensemble members, the mean forward map evaluation, and the true observations
from the second iteration.

```@setup plot
import Dates
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
import ClimaAnalysis
import ClimaAnalysis.Template:
    TemplateVar,
    make_template_var,
    add_attribs,
    add_dim,
    add_time_dim,
    add_lon_dim,
    add_lat_dim,
    add_data,
    ones_data,
    zeros_data,
    one_to_n_data,
    initialize
import ClimaCalibrate
import ClimaCalibrate: ObservationRecipe, EnsembleBuilder, SampleBuilder

lat = [-90.0, -30.0, 30.0, 90.0]
lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
time = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
data3d = [t * cos(x / 90) * sin(y / 180) for t in time, x in lon, y in lat]
var =
    TemplateVar() |>
    add_dim("time", time, units = "s") |>
    add_dim("lon", lon, units = "degrees") |>
    add_dim("lat", lat, units = "degrees") |>
    add_attribs(
        short_name = "hi",
        long_name = "hello",
        start_date = "2007-12-1",
    ) |>
    add_data(; data = data3d) |>
    initialize
# neg_var = -2.0 * var
# ClimaAnalysis.set_short_name!(neg_var, "neg_hi")

covar_estimator = ObservationRecipe.ScalarCovariance(; scalar = 1.0)

start_date = Dates.DateTime(2007, 12)
end_date = start_date + Dates.Second(time[end])
sample_collection = SampleBuilder.build_samples_by_times(
    [var],
    [(start_date, end_date)];
    FT = Float64,
)
obs = ObservationRecipe.observation(covar_estimator, sample_collection, 1)

obs_series = EKP.ObservationSeries(
    Dict(
        "observations" => [obs],
        "names" => ["1"],
        "minibatcher" => ClimaCalibrate.minibatcher_over_samples([1], 1),
    ),
)

prior = constrained_gaussian("pi_groups_coeff", 1.0, 0.3, 0, Inf)


ekp = EKP.EnsembleKalmanProcess(
    obs_series,
    EKP.TransformUnscented(prior, impose_prior = true),
    verbose = true,
    scheduler = EKP.DataMisfitController(on_terminate = "continue"),
)

iters = 3
for _ in 1:iters
    g_ens_builder = EnsembleBuilder.GEnsembleBuilder(ekp)
    for i in 1:EKP.get_N_ens(ekp)
        v = deepcopy(var)
        v.data .+= i - 2
        v.data .+= 0.1 * randn(size(v.data))
        EnsembleBuilder.fill_g_ens_col!(g_ens_builder, i, v)
    end
    g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
    EKP.update_ensemble!(ekp, g_ens)
end
```

```@example plot
import ClimaCalibrate
# To use this extension, one of the Makie backends should be loaded
import CairoMakie

fig = CairoMakie.Figure()
ax = CairoMakie.Axis(
    fig[1, 1],
    title = "G ensemble members, mean forward map evaluation, and observations",
    xlabel = "Index",
    ylabel = "Value",
)
g_plot = ClimaCalibrate.Visualization.plot_g!(
    ax,
    ekp;
    iter = 2,
    color = :black,
    alpha = 0.2,
)
g_mean_plot =
    ClimaCalibrate.Visualization.plot_g_mean!(ax, ekp; iter = 2, color = :black)
obs_plot =
    ClimaCalibrate.Visualization.plot_obs!(ax, ekp; iter = 2, color = :blue)

CairoMakie.Legend(
    fig[1, 2],
    [g_plot, g_mean_plot, obs_plot],
    ["G", "G mean", "Observation"],
)

fig
```

## Residual diagnostics

Plotting the ensemble tells you whether it is approaching the observation.
[`analyze_residual`](@ref) tells you something more specific: how much of the
remaining residual is *structured*, meaning aligned with the leading directions
of the observational noise covariance, rather than noise-like.

```julia
import ClimaAnalysis   # required
result = ClimaCalibrate.analyze_residual(ekp, iteration; n_eigenvectors = 3)
result.structured_energy               # fraction of the residual in those directions
result.structured_energy_by_variable   # the same, per variable
result.residual_norm_by_variable       # which variable dominates the residual
```

It projects `obs - mean(G)` onto the leading eigenvectors of the noise
covariance, normalized by the corresponding eigenvalues, so the projections are
z-scores: a value much larger than one means the residual has structure the
noise model does not account for. A high structured energy in one variable
points at that variable's observation or its part of the observation map.

This requires ClimaAnalysis to be loaded, and observations built by
[`ObservationRecipe`](@ref), since it uses their metadata to attribute the
residual to individual variables. It supports `SVD`, `Diagonal`, and `SVDplusD`
covariances.

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

To plot the mean forward map evaluation, columns of the G ensemble matrix, the
true observation, and the normalized residual, you can use
[`Visualization.plot_g_mean`](@ref), [`Visualization.plot_g`](@ref),
[`Visualization.plot_obs`](@ref), and [`Visualization.plot_residual`](@ref)
respectively. The mutating versions also exist as
[`Visualization.plot_g_mean!`](@ref), [`Visualization.plot_g!`](@ref),
[`Visualization.plot_obs!`](@ref), and
[`Visualization.plot_residual!`](@ref). All plotting functions takes an
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

We can also plot the normalized residual `(mean(G) - obs) / σ` from the second
iteration, where `σ` is the square root of the diagonal of the observation
noise covariance.

```@example plot
fig = CairoMakie.Figure()
ClimaCalibrate.Visualization.plot_residual(fig[1, 1], ekp; iter = 2)
fig
```

## Interpreting the residual

Each entry of the normalized residual measures the mismatch between the mean
forward map evaluation and the observation in units of the observation noise
standard deviation, so it can be read like a z-score.

- **Sign:** A positive value means the mean forward map evaluation over-predicts
  the observation at that index (positive bias), and a negative value means it
  under-predicts (negative bias). Note that this is the opposite sign convention
  from [`analyze_residual`](@ref), which uses `obs - mean(G)`.
- **Magnitude:** Values much larger than ``\pm 2`` in magnitude indicate a
  mismatch that the noise model cannot explain.
- **RMS as a summary:** The root mean square (RMS) of the residual is a useful
  single-number summary. RMS much greater than 1 means learnable signal remains
  (or the noise covariance is too small). RMS near 1 means the calibration has
  fit to the noise floor and can no longer distinguish model error from the
  noise it was told to expect. RMS much less than 1 means the noise covariance
  is too large.
- **Index order:** The x-axis is the index into the stacked observation vector
  for that iteration. If you are using the ClimaAnalysis extension, you can use
  [`ObservationRecipe.reconstruct_residual`](@ref) to reconstruct the residual
  as `OutputVar`s and plot them with the ClimaAnalysis plotting functions.
- **Across iterations:** As the calibration converges, the residual should
  shrink toward the noise level, with most entries settling within ``\pm 2``
  and the RMS approaching 1.

!!! warning "Correlated noise"
    The residual is normalized by only the diagonal of the observation noise
    covariance, so correlations between entries are ignored. For an analysis
    that accounts for the structure of the covariance, see
    [`analyze_residual`](@ref).

!!! note "`NaN`s in the ensemble"
    With `ignore_nan = true` (the default), the mean forward map evaluation at
    each index is computed over the ensemble members that are not `NaN`, so
    iterations with different numbers of failed members average over different
    ensemble sizes.

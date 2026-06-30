```@meta
CurrentModule = ClimaCalibrate
```

# How-to guide and cookbook

!!! note "Abbreviation"
    We use `EKP` as shorthand for `EnsembleKalmanProcesses`.

```@setup reconstruct
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
var =
    TemplateVar() |>
    add_dim("time", time, units = "s") |>
    add_dim("lon", lon, units = "degrees") |>
    add_dim("lat", lat, units = "degrees") |>
    add_attribs(
        short_name = "hi",
        long_name = "hello",
        start_date = "2007-12-1",
        blah = "blah2",
    ) |>
    one_to_n_data(collected = true) |>
    initialize
neg_var = -2.0 * var
ClimaAnalysis.set_short_name!(neg_var, "neg_hi")

covar_estimator = ObservationRecipe.ScalarCovariance(; scalar = 1.0)

start_date = Dates.DateTime(2007, 12)
end_date = start_date + Dates.Second(time[end])
osc = SampleBuilder.choose_obs(
    SampleBuilder.build_samples_by_times(
        [var, neg_var],
        [(start_date, end_date)];
        FT = Float64,
    ),
    1,
)
obs = ObservationRecipe.observation(covar_estimator, osc)

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
        EnsembleBuilder.fill_g_ens_col!(g_ens_builder, i, Float64(i))
    end
    g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
    EKP.update_ensemble!(ekp, g_ens)
end
```

## How do I reconstruct the observations as `OutputVar`s?

If you are using `ObservationRecipe`, you can reconstruct an `EKP.Observation`
with [`ObservationRecipe.reconstruct_vars`](@ref).

```@example reconstruct
import ClimaCalibrate: ObservationRecipe
import ClimaAnalysis

# obs is a `EKP.Observation`
# Return a vector of OutputVar
vars = ObservationRecipe.reconstruct_vars(obs)

# obs_series is a `EKP.ObservationSeries`
# Return a vector of OutputVars reconstructed from the first observation in the
# series
vars =
    obs_series |>
    EKP.get_observations |>
    first |>
    ObservationRecipe.reconstruct_vars

# ekp is an `EKP.EnsembleKalmanProcess` object
# Same result as the example above
vars =
    ekp |>
    EKP.get_observation_series |>
    EKP.get_observations |>
    first |>
    ObservationRecipe.reconstruct_vars

nothing # hide
```

## How do I reconstruct the G ensemble matrix as `OutputVar`s?

If you are using `ObservationRecipe`, you can reconstruct the G ensemble matrix
as `OutputVar`s with [`ObservationRecipe.reconstruct_g`](@ref),
[`ObservationRecipe.reconstruct_g_mean`](@ref), and
[`ObservationRecipe.reconstruct_g_mean_final`](@ref).

```@example reconstruct
import ClimaCalibrate: ObservationRecipe
import ClimaAnalysis

# ekp is an `EKP.EnsembleKalmanProcess` object

# Return a matrix of `OutputVar`s reconstructed from the G ensemble matrix
# at the 2nd iteration
G_ens_mat_as_vars = ObservationRecipe.reconstruct_g(ekp, 2)

# Return a vector of `OutputVar` reconstructed from the mean forward map
# evaluation at the 3rd iteration
g_mean_as_vars = ObservationRecipe.reconstruct_g_mean(ekp, 3)

# Return a vector of `OutputVar` reconstructed from the final mean forward map
# evaluation at the final iteration
ObservationRecipe.reconstruct_g_mean_final(ekp)

nothing # hide
```

## I have a diagonal covariance matrix. How can I reconstruct this as a `OutputVar`?

If you created the diagonal covariance matrix with `ObservationRecipe`, you can
reconstruct with [`ObservationRecipe.reconstruct_diag_cov`](@ref) where `obs` is
a `EKP.Observation`.

```@example reconstruct
import ClimaCalibrate: ObservationRecipe
import ClimaAnalysis

ObservationRecipe.reconstruct_diag_cov(obs)

nothing # hide
```

# How do I handle `NaN`s in the `OutputVar`s so that there are no `NaN`s in the sample and covariance matrix?

`NaN`s should be handled when preprocessing the data. In some cases,
there will be `NaN`s in the data (e.g. calibrating with data that is valid only
over land). In these cases, the `SampleBuilder` module automatically removes
`NaN`s from the data when building the samples, since `ClimaAnalysis.flatten`
drops `NaN`s while flattening each `OutputVar`. It is important to ensure that
across the time slices, the `NaN`s appear in the same coordinates of the
non-temporal dimensions. For example, if the quantity is defined over the
dimensions longitude, latitude, and time, then any slice of the data at a
particular longitude and latitude should either only contain `NaN`s or no `NaN`s
at all. The `SampleBuilder` module checks that the dropped coordinates match
across samples and errors otherwise.

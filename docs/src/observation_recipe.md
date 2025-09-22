# ObservationRecipe

!!! warning
    To enable this module, use `using ClimaAnalysis` or `import
    ClimaAnalysis`.

When handling weather and climate data, it can be tedious and error-prone when
setting up the observation for calibration with `EnsembleKalmanProcesses` (or
`EKP` for short). As such, ClimaCalibrate provides recipes for setting up
observations consisting of samples, noise covariances, names, and metadata.

## How do I use this to set up observation for calibration with EKP?

All functions assume that any data preprocessing is done with `ClimaAnalysis`.

### Covariance Estimators

There are currently two covariance estimators, `ScalarCovariance`,
`SeasonalDiagonalCovariance`, and `SVDplusDCovariance`, which are subtypes of
`AbstractCovarianceEstimator`. `ScalarCovariance` approximates the observation
noise covariance as a scalar diagonal matrix. `SeasonalDiagonalCovariance`
approximates the observation noise covariance as a diagonal of variances across
all the seasons for each observation, neglecting correlations between points.
`SVDplusDCovariance` additionally approximates the correlations between points
from, often limited, time series observations.

### Necessary data preprocessing

The `OutputVar`s should represent **time series data of summary statistics**.
For example, to compute seasonal averages of a `OutputVar`, one can use
`ClimaAnalysis.average_season_across_time`, which will produce a `OutputVar`
that can be used with either `SeasonalDiagonalCovariance` or
`SVDplusDCovariance`.

```julia
import ClimaAnalysis

obs_var = ClimaAnalysis.OutputVar(
    "precip.mon.mean.nc",
    "precip",
    new_start_date = start_date,
    shift_by = Dates.firstdayofmonth,
)

# -- preprocessing for units, times, grid, etc. --

seasonal_averages = ClimaAnalysis.average_season_across_time(obs_var)
```

### Observation

After preprocessing the `OutputVar`s so that they represent time series data of
summary statistics, one can use set up an `EKP.observation` as shown below.

```julia
import ClimaAnalysis
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe

# Vars are OutputVars preprocessed to ensure consistent units, times,
# and grid as the diagonstics produced from the model.
# In this example, we want to calibrate with seasonal averages, so we use
# ClimaAnalysis.average_season_across_time
vars = ClimaAnalysis.average_season_across_time.(vars)

# We want the covariance matrix to be Float32, so we change it here.
vars = ObservationRecipe.change_data_type.(vars, Float32)

# We choose SVDplusDCovariance. We need to supply the start and end dates of
# the samples with `sample_date_ranges`. To do this, we can use the function
# below. In this example, the dates in `vars` are all the same. For debugging,
# it is helpful to use `ClimaAnalysis.dates(var)`.
sample_date_ranges =
    ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges(first(vars))
covar_estimator = SVDplusDCovariance(
    sample_date_ranges,
    model_error_scale = Float32(0.05),
    regularization = Float32(1e-6),
)

# Finally, we form the observation
start_date = sample_date_ranges[1][1]
end_date = sample_date_ranges[1][2]
obs = ObservationRecipe.observation(
    covar_estimator,
    vars,
    start_date,
    end_date,
)
```

## Metadata

!!! note
    Metadata in `EKP.observation` is only added with versions of
    EnsembleKalmanProcesses later than v2.4.2.

When creating an observation with [`ObservationRecipe.observation`](@ref
ClimaCalibrate.ObservationRecipe.observation), metadata is extracted from the
`OutputVar`s and attached to the observation. The metadata for each observation
can be accessed with `EKP.get_metadata(obs::EKP.Observation)` and the metadata
for each iteration can be accessed with
[`ObservationRecipe.get_metadata_for_nth_iteration`](@ref
ClimaCalibrate.ObservationRecipe.get_metadata_for_nth_iteration). The metadata
can be used with `ClimaAnalysis.unflatten` to reconstruct the original
`OutputVar` before flattening. See the ClimaAnalysis
[documentation](https://clima.github.io/ClimaAnalysis.jl/dev/api/#FlatVar) about
`ClimaAnalysis.FlatVar` for more information.

`ObservationRecipe` provides a helper function for reconstructing the mean
forward map evaluation with [`ObservationRecipe.reconstruct_g_mean_final`](@ref
ClimaCalibrate.ObservationRecipe.reconstruct_g_mean_final).

## Frequently asked questions

**Q: I need to compute `g_ensemble` and I do not know how the data of the `OutputVar`s is flattened.**

**A:** When forming the sample, the data in a `OutputVar` is flattened using
`ClimaAnalysis.flatten`. See
[`ClimaAnalysis.flatten`](https://clima.github.io/ClimaAnalysis.jl/dev/flat/#Flatten)
in the ClimaAnalysis documentation for more information. The order of the
variables in the observation is the same as the order of the `OutputVar`s when
creating the `EKP.Observation` using `ObservationRecipe.observation`.

**Q: How do I handle `NaN`s in the `OutputVar`s so that there are no `NaN`s in the sample and covariance matrix?**

**A:** `NaN`s should be handled when preprocessing the data. In some cases,
there will be `NaN`s in the data (e.g. calibrating with data that is valid only
over land). In these cases, the functions for making observations will
automatically remove `NaN`s from the data. It is important to ensure that across
the time slices, the `NaN`s appear in the same coordinates of the non-temporal
dimensions. For example, if the quantity is defined over the dimensions
longitude, latitude, and time, then any slice of the data at a particular
longitude and latitude should either only contain `NaN`s or no `NaN`s at all.

**Q: How is the name of the observation determined?**

**A:** The name of the observation is determined by the short name in the
attributes of the `OutputVar`. If there are multiple `OutputVar`s, then the name
is all the short names separated by semicolons. If no short name is found, then
the name will be `nothing`.

**Q: What is `regularization` and `model_error_scale` when making a covariance matrix?**

**A:** The model error scale and regularization terms are used to inflate the
diagonal of the observation covariance matrix to reflect estimates of
measurement error. You can add a fixed percentage inflation of the noise due to
the model error to the covariance matrix with the `model_error_scale` keyword
argument. Additionally, to prevent very small variance along the diagonal of the
covariance matrix, you can add a regularization with the `regularization`
keyword argument.

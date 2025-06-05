# ObservationRecipe

!!! warning
    To enable this extension, use `using ClimaAnalysis` or `export
    ClimaAnalysis`.

When handling weather and climate data, it can be tedious and error-prone when
setting up the observation for calibration with `EnsembleKalmanProcesses` (or
`EKP` for short). As such, ClimaCalibrate provides recipes for setting up
observations consisting of samples, noise covariances, names, and metadata.

## How do I use this to set up observation for calibration with EKP?

All functions assumes that any data preprocessing is done with `ClimaAnalysis`.

### Covariance Estimators

There are currently two covariance estimators which are
`SeasonalDiagonalCovariance` and `SVDplusDCovariance` which are subtypes of
`AbstractCovarianceEstimator`.

For `SeasonalDiagonalCovariance`, the time series data of seasonal quantities
should be used, while for `SVDplusDCovariance`, any time series data of any
summary statistics can be used. Look at the documentation of these functions for
more information.

### Recommendations for data preprocessing

The `OutputVar`s should represent time series data of summary statistics. For
example, to compute seasonal averages of a `OutputVar`, one can use
`ClimaAnalysis.average_season_across_time`, which will produce a `OutputVar`
that can be used with either `SeasonalDiagonalCovariance` or
`SVDplusDCovariance`.

```julia
import ClimaAnalysis
ClimaAnalysis.average_season_across_time(var)
```

### Observation

After preprocessing the `OutputVar`s, so that they represent time series data of
summary statistics, one can use set up an `EKP.observation` as shown below.

```julia
import ClimaAnalysis
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe

# vars are OutputVars preprocessed with ensure consistency with units, times,
# and dimensions as the diagonstics produced from the model.
# In this example, we want to calibrate with seasonal averages, so we use
# ClimaAnalysis.average_season_across_time
vars = ClimaAnalysis.average_season_across_time.(vars)

# We choose SVDplusDCovariance. We need to supply the start and end dates of
# the samples with `sample_dates`. It is helpful to use
# `ClimaAnalysis.dates(var)` to find the correct dates.
sample_dates = [
        (Dates.DateTime(i, 12, 1), Dates.DateTime(i + 1, 9, 1)) for
        i in 2007:2009
    ]
covar_estimator = SVDplusDCovariance(;
    model_error_scale = 0.05,
    regularization = 1e-6,
    sample_dates = sample_dates,
)

# Finally, we form the observation
start_date = sample_dates[1][1]
end_date = sample_dates[1][2]
obs = ObservationRecipe.observation(covar_estimator, vars, start_date = start_date, end_date = end_date)
```

## Common questions

**Q: I need to compute `g_ensemble` and I do not know how the data of the `OutputVar`s are flattened.**

**A:** When forming the sample, the data in a `OutputVar` is flattened using
`ClimaAnalysis.flatten`. See the ClimaAnalysis documentation for
[`ClimaAnalysis.flatten`](https://clima.github.io/ClimaAnalysis.jl/dev/flat/#Flatten)
for more information.

**Q: How do I handle `NaN`s in the `OutputVar`, so that there are no `NaN`s in the sample and covariance matrix?**

**A:** `NaN`s should be handled when preprocessing the data. In some cases,
there will be `NaN`s in the data (e.g. calibrating data only over land). In
these cases, the functions for making observations will automatically remove
`NaN`s from the data. If `NaN`s are not ignored (e.g. with
`SVDplusDCovariance`), then it is important to ensure that across the time
slices, the `NaN`s appear in the same coordinates of the dimensions that are not
time.

**Q: How is the name in the observation determined?**

**A:** The name of the observation is determined by the short name in the
attributes of the `OutputVar`. If there are multiple `OutputVar`s, then the name
is all the short names separated by semicolons. If no short name is found, then
the name will be `nothing`.

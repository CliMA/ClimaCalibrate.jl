```@meta
CurrentModule = ClimaCalibrate.ObservationRecipe
```

# ObservationRecipe

!!! warning
    If you are not using ClimaAnalysis, you can skip this page.

!!! note
    To enable this module, use `using ClimaAnalysis` or `import
    ClimaAnalysis`.

!!! note "Prerequisites"
    It is recommended that you read the "Sample Builder" section before reading
    this section.

When handling weather and climate data, it can be tedious and error-prone when
setting up the observation for calibration with `EnsembleKalmanProcesses` (or
`EKP` for short). As such, ClimaCalibrate provides recipes for estimating the
noise covariance matrix and packaging it, together with the observation and its
metadata, into an `EKP.Observation`.

We start with an [`ObservedSampleCollection`](@ref
ClimaCalibrateClimaAnalysisExt.ObservedSampleCollection), which the
[`SampleBuilder`](sample_builder.md) module produces by flattening one or more
`ClimaAnalysis.OutputVar`s into a matrix of samples and letting you pick one
column as the observation. `ObservationRecipe` then takes that
`ObservedSampleCollection`, estimates the noise covariance matrix from the
samples, and builds the `EKP.Observation` used in the calibration.

## How do I use this to set up observation for calibration with EKP?

All functions assume that any data preprocessing is done with `ClimaAnalysis`.

### Covariance Estimators

There are currently three covariance estimators, [`ScalarCovariance`](@ref),
[`SeasonalDiagonalCovariance`](@ref), and [`SVDplusDCovariance`](@ref), which
are subtypes of [`AbstractCovarianceEstimator`](@ref). Each estimates the noise
covariance matrix from the matrix of samples in an `ObservedSampleCollection`.
`ScalarCovariance` approximates the observation noise covariance as a scalar
diagonal matrix. `SeasonalDiagonalCovariance` approximates the observation noise
covariance as a diagonal of variances across all the seasons for each
observation, neglecting correlations between points. `SVDplusDCovariance`
additionally approximates the correlations between points from, often limited,
time series observations. Because `SeasonalDiagonalCovariance` and
`SVDplusDCovariance` estimate variances and correlations across samples, they
require at least two samples (columns), whereas `ScalarCovariance` works with a
single sample.

### Necessary data preprocessing

In most cases, the `OutputVar`s represent **time series data of summary
statistics**. For example, to compute seasonal averages of a `OutputVar`, one
can use `ClimaAnalysis.average_season_across_time`, which will produce a
`OutputVar` that can be used with either `SeasonalDiagonalCovariance` or
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
summary statistics, you build the samples with functions provided by the
`SampleBuilder` module and pass the resulting `ObservedSampleCollection` to
[`ObservationRecipe.observation`](@ref observation), as shown below. See [Sample
Builder](sample_builder.md) for the details of [`build_samples_by_times`](@ref
ClimaCalibrate.SampleBuilder.build_samples_by_times) and [`choose_obs`](@ref
ClimaCalibrate.SampleBuilder.choose_obs).

```julia
import ClimaAnalysis
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.SampleBuilder

# Vars are OutputVars preprocessed to ensure consistent units, times,
# and grid as the diagnostics produced from the model.
# In this example, we want to calibrate with seasonal averages, so we use
# ClimaAnalysis.average_season_across_time
vars = ClimaAnalysis.average_season_across_time.(vars)

# We need the start and end dates of each sample. To find these, we can use the
# function below. In this example, the dates in `vars` are all the same. For
# debugging, it is helpful to use `ClimaAnalysis.dates(var)`.
sample_date_ranges =
    ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges(first(vars))

# Build the samples (one per date range) and choose the first one as the
# observation used in the calibration.
samples =
    SampleBuilder.build_samples_by_times(
        vars,
        sample_date_ranges;
        FT = Float32
    )
osc = SampleBuilder.choose_obs(samples, 1)

# We choose SVDplusDCovariance.
covar_estimator = ObservationRecipe.SVDplusDCovariance(
    model_error_scale = Float32(0.05),
    regularization = Float32(1e-6),
)

# Finally, we form the observation
obs = ObservationRecipe.observation(covar_estimator, osc)
```

## Metadata

!!! note
    Metadata in `EKP.observation` is only added with versions of
    EnsembleKalmanProcesses later than v2.4.2.

When creating an observation with [`observation`](@ref), metadata is extracted
from the `OutputVar`s and attached to the observation. The metadata for each
observation can be accessed with `EKP.get_metadata(obs::EKP.Observation)` and
the metadata for each iteration can be accessed with
[`ClimaCalibrate.get_metadata_for_nth_iteration`](@ref
ClimaCalibrate.EKPUtils.get_metadata_for_nth_iteration). The metadata can be
used with `ClimaAnalysis.unflatten` to reconstruct the original `OutputVar`
before flattening. See the ClimaAnalysis
[documentation](https://clima.github.io/ClimaAnalysis.jl/dev/api/#FlatVar) about
`ClimaAnalysis.FlatVar` for more information.

## Debugging observational and simulation data

When setting up a calibration, it may be helpful to visualize the
`EKP.Observation`s or inspect the observational data and metadata together. To
help with this, `ObservationRecipe` provides several functions that reconstruct
the underlying data back into `OutputVar`s:

- [`reconstruct_vars`](@ref) reconstructs the samples of an `EKP.Observation` as
  a vector of `OutputVar`s.
- [`reconstruct_diag_cov`](@ref) reconstructs the diagonal of the covariance
  matrix of an `EKP.Observation` as a vector of `OutputVar`s (only supported for
  diagonal covariance matrices).
- [`reconstruct_g`](@ref) reconstructs the G ensemble matrix of the `it`th
  iteration as a matrix of `OutputVar`s.
- [`reconstruct_g_mean`](@ref) reconstructs the mean forward model evaluation of
  the `it`th iteration as a vector of `OutputVar`s.
- [`reconstruct_g_mean_final`](@ref) reconstructs the mean forward model
  evaluation of the last iteration as a vector of `OutputVar`s.

```julia
# obs is an EKP.Observation
# ekp is the EKP.EnsembleKalmanProcess
# it is the iteration index

ObservationRecipe.reconstruct_vars(obs)
# Reconstructing the diagonal of a covariance matrix as an OutputVar is only
# supported for diagonal covariance matrices
ObservationRecipe.reconstruct_diag_cov(obs)
ObservationRecipe.reconstruct_g(ekp, it)
ObservationRecipe.reconstruct_g_mean(ekp, it)
ObservationRecipe.reconstruct_g_mean_final(ekp)
```

## Creating custom covariance estimators

In the cases where the provided covariance estimators are not sufficient, it is
possible to create your own covariance estimator using the functionality
provided by ClimaCalibrate and ClimaAnalysis.

The steps are:

1. Define a struct that subtypes
   [`ObservationRecipe.AbstractCovarianceEstimator`](@ref
   AbstractCovarianceEstimator). Any fields of this struct will be available
   when implementing the `covariance` method.
2. Implement a method of [`covariance`](@ref) that dispatches on your struct and
   an `ObservedSampleCollection`, with the signature
   `ObservationRecipe.covariance(estimator::YourType, osc)`. It must return a
   noise covariance matrix.

### What you can use

An `ObservedSampleCollection` (the `osc` argument) stores a matrix of samples,
a matrix of metadata, and the observation. Here's a collection of functions
from the `SampleBuilder` module that could be helpful when creating the
covariance matrix.

- [`get_samples`](@ref ClimaCalibrate.SampleBuilder.get_samples): return the
  full sample matrix.
- [`get_obs`](@ref ClimaCalibrate.SampleBuilder.get_obs): return the chosen
  observation, one column of that matrix.
- [`get_obs_metadata`](@ref ClimaCalibrate.SampleBuilder.get_obs_metadata):
  return a vector of `ClimaAnalysis.Var.Metadata` for every variable in the
  observation.
- [`get_metadata`](@ref ClimaCalibrate.SampleBuilder.get_metadata): return the
  full metadata matrix.
- [`num_samples`](@ref ClimaCalibrate.SampleBuilder.num_samples): return the
  number of samples.

The covariance you return must be a square matrix whose side length equals
`length(get_obs(osc))` and must not contain `NaN` or `Inf`.

### Example: per-variable constant variance

The following estimator gives each variable its own constant variance, filling
that variable's block of a diagonal covariance matrix.

```@setup custom_cov
import ClimaAnalysis
import ClimaAnalysis.Template:
    TemplateVar, add_dim, add_attribs, one_to_n_data, initialize
import ClimaCalibrate: ObservationRecipe, SampleBuilder
import EnsembleKalmanProcesses as EKP

pr_lat = [-90.0, 90.0]
pr_time = [0.0, 1.0, 2.0]
rsut_lat = [-90.0, 0.0, 90.0]
rsut_time = [1.0]
pr_var =
    TemplateVar() |>
    add_dim("time", pr_time, units = "s") |>
    add_dim("lat", pr_lat, units = "degrees") |>
    add_attribs(short_name = "pr", start_date = "2008-1-1", units = "mm/day") |>
    one_to_n_data(collected = true) |>
    initialize
rsut_var =
    TemplateVar() |>
    add_dim("time", rsut_time, units = "s") |>
    add_dim("lat", rsut_lat, units = "degrees") |>
    add_attribs(short_name = "rsut", start_date = "2008-1-1", units = "W m-2") |>
    one_to_n_data(collected = true) |>
    initialize
# A single sample built from the two variables.
osc = SampleBuilder.choose_obs(
    SampleBuilder.build_samples([pr_var, rsut_var]; FT = Float32),
    1,
)
```

```@example custom_cov
import ClimaAnalysis
import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.SampleBuilder
import LinearAlgebra: Diagonal

# One variance per variable, in the same order as the OutputVars used to build
# the samples.
struct PerVariableScalar <: ObservationRecipe.AbstractCovarianceEstimator
    variances::Vector{Float64}
end

function ObservationRecipe.covariance(estimator::PerVariableScalar, osc)
    FT = eltype(SampleBuilder.get_samples(osc))
    metadata = SampleBuilder.get_obs_metadata(osc)
    # Repeat each variance over its variable's flattened block, then stack.
    diag_cov = reduce(
        vcat,
        fill(FT(v), ClimaAnalysis.flattened_length(m)) for
        (v, m) in zip(estimator.variances, metadata)
    )
    return Diagonal(diag_cov)
end
nothing # hide
```

Using it is the same as for the built-in estimators:

```@repl custom_cov
osc
estimator = PerVariableScalar([0.1, 0.5])
obs = ObservationRecipe.observation(estimator, osc);
EKP.get_covs(obs)
```

## Frequently asked questions

**Q: I need to compute `g_ensemble` and I do not know how the data of the `OutputVar`s is flattened.**

**A:** When forming the sample, the data in a `OutputVar` is flattened using
`ClimaAnalysis.flatten`. See
[`ClimaAnalysis.flatten`](https://clima.github.io/ClimaAnalysis.jl/dev/flat/#Flatten)
in the ClimaAnalysis documentation for more information. The order of the
variables in the observation is the same as the order of the `OutputVar`s when
creating the `EKP.Observation` using `ObservationRecipe.observation`. If you are
using `ObservationRecipe`, it is recommended that you also use
[`GEnsembleBuilder`](ensemble_builder.md) which simplifies building the
G ensemble matrix.

**Q: How is the name of the observation determined?**

**A:** By default, the name of the observation is determined by the short name
in the attributes of the `OutputVar`. If there are multiple `OutputVar`s, then
the name is all the short names separated by semicolons. If no short name is
found, then the name will be `nothing`. You can override this by passing the
`name` keyword argument to `ObservationRecipe.observation`.

**Q: What is `regularization` and `model_error_scale` when making a covariance matrix?**

**A:** The model error scale and regularization terms are used to inflate the
diagonal of the observation covariance matrix to reflect estimates of
measurement error. You can add a fixed percentage inflation of the noise due to
the model error to the covariance matrix with the `model_error_scale` keyword
argument. Additionally, to prevent very small variance along the diagonal of the
covariance matrix, you can add a regularization with the `regularization`
keyword argument. For `SVDplusDCovariance`, the `regularization` keyword
argument can also be a [`QuantileRegularization`](@ref), which sets the
regularization from a quantile of the model error scale instead of a fixed
value.

**Q: How do I apply latitude weighting to the covariance matrix?**

**A:** All three covariance estimators accept a `use_latitude_weights` keyword
argument. This accounts for the varying area of grid cells with latitude. The
`min_cosd_lat` keyword argument (default `0.1`) sets the minimum value of
`cosd(lat)` used in the weight, which prevents very small values along the
diagonal that can cause issues when inverting the covariance matrix. This
requires the `OutputVar`s to have a latitude dimension.

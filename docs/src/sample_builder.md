```@meta
CurrentModule = ClimaCalibrate.SampleBuilder
```

# SampleBuilder

!!! warning
    If you are not using ClimaAnalysis, you can skip this page.

!!! note
    To enable this module, use `using ClimaAnalysis` or `import
    ClimaAnalysis`.

## Why use SampleBuilder?

For calibration of climate models using EnsembleKalmanProcesses.jl and other
machine learning algorithms, observations are represented as a vector of numbers
along with a covariance matrix describing the uncertainty of the observation.
However, the diagnostics of climate models are typically in the form of gridded
fields with named dimensions such as latitude, longitude, and time.

The `SampleBuilder` module bridges this gap. It flattens one or more
`ClimaAnalysis.OutputVar`s into a matrix of samples along with their associated
metadata and packages the result as a [`SampleCollection`](@ref
ClimaCalibrateClimaAnalysisExt.SampleCollection). From that matrix you pick one
column as the true observation to create an [`ObservedSampleCollection`](@ref
ClimaCalibrateClimaAnalysisExt.ObservedSampleCollection). The
`ObservationRecipe` module takes this output and estimates the covariance matrix
from the matrix of samples and uses it to build an `EKP.Observation`.

## How does it work?

The main entry point is [`build_samples`](@ref) which takes a single
`OutputVar`, a vector of `OutputVar`s, or a matrix of `OutputVar`s and produces
a `SampleCollection` from it. For a single `OutputVar`, this is interpreted as a
single sample consisting of a single `OutputVar`, and for a vector of
`OutputVar`s, this is interpreted as a single sample consisting of multiple
`OutputVar`s. You can think of the latter case as the vector being a column
vector.

```math
\begin{pmatrix}
\mathrm{OV}_{11} & \cdots & \mathrm{OV}_{1m} \\
\vdots & \ddots & \vdots \\
\mathrm{OV}_{n1} & \cdots & \mathrm{OV}_{nm}
\end{pmatrix}
\xrightarrow{\mathtt{build\_samples}}
\left(\begin{array}{ccc}
\mathbf{v}_{11} & \cdots & \mathbf{v}_{1m} \\
\vdots & \ddots & \vdots \\
\mathbf{v}_{n1} & \cdots & \mathbf{v}_{nm}
\end{array}\right)
```

where ``\mathrm{OV}_{ij}`` is an `OutputVar` and
``\mathbf{v}_{ij} = \mathtt{flatten}(\mathrm{OV}_{ij})`` is a column vector.

The rows of the input matrix correspond to `OutputVar`s of the same kind and the
columns correspond to samples. The function `build_samples` flattens each
`OutputVar` into a column vector of floats by calling `flatten` in a fixed
dimension order. Each column of the result is a sample which is the vertical
concatenation of the flattened vectors from every `OutputVar` in that column.
The final `SampleCollection` stores this as a single numeric matrix with the
associated `Metadata` for each `OutputVar` kept separately.

As of now, the matrix of samples is guaranteed not to have `NaN`s. When `NaN`s
are in the `OutputVar`s, `ClimaAnalysis.flatten` automatically removes `NaN`s.
Furthermore, for each row of `OutputVar`s, for dimensions that are not ignored,
`build_samples` checks that
1. the short names are the same,
2. the flattened vector sizes are the same,
3. the units are the same,
4. the dimensions between both are the same,
5. the number of dimensions are the same,
6. the dimension units are the same,
7. the dimension values are the same,
8. the coordinates where the NaNs are dropped are the same.

You can exclude dimensions from these checks with the `ignore_dims` keyword
argument. This is useful when the samples are meant to differ along a dimension.
For example, [`build_samples_by_times`](@ref) ignores the time dimension because
each sample covers a different time range.

After a `SampleCollection` is created, you can choose a column of the matrix of
samples to be the observation for the calibration with [`choose_obs`](@ref).
This creates an `ObservedSampleCollection`.

## Examples

```@setup samples
import ClimaAnalysis
import ClimaAnalysis.Template:
    TemplateVar, add_dim, add_attribs, one_to_n_data, initialize
import ClimaCalibrate: SampleBuilder

# Keep the dimensions small so that each sample stays short
lat = [-90.0, 90.0]
time = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# Two kinds of variables: precipitation (pr) and upwelling shortwave (rsut)
pr_var =
    TemplateVar() |>
    add_dim("time", time, units = "s") |>
    add_dim("lat", lat, units = "degrees") |>
    add_attribs(short_name = "pr", start_date = "2008-1-1", units = "mm/day") |>
    one_to_n_data(collected = true) |>
    initialize
rsut_var =
    TemplateVar() |>
    add_dim("time", time, units = "s") |>
    add_dim("lat", lat, units = "degrees") |>
    add_attribs(short_name = "rsut", start_date = "2008-1-1", units = "W m-2") |>
    one_to_n_data(collected = true) |>
    initialize

# Make a few samples that differ only in their data by shifting the values. The
# short name, dimensions, and units stay the same, which is required for the
# variables in a row to represent the same kind of quantity.
function shift_data(var, by)
    shifted = deepcopy(var)
    shifted.data .= shifted.data .+ by
    return shifted
end
pr1, pr2, pr3 = pr_var, shift_data(pr_var, 10), shift_data(pr_var, 20)
rsut1, rsut2, rsut3 = rsut_var, shift_data(rsut_var, 10), shift_data(rsut_var, 20)
```

Here's an example of using `build_samples` and `choose_obs` to create an
`ObservedSampleCollection`.

```@example samples
# The rows are variables (pr, rsut) and the columns are samples
# Each entry of the input matrix is a OutputVar
var_samples = [pr1   pr2   pr3
               rsut1 rsut2 rsut3]

sample_collection = SampleBuilder.build_samples(
    var_samples;
    ignore_dims = ("time", ), # ignore checking the time dimension
    FT = Float32 # element type of matrix of samples will be Float32
)

# Choose the first sample (the first column) to use as the observation
osc = SampleBuilder.choose_obs(sample_collection, 1)
```

In addition to this, ClimaCalibrate also provides
[`build_samples_by_times`](@ref) which deals with the common case of generating
samples from `OutputVar`s that represent time series data. In this case, we want
to window the `OutputVar`s by time ranges, so that each sample typically
represents a single year of data. For this example, we use short time windows.

```@example samples
# Each time range becomes one sample, with every variable windowed to that range
SampleBuilder.build_samples_by_times(
    [pr_var, rsut_var],
    [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)], # this also works with Dates.DateTime
)
```

There are also getter functions such as [`get_obs`](@ref) for getting the
observation for the calibration, [`get_obs_metadata`](@ref) for getting the
metadata of the observation, [`get_samples`](@ref) for getting the matrix of
samples, [`get_metadata`](@ref) for getting the matrix of metadata.

```@example samples
# The observation as a vector of values
obs = SampleBuilder.get_obs(osc)

# The metadata of each variable in the observation
obs_metadata = SampleBuilder.get_obs_metadata(osc)

# The matrix of samples and its matrix of metadata
samples = SampleBuilder.get_samples(osc)
metadata = SampleBuilder.get_metadata(osc)

nothing # hide
```

Finally, for debugging, you may also find the [`reconstruct_obs`](@ref) and
[`reconstruct_col`](@ref) functions helpful. These functions take a matrix of
samples and metadata and transform them back into a vector of `OutputVar`s.

```@example samples
# Reconstruct the chosen observation as a vector of `OutputVar`s
obs_vars = SampleBuilder.reconstruct_obs(osc)

# reconstruct_col does the same for any column of the matrix of samples in a
# SampleCollection
col_vars = SampleBuilder.reconstruct_col(sample_collection, 1)
first(col_vars)
```

## Next steps

Once you have an `ObservedSampleCollection`, you pass it and a covariance
estimator to [`observation`](@ref ClimaCalibrate.ObservationRecipe.observation)
to build the `EKP.Observation` used in the calibration. See [Observation
Recipes](observation_recipe.md) for the available estimators.

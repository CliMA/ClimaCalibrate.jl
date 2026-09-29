```@meta
CurrentModule = ClimaCalibrate.SampleBuilder
```

# Building samples

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
fields with named dimensions such as latitude, longitude, and time. We use
ClimaAnalysis, which stores diagnostics as OutputVars and provides functions for
postprocessing them.

The `SampleBuilder` module bridges this gap. To do this, three steps are
performed:
1. We flatten each `OutputVar` into a vector and store metadata needed to
   unflatten it and check consistency across samples. This is done by
   [`SampleBuilder.build_samples`](@ref build_samples), which constructs a
   [`SampleCollection`](@ref ClimaCalibrateClimaAnalysisExt.SampleCollection).
2. We choose how the covariance matrix of the flattened samples is formed with a
   concrete [`ObservationRecipe.AbstractCovarianceEstimator`](@ref
   ClimaCalibrate.ObservationRecipe.AbstractCovarianceEstimator).
3. We pick one column of the `SampleCollection` as the true observation.

Passing the estimator, the `SampleCollection`, and the index of column to
[`ObservationRecipe.observation`](@ref
ClimaCalibrate.ObservationRecipe.observation) constructs the covariance matrix
and packages it with the chosen column into an `EKP.Observation`.

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
(\mathrm{V}^{(1)})_1 & \cdots & (\mathrm{V}^{(1)})_m \\
\vdots & \ddots & \vdots \\
(\mathrm{V}^{(n)})_1 & \cdots & (\mathrm{V}^{(n)})_m
\end{pmatrix}
\xrightarrow{\mathtt{build\_samples}}
\left(\begin{pmatrix}
(\mathrm{v}^{(1)})_1 \\
\vdots \\
(\mathrm{v}^{(n)})_1
\end{pmatrix}, \dots,
\begin{pmatrix}
(\mathrm{v}^{(1)})_m \\
\vdots \\
(\mathrm{v}^{(n)})_m
\end{pmatrix}
\right)
```

where, for ``i = 1, \ldots, n`` variables and ``j = 1, \ldots, m`` samples,
``(\mathrm{V}^{(i)})_j`` is a `OutputVar` and ``(\mathrm{v}^{(i)})_j =
\mathrm{vec}((\mathrm{V}^{(i)})_j)`` is a column vector.

The rows of the input matrix correspond to `OutputVar`s of the same kind and the
columns correspond to samples. The function `build_samples` flattens each
`OutputVar` into a column vector of floats by calling `flatten` in a fixed
dimension order. Each column of the result is a sample which is the vertical
concatenation of the flattened vectors from all `OutputVar`s in that column.
The final `SampleCollection` stores this as a single numeric matrix with the
associated `Metadata` for each `OutputVar` kept separately.

As of now, the matrix of samples is guaranteed not to have `NaN`s. When `NaN`s
are in the `OutputVar`s, `ClimaAnalysis.flatten` automatically removes `NaN`s.
Furthermore, for each row of `OutputVar`s, for dimensions that are not ignored,
`build_samples` checks that
1. the short names are the same,
2. the flattened vector sizes are the same,
3. the units are the same,
4. the dimensions are the same,
5. the number of dimensions are the same,
6. the dimension units are the same,
7. the dimension values are the same,
8. the coordinates where the NaNs are dropped are the same.

Keep in mind that a covariance estimator may need the values of a dimension to
be the same across the samples. For example, latitude weighting applies the
weights of the first sample to all samples, so the estimators error when the
latitude dimension is ignored and the latitudes differ across the samples.

After a `SampleCollection` is created, you can choose a column of the matrix of
samples to be the observation for the calibration by passing its index to
[`observation`](@ref ClimaCalibrate.ObservationRecipe.observation).

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

Here's an example of using `build_samples` to create a `SampleCollection`. The
samples here are meant to differ along time, so the `ignore_dims` keyword
argument excludes the time dimension from the checks above. Without it,
`build_samples` would reject samples whose time values do not match.

```@example samples
# The rows are variables (pr, rsut) and the columns are samples
# Each entry of the input matrix is a OutputVar
var_samples = [pr1   pr2   pr3
               rsut1 rsut2 rsut3]

sample_collection = SampleBuilder.build_samples(
    var_samples;
    ignore_dims = ("time", ), # ignore checking the time dimension
    FT = Float32 # element type of the samples and their metadata will be Float32
)
```

In addition to this, ClimaCalibrate also provides
[`build_samples_by_times`](@ref) which deals with the common case of generating
samples from `OutputVar`s that represent time series data. In this case, we want
to window the `OutputVar`s by time ranges, so that each sample typically
represents a single year of data. For this example, we use short time windows.

```@example samples
# Each time range becomes one sample, with all variables windowed to that range
SampleBuilder.build_samples_by_times(
    [pr_var, rsut_var],
    [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)], # this also works with Dates.DateTime
)
```

There are also getter functions such as [`get_samples`](@ref) for getting the
matrix of samples and [`get_metadata`](@ref) for getting the matrix of
metadata.

```@example samples
# The matrix of samples and its matrix of metadata
samples = SampleBuilder.get_samples(sample_collection)
metadata = SampleBuilder.get_metadata(sample_collection)

nothing # hide
```

Finally, for debugging, you may also find the [`reconstruct_col`](@ref)
function helpful. This function takes a column of the matrix of samples and its
metadata and transforms them back into a vector of `OutputVar`s.

```@example samples
# Reconstruct the first sample as a vector of `OutputVar`s
col_vars = SampleBuilder.reconstruct_col(sample_collection, 1)
first(col_vars)
```

## Transformations

There are cases where you want to apply a transform, such as per-variable
weighting or latitude weighting, to the samples for estimating the covariance
matrix, but not to the samples when forming the observations. In
ClimaCalibrate, these operations are [`AbstractTransform`](@ref)s.

### Using a transform

In this example, a sample collection is piped into a transform to apply latitude
weighting and per-variable weighting and `apply_transform` is used to
materialize it.

```@example samples
transformed_collection =
    sample_collection |>
    SampleBuilder.LatitudeWeighting() |>
    SampleBuilder.PerVariableWeighting(Dict("pr" => 2.0, "rsut" => 1.0))
weighted_collection = SampleBuilder.apply_transform(transformed_collection)
```

A transform does not eagerly change the samples. It returns a
[`TransformedSampleCollection`](@ref), which keeps the original samples and the
transforms to apply. See the next page,
[building observations](observation_recipe.md), for how
[`TransformedSampleCollection`](@ref)s are used.

### Built-in transforms

- [`LatitudeWeighting`](@ref) multiplies each value at latitude `lat` by
  `sqrt(1 / max(cosd(lat), min_cosd_lat))`, which inflates the variance of the
  smaller grid cells toward the poles so that they count less. By default,
  every variable with a latitude
  dimension is weighted. To weight only some variables, pass their short names,
  as in `LatitudeWeighting(["pr"])`.
- [`PerVariableWeighting`](@ref) multiplies the samples of each variable by a
  weight. Pass either a vector with one weight per variable, in the order the
  variables were passed to `build_samples` or `build_samples_by_times`, or a
  dictionary from short names to weights, as in
  `PerVariableWeighting(Dict("pr" => 2.0, "rsut" => 1.0))`.
- [`PerCollectionWeighting`](@ref) multiplies all the samples by one weight, for
  example to weight this observation against others that it is combined with.

### Working with transformed samples

Showing a `TransformedSampleCollection` lists its transforms in the order they
are applied. You can get the sequence of transformations with
[`transform_sequence`](@ref).

```@example samples
transformed_collection
```

You can get the original samples with
[`base`](@ref base(::TransformedSampleCollection)).

```@example samples
SampleBuilder.base(transformed_collection) === sample_collection
```

To apply a single transform to a `SampleCollection` directly, for example to
test a transform, use [`apply_transform!`](@ref).

```@example samples
doubled_collection = SampleBuilder.apply_transform!(
    weighted_collection,
    SampleBuilder.PerCollectionWeighting(2.0),
)
SampleBuilder.get_samples(doubled_collection) ==
2 .* SampleBuilder.get_samples(
    SampleBuilder.apply_transform(transformed_collection),
)
```

### Writing your own transform

A custom transform is a subtype of [`AbstractTransform`](@ref) with a method of
[`apply_transform!`](@ref). The following transform sets every sample to zero
which is not practically useful, but serves as an illustrative example.

```@example samples
struct ZeroTransform <: SampleBuilder.AbstractTransform end

function SampleBuilder.apply_transform!(sample_collection, ::ZeroTransform)
    SampleBuilder.get_samples(sample_collection) .= 0
    return sample_collection
end

zeroed_collection =
    SampleBuilder.apply_transform(sample_collection |> ZeroTransform())
# Whether the transformed and the original samples are all zero
all(iszero, SampleBuilder.get_samples(zeroed_collection)),
all(iszero, SampleBuilder.get_samples(sample_collection))
```

The transformed samples are all zero, while the original samples are unchanged.
You can expect the `ZeroTransform` to behave like the other `AbstractTransform`s
where `apply_transform`, piping, calling the transform directly as in
`ZeroTransform()(sample_collection)`, and chaining with other transforms all
work. When implementing `SampleBuilder.apply_transform!`, it must return a
`SampleCollection`. The returned `SampleCollection` must have the same metadata
as the one passed in, since the observation and its metadata come from the
untransformed samples.

## Next steps

Once you have a `SampleCollection` or a `TransformedSampleCollection`, you pass
a covariance estimator, the sample collection, and the index of the sample to
use as the observation to [`observation`](@ref
ClimaCalibrate.ObservationRecipe.observation) to build the `EKP.Observation`
used in the calibration. See [building observations](observation_recipe.md) for
the available estimators.

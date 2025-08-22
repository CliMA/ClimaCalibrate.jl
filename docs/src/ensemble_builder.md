# Building G ensemble matrix

TODO: Put the g_ens stuff in a module since I am not sure about it right now
TODO: Do any other reorganization too in the code
TODO: Add links to all the functions!
TODO: If I put it in another module, change the function name

!!! warning
    To enable this module, use `using ClimaAnalysis` or `import ClimaAnalysis`.

!!! note "Prerequisites"
    This module assumes that you are using `ObservationRecipe` to make your
    observations and `ClimaAnalysis` to preprocess your simulation data. If this
    is not the case, this module is not for you.

!!! note "Other documentation"
    It may be helpful to review the documentation for `FlatVar` in ClimaAnalysis
    and `ObservationRecipe` in ClimaCalibrate.

To help with constructing G ensemble matrix when using `ObservationRecipe`,
`ClimaCalibrate` provides the struct `GEnsembleBuilder` and its related
functions to easily create G ensemble matrix using the metadata stored in the
observation. The metadata stores a rich amount of information that enables
comprehensive validation and checking between simulation and
observational data.

With the metadata stored in the observations, this enables an automatic process
of flattening or vectorizing your `OutputVar` and filling out your G ensemble
matrix with consistency checks between the simulation data and observational
data. Futhermore, this eliminates user errors, where the user did not check for
the ordering of the dimensions when flattening the `OutputVar`, or did not check
for that the dimensions are consistent.

## `GEnsembleBuilder`

To construct a `GEnsembleBuilder`, you can pass the `EKP.EnsembleKalmanProcess`
object and the desired float type to `GEnsembleBuilder`.

```julia
import ClimaCalibrate
import ClimaAnalysis # needed to enable extension
import ClimaCalibrate.EnsembleBuilder

# ekp is a EnsembleKalmanProcesses.EnsembleKalmanProcess object
g_ens_builder = EnsembleBuilder.GEnsembleBuilder(ekp, Float32)
```

Then, in your observation map in your calibration pipeline, you should
preprocess your `OutputVar`s using `ClimaAnalysis`. For a CliMA simulation, this
involves using `SimDir` to load the NetCDF files as `OutputVar`s and
preprocessing them, so that they match the `OutputVar`s that are used to create
the observations.

!!! note "Short names"
    Using `GEnsembleBuilder` requires attaching a short name to all
    `OutputVar`s. The empty string cannot be used a short name. Without the
    short name, it is not possible to determine how the observational data match
    with the simulation data. Furthermore, the short names should be unique. For
    example, if you are calibrating monthly minimum and maximum of
    precipitation, then the short name for the monthly minimum can be `pr_min`
    and the short name for the monthly maximum can be `pr_max`.

In particular, the `OutputVar`s from the simulation data should match the
`OutputVar`s from the observations from
- the short name (see `ClimaAnalysis.short_name`),
- the non-temporal dimensions
- the dates of the simulation data includes all the dates of one or more
  metadata in the observations

!!! info "Spinup and windowing times"
    Internally, the correct dates are matched between the observational and
    simulation data. As a result, you do not need to window the times (e.g. when
    removing spinup) to match the times of the observations.

!!! warning "Matching dates"
    There is no checking for how dates are matched which can easily lead to
    errors. For example, if the simulation data contain monthly averages and
    metadata track seasonal averages, then no error is thrown, because all dates
    in `metadata` is in the set of all dates in `var`.

TODO: Give an example of how to use it and prereq for it (like short names in
observations and what not)

After preprocessing the `OutputVar`s, you can call `fill_g_ens_col!` to fill the
G ensemble matrix using the `OutputVars`.

```julia
# -- preprocessing OutputVars from the first ensemble member--
# var1, var2, and var3 are OutputVars of different quantities
# The second argument is which column of the G ensemble matrix to fill out
EnsembleBuilder.fill_g_ens_col!(g_ens_builder, 1, var1, var2, var3)
```

The function `fill_g_ens_col!` can take in any number of `OutputVar`s. This
allows for flexibility for how you want to preprocess the `OutputVar`s. For
example, you can preprocess all the `OutputVar`s of a single ensemble member and
pass all the `OutputVar`s to `fill_g_ens_col!`. On the other hand, you can also
preprocess a single `OutputVar`, pass it to `fill_g_ens_col!`, and repeat for
all the `OutputVar`s. We recommend the latter approach as loading all the
`OutputVar`s simultaneously can consume a lot of memory.

!!! note "Unused `OutputVar`s"
    If an `OutputVar` is passed to `fill_g_ens_col!`, an error will not be
    thrown. However, *warnings* will be thrown when a `OutputVar` is not used.

Next, you can check if the G ensemble matrix is filled out with `is_complete`
which return `true` if the G ensemble matrix is completed and `false` otherwise.

```julia
is_complete(g_ens_builder)
```

If it returns `false`, then you should check the warnings thrown to determine
why a `OutputVar` is not used to fill out the G ensemble matrix.

Finally, you get the G ensemble matrix is `get_g_ensemble` and return it in
`ClimaCalibrate.observation_map`.

```julia
g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
```

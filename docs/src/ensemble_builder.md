# Building G ensemble matrix

!!! warning
    To enable this module, use `using ClimaAnalysis` or `import ClimaAnalysis`.

!!! note "Prerequisites"
    This module assumes that you are using `ObservationRecipe` to make your
    observations and `ClimaAnalysis` to preprocess your simulation data. If this
    is not the case, this module is not for you.

!!! note "Version of ClimaAnalysis"
    This module requires a version of ClimaAnalysis greater than v0.5.19.

!!! note "Other documentation"
    It may be helpful to review the documentation for
    [`FlatVar`](https://clima.github.io/ClimaAnalysis.jl/dev/flat/) in
    ClimaAnalysis and [`ObservationRecipe`](@ref) in ClimaCalibrate.

To help with constructing G ensemble matrix when using `ObservationRecipe`,
`ClimaCalibrate` provides the struct `GEnsembleBuilder` and its related
functions to easily create G ensemble matrix using the metadata stored in the
observation. The metadata stores a rich amount of information that enables
comprehensive validation and checking between simulation and
observational data.

The metadata stored in the observations enable an automatic process of
flattening or vectorizing your `OutputVar` and filling out your G ensemble
matrix with consistency checks between the simulation data and observational
data. This eliminates user errors, such as checking for the ordering of the
dimensions when flattening the `OutputVar`, checking that the dimensions are
consistent, and checking the units between observational and simulation data.

## `GEnsembleBuilder`

To construct a [`GEnsembleBuilder`](@ref ClimaCalibrate.EnsembleBuilder.GEnsembleBuilder),
you pass the `EKP.EnsembleKalmanProcess` object to the constructor.

```julia
import ClimaCalibrate
import ClimaAnalysis # needed to enable extension
import ClimaCalibrate.EnsembleBuilder

# ekp is a EnsembleKalmanProcesses.EnsembleKalmanProcess object
g_ens_builder = EnsembleBuilder.GEnsembleBuilder(ekp)
```

Then, in your observation map of your calibration, you should preprocess your
`OutputVar`s using `ClimaAnalysis`. For a CliMA simulation, this involves using
`SimDir` to load the NetCDF files as `OutputVar`s and preprocessing them, so
that they match the `OutputVar`s that are used to create the observations.

!!! note "Short names"
    Using `GEnsembleBuilder` requires attaching a short name to all `OutputVar`s
    for both simulation and observational data. The empty string cannot be used
    as a short name. Without the short name, it is not possible to determine
    which simulation data match with which observational data. Furthermore, the
    short names should be unique. For example, if you are calibrating monthly
    minimum and maximum of precipitation, then the short name for the monthly
    minimum can be `pr_min` and the short name for the monthly maximum can be
    `pr_max`.

In particular, the `OutputVar`s from the simulation data should match the
`OutputVar`s from the observations from
- the short name (see
  [`ClimaAnalysis.short_name`](https://clima.github.io/ClimaAnalysis.jl/dev/api/#ClimaAnalysis.Var.short_name)),
- the non-temporal dimensions,
- the dates of the simulation data includes all the dates of one or more
  metadata in the observations,
- the units of the variables.

!!! info "Spinup and windowing times"
    Internally, the correct dates are matched between the observational and
    simulation data. As a result, you do not need to window the times (e.g. when
    removing spinup) to match the times of the observations.

!!! warning "Matching dates"
    There are no checks for how dates are matched which can easily lead to
    errors. For example, if the simulation data contain monthly averages and
    metadata track seasonal averages, then no error is thrown, because all dates
    in `metadata` are in all the dates in `var`.

After preprocessing the `OutputVar`s, you can call
[`fill_g_ens_col!`](@ref ClimaCalibrate.EnsembleBuilder.fill_g_ens_col!) to fill
the G ensemble matrix using the `OutputVars`.

```julia
# -- preprocessing OutputVars from the first ensemble member--
# var1, var2, and var3 are OutputVars of different quantities
# The second argument is which column of the G ensemble matrix to fill out
EnsembleBuilder.fill_g_ens_col!(g_ens_builder, 1, var1, var2, var3)
```

The function `fill_g_ens_col!` can take in any number of `OutputVar`s. This
allows for flexibility for how you want to preprocess the `OutputVar`s. For
example, you can preprocess all the `OutputVar`s of a single ensemble member and
pass all the `OutputVar`s to `fill_g_ens_col!`. On the other hand, you can
preprocess a single `OutputVar`, pass it to `fill_g_ens_col!`, and repeat for
all the other `OutputVar`s. We recommend the latter approach as loading all the
`OutputVar`s simultaneously can consume a lot of memory. Furthermore, it allows
for easier debugging when only a single `OutputVar` is passed to
`fill_g_ens_col!`.

!!! note "Unused OutputVars"
    If a `OutputVar` is passed to `fill_g_ens_col!` and it is not used, an error
    will not be thrown, but a *warning* will be thrown instead.

Next, you can check if the G ensemble matrix is filled out with
[`is_complete`](@ref ClimaCalibrate.EnsembleBuilder.is_complete) which returns
`true` if the G ensemble matrix is completed and `false` otherwise.

```julia
EnsembleBuilder.is_complete(g_ens_builder)
```

If this returns `false`, then you should review the warnings to determine why a
`OutputVar` is not used to fill out the G ensemble matrix.

Finally, you can get the G ensemble matrix with
[`get_g_ensemble`](@ref ClimaCalibrate.EnsembleBuilder.get_g_ensemble) and
return it from `ClimaCalibrate.observation_map`.

```julia
g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
```

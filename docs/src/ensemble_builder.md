# Building G ensemble matrix

!!! note
    If you are not using ClimaAnalysis, you can skip this page. To enable this
    module, use `using ClimaAnalysis` or `import ClimaAnalysis`. This module
    requires a version of ClimaAnalysis greater than v0.5.19.

!!! note "Prerequisites"
    This module assumes that you are using `ObservationRecipe` to make your
    observations and `ClimaAnalysis` to preprocess your simulation data. If that
    is not the case, this module is not for you.

!!! note "Other documentation"
    It may be helpful to review the documentation for
    [`FlatVar`](https://clima.github.io/ClimaAnalysis.jl/dev/flat/) in
    ClimaAnalysis and [`ObservationRecipe`](@ref) in ClimaCalibrate.

Calibration involves implementing an [`observation_map`](@ref) by hand which
postprocess, slice, and stack simulation diagnostics into vectors.
`GEnsembleBuilder` automates this, using the metadata baked into your
ObservationRecipe observations to match, validate, and fill each column for you.

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

For more information about the checks that are performed, see the
[Checkers](#checkers) section.

!!! info "Spinup and windowing times"
    Internally, the correct dates are matched between the observational and
    simulation data. As a result, you do not need to window the times (e.g. when
    removing spinup) to match the times of the observations.

!!! warning "Matching dates"
    There are no checks for how dates are matched which can easily lead to
    errors. For example, if the simulation data contain monthly averages and
    metadata track seasonal averages, then no error is thrown, because all dates
    in `metadata` are in all the dates in `var`.

    For these reasons, it is recommended to pass
    [`SequentialIndicesChecker`](@ref ClimaCalibrate.Checker.SequentialIndicesChecker)
    to the `checkers` keyword argument of
    [`fill_g_ens_col!`](@ref ClimaCalibrate.EnsembleBuilder.fill_g_ens_col!)
    which will check that the dates used to fill the G ensemble matrix
    correspond to sequential indices in the simulation data.

After preprocessing the `OutputVar`s, you can call
[`fill_g_ens_col!`](@ref ClimaCalibrate.EnsembleBuilder.fill_g_ens_col!) to fill
the G ensemble matrix using the `OutputVars`.

```julia
# -- preprocessing OutputVars from the first ensemble member--
# var1, var2, and var3 are OutputVars of different quantities
# The second argument is which column of the G ensemble matrix to fill out
vars = (var1, var2, var3)
for var in vars
    EnsembleBuilder.fill_g_ens_col!(g_ens_builder, 1, var)
end
```

In this example, all the `OutputVar`s of a single ensemble member are
preprocessed and pass to `fill_g_ens_col!` in a loop. On the other hand, one can
preprocess a single `OutputVar`, pass it to `fill_g_ens_col!`, and repeat for
all the other `OutputVar`s. We recommend the latter approach as loading all the
`OutputVar`s simultaneously can consume a lot of memory.

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

## Example

A complete example of using the `GEnsembleBuilder` looks like this.

```julia
import ClimaCalibrate
import ClimaCalibrate.EnsembleBuilder

function ClimaCalibrate.observation_map(interface::MyModelInterface, iteration)
    # In this example, output_dir is stored in interface as a field
    (; output_dir) = interface
    ekp = JLD2.load_object(ClimaCalibrate.ekp_path(output_dir, iteration))
    ensemble_size = EKP.get_N_ens(ekp)

    g_ens_builder = EnsembleBuilder.GEnsembleBuilder(ekp)
    for m in 1:ensemble_size
        try
        member_path =
            ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)
        diagnostics_path = joinpath(member_path, "output_active")
        @info "Processing member $m: $diagnostics_path"
            process_member_data!(g_ens_builder, m, diagnostics_path)
        catch e
            @error "Error processing member $m, filling observation map entry with NaNs" exception =
                e
            EnsembleBuilder.fill_g_ens_col!(g_ens_builder, m, NaN)
        end
    end

    if EnsembleBuilder.is_complete(g_ens_builder)
        return EnsembleBuilder.get_g_ensemble(g_ens_builder)
    else
        @error "G ensemble matrix is not completed. You may find it useful to call `EnsembleBuilder.missing_short_names(g_ens_builder, 1) or display the GEnsembleBuilder object in the REPL"
    end
end

function process_member_data!(
    g_ens_builder,
    col_idx,
    diagnostics_folder_path,
)
    # The implementation of preprocess differs for each calibration pipeline,
    # but this load and preprocess OutputVars using ClimaAnalysis
    vars = preprocess(m, diagnostics_folder_path)

    # It is strongly recommended to use SequentialIndicesChecker
    seq_indices_checker = Checker.SequentialIndicesChecker()
    checkers = (seq_indices_checker,)

    # fill_g_ens_col! will remove the spinup and is mask-aware.
    # g_ens_builder contain the metadata from the observations, so
    # fill_g_ens_col! will only choose values over temporal and spatial
    # coordinates that exist in the observational data
    for var in vars
        use_var = EnsembleBuilder.fill_g_ens_col!(
            g_ens_builder,
            col_idx,
            var;
            checkers,
            verbose = true,
        )
        use_var || error(
            "OutputVar with short name ($(ClimaAnalysis.short_name(var))) was passed, but not used",
        )
    end
    return nothing
end
```

# Checkers

To determine whether a `OutputVar` matches with a metadata, `GEnsembleBuilder`
uses `Checker`s to check and compare the contents of a `OutputVar` and with that
of the metadata. For example, the short names are checked between the
`OutputVar` and metadata with
[`ShortNameChecker`](@ref ClimaCalibrate.Checker.ShortNameChecker).

## Built-in checkers

`ClimaCalibrate` provides several built-in checkers:

- [`ShortNameChecker`](@ref ClimaCalibrate.Checker.ShortNameChecker): Check
  that the short names match between `OutputVar` and metadata
- [`DimNameChecker`](@ref ClimaCalibrate.Checker.DimNameChecker): Check
  the type of dimensions are the same
- [`UnitsChecker`](@ref ClimaCalibrate.Checker.UnitsChecker): Check the
  variable units are the same
- [`DimUnitsChecker`](@ref ClimaCalibrate.Checker.DimUnitsChecker): Check
  the units of the dimensions are the same
- [`DimValuesChecker`](@ref ClimaCalibrate.Checker.DimValuesChecker): Check the
  values of the dimensions are the same
- [`SequentialIndicesChecker`](@ref ClimaCalibrate.Checker.SequentialIndicesChecker):
  Check the indices of the dates of the simulation data corresponding to the
  dates of the metadata is sequential.
- [`SignChecker`](@ref ClimaCalibrate.Checker.SignChecker):
  Check that the proportion of positive values in the simulation data and
  observational data are approximately equal (within a default threshold of
  0.05).

By default, `GEnsembleBuilder` uses the first five checkers to validate
compatibility between observational data and simulation data. You can also
provide additional checkers using the `checkers` keyword argument in
`fill_g_ens_col!`:

```julia
# Use additional checker for sequential indices
EnsembleBuilder.fill_g_ens_col!(
    g_ens_builder,
    1, 
    var,
    checkers = (SequentialIndicesChecker(),)
)
```

## Implementing custom checkers

To create a custom checker, define a struct that inherits from `AbstractChecker`
and implement the `check` method:

```julia
import ClimaCalibrate.Checker

# Define your custom checker
struct NothingChecker <: Checker.AbstractChecker end

# Implement the check method
function Checker.check(
    ::NothingChecker,
    var,
    metadata;
    data = nothing,
    verbose = false,
)
    verbose && @info "This is always true."
    return true
end
```

The `Checker.check` function should
- accept a checker instance, an `OutputVar`, and `Metadata`,
- return `true` if the check passes, `false` otherwise,
- and optionally log informative messages when `verbose = true`.

For more information about `OutputVar` and `Metadata`, see the
[ClimaAnalysis documentation](https://clima.github.io/ClimaAnalysis.jl/dev/).

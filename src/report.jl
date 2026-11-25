"""
    ClimaCalibrate.Report

Reporters that summarize the state of a calibration in a report at events of the
calibration, such as the end of each iteration.

[`CalibrationReport`](@ref) puts the report together from reporters, each of
which adds a section on

1. the walltime ([`WalltimeReporter`](@ref)),
2. the timesteps ([`TimestepReporter`](@ref)),
3. the error ([`ErrorReporter`](@ref)),
4. the parameters ([`ParameterReporter`](@ref)),
5. or the state of the ensemble ([`EnsembleReporter`](@ref)).
"""
module Report

export AbstractEvent,
    CalibrationStart,
    IterationStart,
    IterationEnd,
    CalibrationEnd,
    ReportContext,
    AbstractReporter,
    CalibrationReport,
    WalltimeReporter,
    TimestepReporter,
    ErrorReporter,
    ParameterReporter,
    EnsembleReporter,
    report,
    title,
    print_aligned

import Dates
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import ..EKPUtils: _constrained_spread

"""
    AbstractEvent

An event of a calibration at which a [`CalibrationReport`](@ref) is logged.

The events are [`CalibrationStart`](@ref), [`IterationStart`](@ref),
[`IterationEnd`](@ref), and [`CalibrationEnd`](@ref). For a calibration of `n`
iterations, they happen in this order:

```
CalibrationStart(1)
IterationStart(1), IterationEnd(1)
IterationStart(2), IterationEnd(2)
⋮
IterationStart(n), IterationEnd(n)
CalibrationEnd(n)
```

A resumed calibration starts at the first iteration that is not completed. When
the scheduler terminates the calibration in iteration `i`, there is no
`IterationEnd(i)` and the last event is `CalibrationEnd(i - 1)`.
"""
abstract type AbstractEvent end

"""
    CalibrationStart <: AbstractEvent

The event at the start of a calibration, before its first iteration runs.

# Fields
- `iteration`: The first iteration to run. When a calibration is resumed, this
  is the first iteration that is not completed, which can be greater than 1.
"""
struct CalibrationStart <: AbstractEvent
    iteration::Int
end

"""
    IterationStart <: AbstractEvent

The event at the start of an iteration, before the forward models of the
iteration run.

# Fields
- `iteration`: The iteration that is starting.
"""
struct IterationStart <: AbstractEvent
    iteration::Int
end

"""
    IterationEnd <: AbstractEvent

The event at the end of an iteration, after the `EnsembleKalmanProcess` is
updated with the output of the forward models of the iteration.

There is no `IterationEnd` for the iteration where the scheduler terminates the
calibration, since the `EnsembleKalmanProcess` is not updated.

`IterationEnd` happens at the end of every iteration, while
[`CalibrationEnd`](@ref) happens once, after the last iteration. Use
`IterationEnd` to report on each iteration and `CalibrationEnd` to summarize the
whole calibration.

# Fields
- `iteration`: The iteration that ended.
"""
struct IterationEnd <: AbstractEvent
    iteration::Int
end

"""
    CalibrationEnd <: AbstractEvent

The event at the end of a calibration, after its last iteration.

`CalibrationEnd` happens once, whether the calibration runs all of its
iterations or the scheduler terminates it.

# Fields
- `iteration`: The last iteration whose update is in the
  `EnsembleKalmanProcess`. For a calibration that runs all `n` of its
  iterations, this is `n`. The scheduler terminates the calibration when the
  ensemble is about to be updated at the end of an iteration `i`, after the
  forward models of the iteration have run. The ensemble is not updated, so
  this is `i - 1`. The output of the forward models of iteration `i` is saved
  to the output directory, but is not added to the `EnsembleKalmanProcess`.
"""
struct CalibrationEnd <: AbstractEvent
    iteration::Int
end

"""
    ReportContext(ekp, prior, output_dir)

The state of the calibration that is given to each reporter, made of

- `ekp`: The `EnsembleKalmanProcess` of the calibration,
- `prior`: The `ParameterDistribution` of the calibration,
- `output_dir`: The output directory of the calibration.
"""
struct ReportContext
    ekp::EKP.EnsembleKalmanProcess
    prior::PD.ParameterDistribution
    output_dir::String
end

"""
    AbstractReporter

An object that reports information about the state of the calibration at
events of the calibration, such as the end of each iteration, from a
[`ReportContext`](@ref), which holds the `ekp::EnsembleKalmanProcess` and
`prior::ParameterDistribution` of the calibration.

## Define your own reporter

An `AbstractReporter` must implement `ClimaCalibrate.Report.report` for each
[`AbstractEvent`](@ref) it reports on, and optionally,
`ClimaCalibrate.Report.title`. A `report` method that does not specify the type
of the event reports on every event. A reporter that prints nothing, such as
one without a `report` method for the event, is left out of the report. For a
minimal example, see the "Report" section on the `ClimaCalibrate`
documentation.
"""
abstract type AbstractReporter end

# Only the reporter is typed, so that any method for a reporter is more specific
report(_, ::AbstractReporter, _, _) = nothing

"""
    title(reporter::AbstractReporter)

Return the title of the section of the report as a string.
"""
function title(reporter::AbstractReporter)
    return string(nameof(typeof(reporter)))
end

"""
    print_aligned(io, rows)

Print each `label => value` pair in `rows` to `io` as `label: value`, padding
the labels so that the values line up.
"""
function print_aligned(io, rows)
    isempty(rows) && return nothing
    width = maximum(row -> textwidth(first(row)), rows) + 1
    for (label, value) in rows
        println(io, rpad(label * ":", width), " ", value)
    end
    return nothing
end

"""
    CalibrationReport

A report on the state of the calibration, logged at events of the calibration,
such as the end of each iteration.

The report starts with a header of the event, date, and time. Each reporter in
`reporters` that prints something for the event then adds a section, in order,
made of its [`title`](@ref) and what its `report` method prints.

To add a section of your own, see [`AbstractReporter`](@ref).
"""
struct CalibrationReport
    reporters::Tuple{Vararg{AbstractReporter}}
end

"""
    CalibrationReport(;
        reporters = (
            WalltimeReporter(),
            TimestepReporter(),
            ErrorReporter(),
            ParameterReporter(),
            EnsembleReporter(),
        ),
    )

Construct a `CalibrationReport` from `reporters`.

# Keyword Arguments
- `reporters`: The reporters that make up the report, in order. By default,
  these are `WalltimeReporter()`, `TimestepReporter()`, `ErrorReporter()`,
  `ParameterReporter()`, and `EnsembleReporter()`.
"""
function CalibrationReport(;
    reporters = (
        WalltimeReporter(),
        TimestepReporter(),
        ErrorReporter(),
        ParameterReporter(),
        EnsembleReporter(),
    ),
)
    return CalibrationReport(Tuple(reporters))
end

"""
    report(
        calibration_report::CalibrationReport,
        event::AbstractEvent,
        context::ReportContext,
    )

Log a report of `event` with a header of the event, date, and time and the
contents of each reporter in `calibration_report` that prints something for
`event`, using `context`. If no reporter prints anything, nothing is logged.

If a reporter throws an error, its section says so and the error is logged after
the report, so the other sections are still reported.
"""
function report(
    calibration_report::CalibrationReport,
    event::AbstractEvent,
    context::ReportContext,
)
    (; reporters) = calibration_report
    io = IOBuffer()
    failures = Pair{String, Any}[]
    for reporter in reporters
        # Fall back to the default title if `title` throws
        section_title = string(nameof(typeof(reporter)))
        body = try
            section_title = title(reporter)
            sprint(io -> report(io, reporter, event, context))
        catch err
            err isa InterruptException && rethrow()
            push!(failures, section_title => (err, catch_backtrace()))
            "Encountered an error while generating this section\n"
        end
        isempty(body) && continue
        println(io, "  ", section_title)
        for line in eachline(IOBuffer(body))
            println(io, "    ", line)
        end
    end
    sections = String(take!(io))
    isempty(sections) && return nothing
    header = "Calibration report · $(_describe(event)) · \
              $(trunc(Dates.now(), Dates.Second))"
    @info header * "\n" * chomp(sections)

    # The errors are logged after the report so that one reporter failing does
    # not stop the rest of the report
    for (section_title, exception) in failures
        @error "Generating $section_title in the calibration report \
                resulted in an error" exception
    end
    return nothing
end

# Helper functions for printing the events of a calibration
_describe(::CalibrationStart) = "start of calibration"
_describe(event::IterationStart) = "start of iteration $(event.iteration)"
_describe(event::IterationEnd) = "iteration $(event.iteration)"
_describe(::CalibrationEnd) = "end of calibration"

"""
    WalltimeReporter <: AbstractReporter

A struct for reporting the walltime and an estimate of the remaining walltime.
"""
struct WalltimeReporter <: AbstractReporter
    times::Vector{Float64}
    n_iterations::Union{Int, Nothing}
end

"""
    WalltimeReporter(n_iterations = nothing)

Construct a `WalltimeReporter` from `n_iterations`, the total number of
iterations.

If `n_iterations` is `nothing`, then an estimate of the remaining walltime is
not reported.

The walltime is measured from [`CalibrationStart`](@ref).
"""
function WalltimeReporter(n_iterations = nothing)
    !isnothing(n_iterations) &&
        n_iterations <= 0 &&
        error("Total number of iterations ($n_iterations) should be positive")
    # Populate with a non-empty vector if this reporter is called manually
    return WalltimeReporter(Float64[time()], n_iterations)
end

"""
    title(::WalltimeReporter)

Return the section title, "Walltime".
"""
title(::WalltimeReporter) = "Walltime"

function report(
    ::IO,
    reporter::WalltimeReporter,
    ::CalibrationStart,
    ::ReportContext,
)
    # Measure the walltime from the start of the calibration
    (; times) = reporter
    empty!(times)
    push!(times, time())
    return nothing
end

"""
    report(
        io::IO,
        reporter::WalltimeReporter,
        ::IterationEnd,
        context::ReportContext,
    )

Report the walltime of the last iteration, the total walltime, and an estimate
of the remaining walltime.
"""
function report(
    io::IO,
    reporter::WalltimeReporter,
    ::IterationEnd,
    context::ReportContext,
)
    (; ekp) = context
    (; times, n_iterations) = reporter
    push!(times, time())

    # Print human readable walltime
    walltime = times[end] - times[end - 1]
    N = EKP.get_N_iterations(ekp)
    total_walltime = last(times) - first(times)
    rows = [
        "iteration walltime" => _format_duration(walltime),
        "total walltime" => _format_duration(total_walltime),
    ]

    if !isnothing(n_iterations) && N < n_iterations
        n_remaining = n_iterations - N
        # Estimate walltime using the mean
        mean_time_diff = (last(times) - first(times)) / (length(times) - 1)
        remaining = Dates.Second(round(Int, mean_time_diff * n_remaining))
        finish_date = trunc(Dates.now() + remaining, Dates.Second)
        iterations = n_remaining == 1 ? "iteration" : "iterations"
        push!(
            rows,
            "walltime remaining for $n_remaining $iterations" => "$(_format_duration(Dates.value(remaining))) ($finish_date)",
        )
    end

    print_aligned(io, rows)
    return nothing
end

"""
    _format_duration(seconds)

Return `seconds`, rounded to whole seconds, as a human readable string like
`"1 hour, 2 minutes, 5 seconds"`.

Return `"0 seconds"` if `seconds` rounds to zero.
"""
function _format_duration(seconds)
    seconds = round(Int, seconds)
    # `Dates.canonicalize` turns zero seconds into an "empty period"
    iszero(seconds) && return "0 seconds"
    return string(Dates.canonicalize(Dates.Second(seconds)))
end

"""
    TimestepReporter <: AbstractReporter

A struct for reporting information about the timesteps taken by the ensemble
Kalman process algorithm.

## Examples

You do not need to pass any arguments to `TimestepReporter` to construct one.

```julia
timestep_reporter = TimestepReporter()
```
"""
struct TimestepReporter <: AbstractReporter end

"""
    title(::TimestepReporter)

Return the section title, "Timestep".
"""
title(::TimestepReporter) = "Timestep"

"""
    report(
        io::IO,
        ::TimestepReporter,
        ::IterationEnd,
        context::ReportContext,
    )

Report the current ensemble Kalman process algorithm time, current timestep
size, and the recent timestep history.
"""
function report(
    io::IO,
    ::TimestepReporter,
    ::IterationEnd,
    context::ReportContext,
)
    (; ekp) = context
    Δts = EKP.get_Δt(ekp)
    isempty(Δts) && return nothing

    algorithm_time = last(EKP.get_algorithm_time(ekp))
    print_aligned(
        io,
        [
            # Report current time
            "current algorithm time" =>
                string(round(algorithm_time; sigdigits = 3)),
            # Report timestep size taken
            "current timestep size" => string(round(last(Δts); sigdigits = 3)),
            # Report series of timesteps taken
            "timestep history" => _format_recent(Δts),
        ],
    )

    return nothing
end

"""
    ErrorReporter <: AbstractReporter

A struct for reporting information about the error metrics.
"""
struct ErrorReporter <: AbstractReporter
    error_metrics::Vector{String}
end

"""
    ErrorReporter(;
       error_metrics::Vector{String} = ["loss", "unweighted_loss"],
    )

Construct an `ErrorReporter` for reporting information about `error_metrics`.

For a list of available error metrics, see the documentation in
EnsembleKalmanProcesses.
"""
function ErrorReporter(;
    error_metrics::Vector{String} = ["loss", "unweighted_loss"],
)
    available_error_metrics = [
        "loss",
        "unweighted_loss",
        "avg_rmse",
        "unweighted_avg_rmse",
        "bayes_loss",
        "crps",
    ]
    unknown_error_metrics = setdiff(error_metrics, available_error_metrics)
    isempty(unknown_error_metrics) || throw(
        error(
            "Unknown error metrics $unknown_error_metrics. The available error \
            metrics are $available_error_metrics",
        ),
    )
    return ErrorReporter(error_metrics)
end

"""
    title(::ErrorReporter)

Return the section title, "Error".
"""
title(::ErrorReporter) = "Error"

"""
    report(
        io::IO,
        reporter::ErrorReporter,
        ::IterationEnd,
        context::ReportContext,
    )

Report the error metrics as desired in `reporter` with how it changes from the
error from the previous iteration, the best error so far and the iteration it
was reached at, and a recent history of the error metrics.
"""
function report(
    io::IO,
    reporter::ErrorReporter,
    ::IterationEnd,
    context::ReportContext,
)
    (; ekp) = context
    all_error_metrics = EKP.get_error_metrics(ekp)
    isempty(all_error_metrics) && return nothing

    rows = Pair{String, String}[]
    for name in reporter.error_metrics
        errors = all_error_metrics[name]
        push!(rows, "current $name" => "$(round(last(errors); sigdigits = 3))\
                        $(_format_change(errors))")
        # Inf and NaN are skipped when finding the best error
        finite_iterations = findall(isfinite, errors)
        if !isempty(finite_iterations)
            best_error, k = findmin(errors[finite_iterations])
            push!(rows, "best $name" => "$(round(best_error; sigdigits = 3)) \
                            (iteration $(finite_iterations[k]))")
        end
        push!(rows, "$name history" => _format_recent(errors))
    end
    print_aligned(io, rows)

    return nothing
end

"""
    ParameterReporter <: AbstractReporter

A struct for reporting information about the parameter trajectory.

## Examples

You do not need to pass any arguments to `ParameterReporter` to construct one.

```julia
parameter_reporter = ParameterReporter()
```
"""
struct ParameterReporter <: AbstractReporter end


"""
    title(::ParameterReporter)

Return the section title, "Parameter".
"""
title(::ParameterReporter) = "Parameter"

"""
    report(
        io::IO,
        ::ParameterReporter,
        ::IterationEnd,
        context::ReportContext,
    )

Report the mean and spread of each constrained parameter and their change from
the previous iteration.
"""
function report(
    io::IO,
    ::ParameterReporter,
    ::IterationEnd,
    context::ReportContext,
)
    (; ekp, prior) = context
    N = EKP.get_N_iterations(ekp)
    N == 0 && return nothing

    previous_means = EKP.get_ϕ_mean(prior, ekp, N)
    means = EKP.get_ϕ_mean_final(prior, ekp)
    previous_spreads = _constrained_spread(ekp, prior, N)
    spreads = _constrained_spread(ekp, prior, N + 1)

    rows = Pair{String, String}[]
    for (j, label) in enumerate(_parameter_labels(prior))
        # Print average constrained parameters and whether they shrink or grow
        mean_change = _format_change([previous_means[j], means[j]])
        push!(rows, label => "$(round(means[j]; sigdigits = 3))$mean_change")

        # Print spread and whether it shrinks or grows
        spread_change = _format_change([previous_spreads[j], spreads[j]])
        push!(
            rows,
            "$label spread" => "$(round(spreads[j]; sigdigits = 3))$spread_change",
        )
    end
    print_aligned(io, rows)
    return nothing
end

"""
    EnsembleReporter <: AbstractReporter

A struct for reporting information about the state of the ensemble.

## Examples

You do not need to pass any arguments to `EnsembleReporter` to construct one.

```julia
ensemble_reporter = EnsembleReporter()
```
"""
struct EnsembleReporter <: AbstractReporter end

"""
    title(::EnsembleReporter)

Return the section title, "Ensemble".
"""
title(::EnsembleReporter) = "Ensemble"

"""
    report(
        io::IO,
        ::EnsembleReporter,
        ::IterationEnd,
        context::ReportContext,
    )

Report the number of successful ensemble members out of the total number of
ensemble members and list the failing ensemble members.
"""
function report(
    io::IO,
    ::EnsembleReporter,
    ::IterationEnd,
    context::ReportContext,
)
    (; ekp) = context
    EKP.get_N_iterations(ekp) == 0 && return nothing
    g = EKP.get_g_final(ekp)
    # Found directly since `EKP.split_indices_by_success` warns and throws
    failed_members = findall(member -> any(isnan, member), eachcol(g))
    n_members = size(g, 2)
    n_successful = n_members - length(failed_members)
    rows = ["successful members" => "$n_successful of $n_members"]
    isempty(failed_members) ||
        push!(rows, "failed members" => join(failed_members, ", "))
    print_aligned(io, rows)
    return nothing
end

"""
    _format_recent(values; n = 5)

Return the last `n` of `values`, rounded to 3 significant figures, as a string
like `"[…, 1.2, 0.8]"`.
"""
function _format_recent(values; n = 5)
    recent = round.(last(values, n); sigdigits = 3)
    ellipsis = length(values) > n ? "…, " : ""
    return "[" * ellipsis * join(recent, ", ") * "]"
end

"""
    _format_change(values)

Return the percent change of the last of `values` from the one before it, as a
string like `" (-38.2% from previous)"`.

Return an empty string if there is no previous value or if the change is not
finite.
"""
function _format_change(values)
    length(values) < 2 && return ""
    previous, current = values[end - 1], values[end]
    change = 100 * (current - previous) / abs(previous)
    isfinite(change) || return ""
    change = round(change; digits = 1)
    # A small decrease rounds to -0.0, which should be printed as 0.0
    iszero(change) && (change = zero(change))
    sign = change > 0 ? "+" : ""
    return " ($sign$change% from previous)"
end

"""
    _parameter_labels(prior)

Return a label for each dimension of the parameters in `prior`, in the order of
the parameter vector.

A parameter with one dimension is labeled with its name. A parameter with more
than one dimension is labeled with its name and the index of the dimension, such
as `"name[2]"`.
"""
function _parameter_labels(prior)
    labels = String[]
    for (name, indices) in zip(
        PD.get_name(prior),
        PD.batch(prior; function_parameter_opt = "eval"),
    )
        if length(indices) == 1
            push!(labels, name)
        else
            append!(labels, ["$name[$k]" for k in eachindex(indices)])
        end
    end
    return labels
end

end

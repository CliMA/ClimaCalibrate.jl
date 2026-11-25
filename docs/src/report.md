```@meta
CurrentModule = ClimaCalibrate.Report
```

# Calibration report

ClimaCalibrate provides utilities for reporting the state of each iteration of a
calibration. This is implemented with a [`CalibrationReport`](@ref) which is
built from [`AbstractReporter`](@ref)s such as

- [`WalltimeReporter`](@ref) for reporting the walltime information,
- [`TimestepReporter`](@ref) for reporting the timesteps taken,
- [`ErrorReporter`](@ref) for reporting the error metrics,
- [`ParameterReporter`](@ref) for reporting the parameter trajectory,
- and [`EnsembleReporter`](@ref) for reporting the state of the ensemble.

## Example

```@setup report
import ClimaCalibrate as CAL
import ClimaCalibrate.Report as Report
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import Logging

prior = PD.combine_distributions([
    PD.constrained_gaussian("a", 1.0, 0.5, -Inf, Inf),
    PD.constrained_gaussian("b", 2.0, 1.0, 0, Inf),
])
G(ϕ) = [ϕ[1] + ϕ[2], ϕ[1] - ϕ[2], 2ϕ[1]]
ekp = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, 3),
    G([1.5, 2.5]),
    [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01],
    EKP.Inversion();
    scheduler = EKP.DataMisfitController(on_terminate = "continue"),
)

struct ToyInterface <: CAL.AbstractModelInterface
    output_dir::String
end

CAL.forward_model(::ToyInterface, iteration, member) = nothing

function CAL.observation_map(interface::ToyInterface, iteration)
    # A nonzero walltime for the WalltimeReporter
    sleep(1)
    ekp = CAL.load_ekp_struct(interface.output_dir, iteration)
    ϕ = EKP.get_ϕ_final(prior, ekp)
    return reduce(hcat, G.(eachcol(ϕ)))
end

# Compile calibrate before the WalltimeReporter below starts timing
Logging.with_logger(Logging.NullLogger()) do
    output_dir = mktempdir()
    CAL.calibrate(
        CAL.JuliaBackend(),
        deepcopy(ekp),
        ToyInterface(output_dir),
        1,
        prior,
        output_dir;
        calibration_report = Report.CalibrationReport(; reporters = ()),
    )
end
```

You can build a [`CalibrationReport`](@ref) by passing a tuple of
[`AbstractReporter`](@ref)s to the `reporters` keyword argument.

```@example report
import ClimaCalibrate.Report as Report

reporters = (
    # Estimate the remaining walltime for a calibration of 10 iterations
    Report.WalltimeReporter(10),
    Report.TimestepReporter(),
    Report.ErrorReporter(; error_metrics = ["loss", "unweighted_loss"]),
    Report.ParameterReporter(),
    Report.EnsembleReporter(),
)
calibration_report = Report.CalibrationReport(; reporters)

nothing # hide
```

The output of the first two iterations of a calibration looks like:

```@example report
output_dir = mktempdir() # hide
CAL.calibrate( # hide
    CAL.JuliaBackend(), # hide
    ekp, # hide
    ToyInterface(output_dir), # hide
    2, # hide
    prior, # hide
    output_dir; # hide
    calibration_report, # hide
) # hide
nothing # hide
```

## Events

A calibration logs a report at each of these [`AbstractEvent`](@ref)s:

- [`CalibrationStart`](@ref): before the first iteration that runs, including
  when a calibration is resumed,
- [`IterationStart`](@ref): at the start of each iteration, before the forward
  models run,
- [`IterationEnd`](@ref): at the end of each iteration, after the
  `EnsembleKalmanProcess` is updated,
- [`CalibrationEnd`](@ref): after the last iteration.

For a calibration of `n` iterations, the events happen in this order:

```
CalibrationStart(1)
IterationStart(1), IterationEnd(1)
IterationStart(2), IterationEnd(2)
⋮
IterationStart(n), IterationEnd(n)
CalibrationEnd(n)
```

`IterationEnd` happens at the end of every iteration, while `CalibrationEnd`
happens once, after the last iteration. Use `IterationEnd` to report on each
iteration, such as its error, and `CalibrationEnd` to summarize the whole
calibration.

The order changes in these cases:

- When a calibration is resumed, it starts at the first iteration that is not
  completed, so the first event is `CalibrationStart(k)` for that iteration `k`.
- When the scheduler terminates the calibration in iteration `i`, the
  `EnsembleKalmanProcess` is not updated, so there is no `IterationEnd(i)` and
  the last event is `CalibrationEnd(i - 1)`.

Each reporter decides which events it reports on. A reporter that prints nothing
for an event is left out of the report of that event, and if every reporter
prints nothing, nothing is logged.

## Implementing your own AbstractReporter

To write your own reporter, subtype [`AbstractReporter`](@ref) and add a
method to [`report`](@ref), and optionally to [`title`](@ref). The method below
reports at the end of each iteration.

```@example reporter
import ClimaCalibrate.Report as Report

struct YourReporter <: Report.AbstractReporter end

function Report.report(
    io::IO,
    reporter::YourReporter,
    event::Report.IterationEnd,
    context::Report.ReportContext,
)
    println(io, "Your report for iteration $(event.iteration)")
end

# Optional to implement
Report.title(reporter::YourReporter) = "Your section title"

calibration_report = Report.CalibrationReport(;
    reporters = (Report.ErrorReporter(), YourReporter()),
)
nothing # hide
```

The type of `event` decides when a section is shown. The method above is for
[`IterationEnd`](@ref), so its section is shown at the end of each iteration. To
show a section at another [event](@ref "Events"), such as a summary at
[`CalibrationEnd`](@ref), add a method for that event, and to show it at every
event, leave the type of `event` unspecified.

The sections of the report are in the same order as the reporters passed to
[`CalibrationReport`](@ref).

To format a section like the built-in reporters, pass a vector of
`label => value` pairs with string labels to [`print_aligned`](@ref). It prints
each pair as `label: value` and pads the labels so that the values line up.

### How are reports generated?

The arguments to [`report`](@ref) are

- `io`: Where to write the section. Write to `io` with `print` or `println`
  instead of logging with `@info` or printing to `stdout`, so that the section
  is part of the report.
- `event`: The [`AbstractEvent`](@ref) that is reported.
- `context`: The [`ReportContext`](@ref) of the calibration, which holds
  - `ekp`: The `EnsembleKalmanProcess`. At [`CalibrationStart`](@ref) and
    [`IterationStart`](@ref), it is the process before the iteration runs. At
    [`IterationEnd`](@ref) and [`CalibrationEnd`](@ref), it is the process
    after the update of the last iteration.
  - `prior`: The `ParameterDistribution` of the calibration.
  - `output_dir`: Directory to write logs or plots to.

[`CalibrationReport`](@ref) prints the title of the section and indents the
lines below it, so a reporter should not print its own title or indent its
lines.

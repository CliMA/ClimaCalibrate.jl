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

```@setup report
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import Logging

prior = PD.combine_distributions([
    PD.constrained_gaussian("a", 1.0, 0.5, -Inf, Inf),
    PD.constrained_gaussian("b", 2.0, 1.0, 0, Inf),
])
G(ϕ) = [ϕ[1] + ϕ[2], ϕ[1] - ϕ[2], 2ϕ[1]]
ekp = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, 10),
    G([1.5, 2.5]),
    [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01],
    EKP.Inversion();
    scheduler = EKP.DataMisfitController(on_terminate = "continue"),
)
```

The output of the first two iterations of a calibration looks like:

```@example report
Logging.with_logger(Logging.ConsoleLogger(stdout)) do # hide
    for _ in 1:2 # hide
        sleep(1) # hide
        ϕ = EKP.get_ϕ_final(prior, ekp) # hide
        EKP.update_ensemble!(ekp, reduce(hcat, G.(eachcol(ϕ)))) # hide
        Report.report(calibration_report, ekp, prior, ".") # hide
    end # hide
end # hide
nothing # hide
```

## Implementing your own AbstractReporter

To write your own reporter, subtype [`AbstractReporter`](@ref) and add a
method to [`report`](@ref), and optionally to [`title`](@ref).

```@example reporter
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import ClimaCalibrate.Report as Report

struct YourReporter <: Report.AbstractReporter end

function Report.report(
    io::IO,
    reporter::YourReporter,
    ekp::EKP.EnsembleKalmanProcess,
    prior::PD.ParameterDistribution,
    output_dir,
)
    println(io, "Your report")
end

# Optional to implement
Report.title(reporter::YourReporter) = "Your section title"

calibration_report = Report.CalibrationReport(;
    reporters = (Report.ErrorReporter(), YourReporter()),
)
nothing # hide
```

The sections of the report are in the same order as the reporters passed to
[`CalibrationReport`](@ref).

### How are reports generated?

The arguments to [`report`](@ref) are

- `io`: Where to write the section. Write to `io` with `print` or `println`
  instead of logging with `@info` or printing to `stdout`, so that the section
  is part of the report.
- `ekp`: The `EnsembleKalmanProcess` after the update of the current iteration.
- `prior`: The `ParameterDistribution` of the calibration.
- `output_dir`: Directory to write logs or plots to.

[`CalibrationReport`](@ref) prints the title of the section and indents the
lines below it, so a reporter should not print its own title or indent its
lines. Also, [`report`](@ref) is called once per iteration, after `ekp` is
updated with the forward model output of that iteration.

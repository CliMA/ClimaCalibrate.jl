using Test

import Dates
import LinearAlgebra
import Logging
import Random
import Statistics
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import ClimaCalibrate.Report as Report

# A small linear calibration problem, updated `n_updates` times
function make_ekp(n_updates; rng = Random.MersenneTwister(1234))
    prior = PD.combine_distributions([
        PD.constrained_gaussian("a", 1.0, 0.5, -Inf, Inf),
        PD.constrained_gaussian("b", 2.0, 1.0, 0, Inf),
    ])
    G(ϕ) = [ϕ[1] + ϕ[2], ϕ[1] - ϕ[2], 2ϕ[1]]
    ekp = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng, prior, 10),
        G([1.5, 2.5]),
        0.01 * LinearAlgebra.I(3),
        EKP.Inversion();
        rng,
        scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    )
    for _ in 1:n_updates
        ϕ = EKP.get_ϕ_final(prior, ekp)
        EKP.update_ensemble!(ekp, reduce(hcat, G.(eachcol(ϕ))))
    end
    return ekp, prior
end

# Split each line of a reporter's output into its label and its value, ignoring
# the padding that aligns the values
function parse_rows(output)
    return map(split(chomp(output), '\n')) do line
        label, value = split(line, ": "; limit = 2)
        return label => strip(value)
    end
end

# Matches a value followed by its change from the previous iteration
const CHANGE_REGEX = r"^(\S+) \(([+-]?\d+\.\d)% from previous\)$"

@testset "WalltimeReporter" begin
    # Test constructor
    reporter = Report.WalltimeReporter()
    @test reporter isa Report.AbstractReporter
    @test isnothing(reporter.n_iterations)
    @test length(reporter.times) == 1

    @test Report.WalltimeReporter(5).n_iterations == 5
    @test_throws ErrorException Report.WalltimeReporter(0)
    @test_throws ErrorException Report.WalltimeReporter(-1)

    # Test title function
    @test Report.title(Report.WalltimeReporter()) == "Walltime"

    # Test report without the total number of iterations
    ekp, prior = make_ekp(2)
    reporter = Report.WalltimeReporter()
    # Pretend the iteration took about an hour so the walltime is not zero
    reporter.times[1] -= 3600
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))

    # Helper function to get the human readable duration
    readable_duration(seconds) =
        string(Dates.canonicalize(Dates.Second(round(Int, seconds))))

    (; times) = reporter
    @test length(times) == 2
    # Only one iteration has passed, so the total is the iteration walltime
    walltime = readable_duration(times[2] - times[1])
    @test parse_rows(output) ==
          ["iteration walltime" => walltime, "total walltime" => walltime]

    # Test that the total walltime accumulates over iterations
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    (; times) = reporter
    @test parse_rows(output) == [
        "iteration walltime" => readable_duration(times[3] - times[2]),
        "total walltime" => readable_duration(times[3] - times[1]),
    ]

    # Test report with the total number of iterations
    ekp, prior = make_ekp(2)
    reporter = Report.WalltimeReporter(5)
    reporter.times[1] -= 3600
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    lines = split(chomp(output), '\n')
    @test length(lines) == 3

    (; times) = reporter
    walltime = readable_duration(times[2] - times[1])
    current = match(r"^iteration walltime: +(.+)$", lines[1])
    @test !isnothing(current)
    @test current[1] == walltime
    total = match(r"^total walltime: +(.+)$", lines[2])
    @test !isnothing(total)
    @test total[1] == walltime

    # Two of the five iterations are done, so three remain
    remaining = readable_duration(3 * (times[2] - times[1]))
    estimate = match(
        r"^walltime remaining for 3 iterations: (.+) \((\S+)\)$",
        lines[3],
    )
    @test !isnothing(estimate)
    @test estimate[1] == remaining
    @test Dates.DateTime(estimate[2]) > Dates.now()

    # Test that no estimate is shown when all iterations are done
    ekp, prior = make_ekp(2)
    reporter = Report.WalltimeReporter(2)
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    @test startswith(output, "iteration walltime:")
    @test !occursin("walltime remaining", output)
end

@testset "TimestepReporter" begin
    # Test constructor
    reporter = Report.TimestepReporter()
    @test reporter isa Report.AbstractReporter

    # Test title function
    @test Report.title(reporter) == "Timestep"

    # Test that nothing is reported before the first update
    ekp, prior = make_ekp(0)
    @test sprint(io -> Report.report(io, reporter, ekp, prior, "")) == ""

    # Test report
    ekp, prior = make_ekp(2)
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    Δts = EKP.get_Δt(ekp)
    algorithm_time = round(sum(Δts); sigdigits = 3)
    rounded_Δts = round.(Δts; sigdigits = 3)
    @test parse_rows(output) == [
        "current algorithm time" => string(algorithm_time),
        "current timestep size" => string(rounded_Δts[2]),
        "timestep history" => "[$(rounded_Δts[1]), $(rounded_Δts[2])]",
    ]
end

@testset "ErrorReporter" begin
    # Test constructor
    @test Report.ErrorReporter().error_metrics == ["loss", "unweighted_loss"]
    reporter = Report.ErrorReporter(; error_metrics = ["crps"])
    @test reporter isa Report.AbstractReporter
    @test reporter.error_metrics == ["crps"]
    @test_throws ErrorException Report.ErrorReporter(; error_metrics = ["los"])

    # Test title function
    @test Report.title(Report.ErrorReporter()) == "Error"

    # Test that nothing is reported before the first update
    reporter = Report.ErrorReporter()
    ekp, prior = make_ekp(0)
    @test sprint(io -> Report.report(io, reporter, ekp, prior, "")) == ""

    # Test report after one update, which has no previous error to compare to
    ekp, prior = make_ekp(1)
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    error_metrics = EKP.get_error_metrics(ekp)
    loss = round(only(error_metrics["loss"]); sigdigits = 3)
    unweighted_loss =
        round(only(error_metrics["unweighted_loss"]); sigdigits = 3)
    @test parse_rows(output) == [
        "current loss" => string(loss),
        "best loss" => "$loss (iteration 1)",
        "loss history" => "[$loss]",
        "current unweighted_loss" => string(unweighted_loss),
        "best unweighted_loss" => "$unweighted_loss (iteration 1)",
        "unweighted_loss history" => "[$unweighted_loss]",
    ]

    # Test report with the change from the previous iteration
    ekp, prior = make_ekp(2)
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    rows = parse_rows(output)
    @test first.(rows) == [
        "current loss",
        "best loss",
        "loss history",
        "current unweighted_loss",
        "best unweighted_loss",
        "unweighted_loss history",
    ]
    for (i, name) in enumerate(reporter.error_metrics)
        errors = EKP.get_error_metrics(ekp)[name]
        current = match(CHANGE_REGEX, last(rows[3i - 2]))
        @test !isnothing(current)
        @test current[1] == string(round(errors[2]; sigdigits = 3))
        change = round(100 * (errors[2] - errors[1]) / errors[1]; digits = 1)
        @test parse(Float64, current[2]) == change

        best_iteration = argmin(errors)
        best_error = round(errors[best_iteration]; sigdigits = 3)
        @test last(rows[3i - 1]) == "$best_error (iteration $best_iteration)"

        rounded_errors = round.(errors; sigdigits = 3)
        @test last(rows[3i]) == "[$(rounded_errors[1]), $(rounded_errors[2])]"
    end

    # Test that only the requested error metrics are reported
    reporter = Report.ErrorReporter(; error_metrics = ["crps"])
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    @test first.(parse_rows(output)) ==
          ["current crps", "best crps", "crps history"]
end

@testset "ParameterReporter" begin
    # Test constructor
    reporter = Report.ParameterReporter()
    @test reporter isa Report.AbstractReporter

    # Test title function
    @test Report.title(reporter) == "Parameter"

    # Test that nothing is reported before the first update
    ekp, prior = make_ekp(0)
    @test sprint(io -> Report.report(io, reporter, ekp, prior, "")) == ""

    # Test report
    ekp, prior = make_ekp(2)
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    rows = parse_rows(output)
    @test first.(rows) == ["a", "a spread", "b", "b spread"]

    # The mean and spread of the updated ensemble are compared to those of the
    # ensemble that was evaluated in the last iteration
    previous_means = EKP.get_ϕ_mean(prior, ekp, 2)
    means = EKP.get_ϕ_mean_final(prior, ekp)
    previous_spreads = Report._constrained_spread(ekp, prior, 2)
    spreads = Report._constrained_spread(ekp, prior, 3)
    for j in 1:2
        current = match(CHANGE_REGEX, last(rows[2j - 1]))
        @test !isnothing(current)
        @test current[1] == string(round(means[j]; sigdigits = 3))
        change = round(
            100 * (means[j] - previous_means[j]) / abs(previous_means[j]);
            digits = 1,
        )
        @test parse(Float64, current[2]) == change

        current = match(CHANGE_REGEX, last(rows[2j]))
        @test !isnothing(current)
        @test current[1] == string(round(spreads[j]; sigdigits = 3))
        change = round(
            100 * (spreads[j] - previous_spreads[j]) / previous_spreads[j];
            digits = 1,
        )
        @test parse(Float64, current[2]) == change
    end
end

@testset "EnsembleReporter" begin
    # Test constructor
    reporter = Report.EnsembleReporter()
    @test reporter isa Report.AbstractReporter

    # Test title function
    @test Report.title(reporter) == "Ensemble"

    # Test that nothing is reported before the first update
    ekp, prior = make_ekp(0)
    @test sprint(io -> Report.report(io, reporter, ekp, prior, "")) == ""

    # Test report when every member succeeds
    ekp, prior = make_ekp(1)
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    @test output == "successful members: 10 of 10\n"

    # Test report when members 3 and 7 fail
    g = copy(EKP.get_g_final(ekp))
    g[:, [3, 7]] .= NaN
    EKP.update_ensemble!(ekp, g)
    output = sprint(io -> Report.report(io, reporter, ekp, prior, ""))
    @test parse_rows(output) ==
          ["successful members" => "8 of 10", "failed members" => "3, 7"]
end

@testset "Calibration report" begin
    # A reporter that prints fixed lines and the output directory it is given,
    # and counts how many times it is called
    mutable struct CountingReporter <: Report.AbstractReporter
        n_calls::Int
    end

    CountingReporter() = CountingReporter(0)

    Report.title(::CountingReporter) = "Counting"

    function Report.report(io::IO, reporter::CountingReporter, _, _, output_dir)
        reporter.n_calls += 1
        println(io, "first line")
        println(io, "output directory: $output_dir")
        return nothing
    end

    # Log the report with a `TestLogger` and return the logged records
    function log_report(calibration_report, ekp, prior, output_dir)
        logger = Test.TestLogger()
        Logging.with_logger(logger) do
            Report.report(calibration_report, ekp, prior, output_dir)
        end
        return logger.logs
    end

    # Test constructor
    calibration_report = Report.CalibrationReport()
    @test typeof.(calibration_report.reporters) == (
        Report.WalltimeReporter,
        Report.TimestepReporter,
        Report.ErrorReporter,
        Report.ParameterReporter,
        Report.EnsembleReporter,
    )

    reporters = (Report.ErrorReporter(), Report.EnsembleReporter())
    @test Report.CalibrationReport(; reporters).reporters == reporters
    @test Report.CalibrationReport(; reporters = ()).reporters == ()

    # Test that the report is logged as a single message with a header
    ekp, prior = make_ekp(2)
    reporter = CountingReporter()
    calibration_report = Report.CalibrationReport(; reporters = (reporter,))
    logs = log_report(calibration_report, ekp, prior, "output")
    @test length(logs) == 1
    @test only(logs).level == Logging.Info

    lines = split(only(logs).message, '\n')
    header = match(r"^Calibration report · iteration (\d+) · (\S+)$", lines[1])
    @test !isnothing(header)
    @test parse(Int, header[1]) == EKP.get_N_iterations(ekp)
    date = Dates.DateTime(header[2])
    @test date == trunc(date, Dates.Second)

    # Test that each reporter is called once and indentation is correct
    @test lines[2:end] ==
          ["  Counting", "    first line", "    output directory: output"]
    @test reporter.n_calls == 1

    # Test that a report without reporters is only the header
    calibration_report = Report.CalibrationReport(; reporters = ())
    logs = log_report(calibration_report, ekp, prior, "")
    lines = split(chomp(only(logs).message), '\n')
    @test length(lines) == 1
    @test startswith(only(lines), "Calibration report")

    # Test that the default report works before the first update
    ekp, prior = make_ekp(0)
    logs = log_report(Report.CalibrationReport(), ekp, prior, "")
    lines = split(only(logs).message, '\n')
    @test startswith(lines[1], "Calibration report · iteration 0 · ")
    titles = [strip(line) for line in lines if occursin(r"^  \S", line)]
    @test titles == ["Walltime", "Timestep", "Error", "Parameter", "Ensemble"]
end

@testset "Helper functions" begin
    # Test _constrained_spread
    ekp, prior = make_ekp(2)
    for iteration in 1:3
        ϕ = EKP.get_ϕ(prior, ekp, iteration)
        @test Report._constrained_spread(ekp, prior, iteration) ≈
              Statistics.std.(eachrow(ϕ))
    end

    # The weighted spread of the initial sigma points is the standard deviation
    # of the prior, unlike the sample standard deviation of the sigma points
    uki_prior = PD.constrained_gaussian("c", 1.0, 0.5, -Inf, Inf; repeats = 5)
    uki = EKP.EnsembleKalmanProcess(
        [1.0, 2.0],
        0.01 * LinearAlgebra.I(2),
        EKP.Unscented(uki_prior),
    )
    @test isapprox(
        Report._constrained_spread(uki, uki_prior, 1),
        fill(0.5, 5);
        rtol = 1e-6,
    )

    # Test _print_aligned
    @test sprint(io -> Report._print_aligned(io, Pair{String, String}[])) == ""
    rows = ["a" => "1", "long label" => "2", "ϕ" => "3"]
    @test sprint(io -> Report._print_aligned(io, rows)) == """
        a:          1
        long label: 2
        ϕ:          3
        """

    # Test _format_recent
    @test Report._format_recent(Float64[]) == "[]"
    @test Report._format_recent([1.0, 2.0]) == "[1.0, 2.0]"
    @test Report._format_recent(collect(1.0:5.0)) == "[1.0, 2.0, 3.0, 4.0, 5.0]"
    @test Report._format_recent(collect(1.0:7.0)) ==
          "[…, 3.0, 4.0, 5.0, 6.0, 7.0]"
    @test Report._format_recent(collect(1.0:3.0); n = 2) == "[…, 2.0, 3.0]"
    @test Report._format_recent([0.98765, 1234.5678]) == "[0.988, 1230.0]"

    # Test _format_change
    @test Report._format_change(Float64[]) == ""
    @test Report._format_change([1.0]) == ""
    @test Report._format_change([0.0, 1.0]) == ""
    @test Report._format_change([2.0, 3.0]) == " (+50.0% from previous)"
    @test Report._format_change([4.0, 3.0]) == " (-25.0% from previous)"
    @test Report._format_change([2.0, 2.0]) == " (0.0% from previous)"
    @test Report._format_change([3.0, 4.0]) == " (+33.3% from previous)"
    # Only the last two values are compared
    @test Report._format_change([100.0, 2.0, 3.0]) == " (+50.0% from previous)"

    # Test _parameter_labels
    @test Report._parameter_labels(prior) == ["a", "b"]
    @test Report._parameter_labels(uki_prior) == ["c[$k]" for k in 1:5]
    prior = PD.combine_distributions([
        PD.constrained_gaussian("a", 1.0, 0.5, -Inf, Inf),
        PD.constrained_gaussian("c", 0.0, 1.0, -Inf, Inf; repeats = 3),
    ])
    @test Report._parameter_labels(prior) == ["a", "c[1]", "c[2]", "c[3]"]
end

using Test

import Dates
import LinearAlgebra
import Logging
import Random
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD
import ClimaCalibrate as CAL
import ClimaCalibrate.Report as Report

# A small linear calibration problem, updated `n_updates` times
function make_ekp(n_updates; rng = Random.MersenneTwister(1234))
    # A 4-D correlated variable
    correlated = PD.ParameterDistribution(
        Dict(
            "distribution" => PD.Parameterized(
                PD.Distributions.MvNormal(zeros(4), 0.1 * LinearAlgebra.I),
            ),
            "constraint" => [
                PD.no_constraint(),
                PD.bounded_below(-1.0),
                PD.bounded_above(0.4),
                PD.bounded(-0.1, 0.2),
            ],
            "name" => "c",
        ),
    )
    # A 2-D variable given only by samples
    sampled = PD.ParameterDistribution(
        Dict(
            "distribution" =>
                PD.Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0]),
            "constraint" => [PD.bounded(10, 15), PD.no_constraint()],
            "name" => "d",
        ),
    )
    prior = PD.combine_distributions([
        PD.constrained_gaussian("a", 1.0, 0.5, -Inf, Inf),
        PD.constrained_gaussian("b", 2.0, 1.0, 0, Inf),
        correlated,
        sampled,
        PD.constrained_gaussian("e", 2.0, 1.0, 0, Inf; repeats = 3),
    ])
    G(ϕ) = [ϕ[1] + ϕ[2]; ϕ[1] - ϕ[2]; 2ϕ[1]; ϕ[3:end]]
    ekp = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng, prior, 10),
        G([1.5, 2.5, 0.2, 0.5, -0.3, 0.1, 12.0, 11.0, 2.5, 1.5, 3.0]),
        0.01 * LinearAlgebra.I(12),
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

# Return what `reporter` prints at the end of the latest iteration of the ekp in
# `context`
function report_iteration_end(reporter, context)
    event = Report.IterationEnd(EKP.get_N_iterations(context.ekp))
    return sprint(io -> Report.report(io, reporter, event, context))
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
    context = Report.ReportContext(ekp, prior, "")
    reporter = Report.WalltimeReporter()
    # Pretend the iteration took about an hour so the walltime is not zero
    reporter.times[1] -= 3600
    output = report_iteration_end(reporter, context)

    (; times) = reporter
    @test length(times) == 2
    # Only one iteration has passed, so the total is the iteration walltime
    walltime = Report._format_duration(times[2] - times[1])
    @test parse_rows(output) ==
          ["iteration walltime" => walltime, "total walltime" => walltime]

    # Test that the total walltime accumulates over iterations
    output = report_iteration_end(reporter, context)
    (; times) = reporter
    @test parse_rows(output) == [
        "iteration walltime" => Report._format_duration(times[3] - times[2]),
        "total walltime" => Report._format_duration(times[3] - times[1]),
    ]

    # Test report with the total number of iterations
    ekp, prior = make_ekp(2)
    context = Report.ReportContext(ekp, prior, "")
    reporter = Report.WalltimeReporter(5)
    reporter.times[1] -= 3600
    output = report_iteration_end(reporter, context)
    lines = split(chomp(output), '\n')
    @test length(lines) == 3

    (; times) = reporter
    walltime = Report._format_duration(times[2] - times[1])
    current = match(r"^iteration walltime: +(.+)$", lines[1])
    @test !isnothing(current)
    @test current[1] == walltime
    total = match(r"^total walltime: +(.+)$", lines[2])
    @test !isnothing(total)
    @test total[1] == walltime

    # Two of the five iterations are done, so three remain
    remaining = Report._format_duration(3 * (times[2] - times[1]))
    estimate = match(
        r"^walltime remaining for 3 iterations: (.+) \((\S+)\)$",
        lines[3],
    )
    @test !isnothing(estimate)
    @test estimate[1] == remaining
    @test Dates.DateTime(estimate[2]) > Dates.now()

    # Test that no estimate is shown when all iterations are done
    ekp, prior = make_ekp(2)
    context = Report.ReportContext(ekp, prior, "")
    reporter = Report.WalltimeReporter(2)
    output = report_iteration_end(reporter, context)
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
    context = Report.ReportContext(ekp, prior, "")
    @test report_iteration_end(reporter, context) == ""

    # Test report
    ekp, prior = make_ekp(2)
    context = Report.ReportContext(ekp, prior, "")
    output = report_iteration_end(reporter, context)
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
    context = Report.ReportContext(ekp, prior, "")
    @test report_iteration_end(reporter, context) == ""

    # Test report after one update, which has no previous error to compare to
    ekp, prior = make_ekp(1)
    context = Report.ReportContext(ekp, prior, "")
    output = report_iteration_end(reporter, context)
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
    context = Report.ReportContext(ekp, prior, "")
    output = report_iteration_end(reporter, context)
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
    output = report_iteration_end(reporter, context)
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
    context = Report.ReportContext(ekp, prior, "")
    @test report_iteration_end(reporter, context) == ""

    # Test report
    ekp, prior = make_ekp(2)
    context = Report.ReportContext(ekp, prior, "")
    output = report_iteration_end(reporter, context)
    rows = parse_rows(output)
    labels = [
        "a",
        "b",
        "c[1]",
        "c[2]",
        "c[3]",
        "c[4]",
        "d[1]",
        "d[2]",
        "e[1]",
        "e[2]",
        "e[3]",
    ]
    @test first.(rows) ==
          reduce(vcat, [[label, "$label spread"] for label in labels])

    # The mean and spread of the updated ensemble are compared to those of the
    # ensemble that was evaluated in the last iteration
    previous_means = EKP.get_ϕ_mean(prior, ekp, 2)
    means = EKP.get_ϕ_mean_final(prior, ekp)
    previous_spreads = CAL.EKPUtils._constrained_spread(ekp, prior, 2)
    spreads = CAL.EKPUtils._constrained_spread(ekp, prior, 3)
    for j in eachindex(labels)
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
    context = Report.ReportContext(ekp, prior, "")
    @test report_iteration_end(reporter, context) == ""

    # Test report when every member succeeds
    ekp, prior = make_ekp(1)
    context = Report.ReportContext(ekp, prior, "")
    output = report_iteration_end(reporter, context)
    @test output == "successful members: 10 of 10\n"

    # Test report when members 3 and 7 fail
    g = copy(EKP.get_g_final(ekp))
    g[:, [3, 7]] .= NaN
    EKP.update_ensemble!(ekp, g)
    output = report_iteration_end(reporter, context)
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

    function Report.report(
        io::IO,
        reporter::CountingReporter,
        ::Report.IterationEnd,
        context,
    )
        reporter.n_calls += 1
        println(io, "first line")
        println(io, "output directory: $(context.output_dir)")
        return nothing
    end

    # Log the report of the end of the latest iteration with a `TestLogger` and
    # return the logged records
    function log_report(calibration_report, context)
        event = Report.IterationEnd(EKP.get_N_iterations(context.ekp))
        logger = Test.TestLogger()
        Logging.with_logger(logger) do
            Report.report(calibration_report, event, context)
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
    context = Report.ReportContext(ekp, prior, "output")
    reporter = CountingReporter()
    calibration_report = Report.CalibrationReport(; reporters = (reporter,))
    logs = log_report(calibration_report, context)
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

    # Test that a report without reporters logs nothing
    calibration_report = Report.CalibrationReport(; reporters = ())
    logs = log_report(calibration_report, context)
    @test isempty(logs)

    # Test that the default report works before the first update
    ekp, prior = make_ekp(0)
    context = Report.ReportContext(ekp, prior, "")
    logs = log_report(Report.CalibrationReport(), context)
    lines = split(only(logs).message, '\n')
    @test startswith(lines[1], "Calibration report · iteration 0 · ")
    titles = [strip(line) for line in lines if occursin(r"^  \S", line)]
    # The other reporters print nothing before the first update
    @test titles == ["Walltime"]

    # Test that every reporter of the default report prints a section after the
    # first update
    ekp, prior = make_ekp(1)
    context = Report.ReportContext(ekp, prior, "")
    logs = log_report(Report.CalibrationReport(), context)
    lines = split(only(logs).message, '\n')
    @test startswith(lines[1], "Calibration report · iteration 1 · ")
    titles = [strip(line) for line in lines if occursin(r"^  \S", line)]
    @test titles == ["Walltime", "Timestep", "Error", "Parameter", "Ensemble"]
end

@testset "ReportContext" begin
    # Test constructor
    ekp, prior = make_ekp(1)
    context = Report.ReportContext(ekp, prior, "output")
    @test context.ekp === ekp
    @test context.prior === prior
    @test context.output_dir == "output"
end

@testset "Events" begin
    # A reporter that only reports at the start of an iteration
    struct IterationStartReporter <: Report.AbstractReporter end

    function Report.report(
        io::IO,
        ::IterationStartReporter,
        event::Report.IterationStart,
        context,
    )
        println(io, "start of iteration $(event.iteration)")
        return nothing
    end

    # A reporter that reports on every event and reports something else at the
    # end of an iteration
    struct AllEventsReporter <: Report.AbstractReporter end

    function Report.report(io, ::AllEventsReporter, event, context)
        println(io, "event: $(nameof(typeof(event)))")
        return nothing
    end

    function Report.report(
        io,
        ::AllEventsReporter,
        ::Report.IterationEnd,
        context,
    )
        println(io, "end of iteration")
        return nothing
    end

    # A reporter that counts the events it is given and prints nothing
    mutable struct SilentReporter <: Report.AbstractReporter
        n_calls::Int
    end

    function Report.report(io, reporter::SilentReporter, event, context)
        reporter.n_calls += 1
        return nothing
    end

    # Log the report of `event` with a `TestLogger` and return the logged
    # records
    function log_event_report(calibration_report, event, context)
        logger = Test.TestLogger()
        Logging.with_logger(logger) do
            Report.report(calibration_report, event, context)
        end
        return logger.logs
    end

    ekp, prior = make_ekp(1)
    context = Report.ReportContext(ekp, prior, "")
    silent_reporter = SilentReporter(0)
    calibration_report = Report.CalibrationReport(;
        reporters = (
            IterationStartReporter(),
            AllEventsReporter(),
            silent_reporter,
        ),
    )

    # Test the header of each event and that only the reporters that print
    # something for the event have a section
    expected = [
        Report.CalibrationStart(1) => (
            "start of calibration",
            ["  AllEventsReporter", "    event: CalibrationStart"],
        ),
        Report.IterationStart(2) => (
            "start of iteration 2",
            [
                "  IterationStartReporter",
                "    start of iteration 2",
                "  AllEventsReporter",
                "    event: IterationStart",
            ],
        ),
        Report.IterationEnd(2) => (
            "iteration 2",
            ["  AllEventsReporter", "    end of iteration"],
        ),
        Report.CalibrationEnd(2) => (
            "end of calibration",
            ["  AllEventsReporter", "    event: CalibrationEnd"],
        ),
    ]
    for (event, (description, sections)) in expected
        logs = log_event_report(calibration_report, event, context)
        lines = split(only(logs).message, '\n')
        @test startswith(lines[1], "Calibration report · $description · ")
        @test lines[2:end] == sections
    end

    # Test that every reporter is given every event
    @test silent_reporter.n_calls == length(expected)

    # Test that nothing is logged when every reporter prints nothing
    calibration_report =
        Report.CalibrationReport(; reporters = (SilentReporter(0),))
    logs = log_event_report(calibration_report, Report.IterationEnd(2), context)
    @test isempty(logs)

    # Test that the built-in reporters only report at the end of an iteration
    calibration_report = Report.CalibrationReport()
    for event in (
        Report.CalibrationStart(1),
        Report.IterationStart(2),
        Report.CalibrationEnd(2),
    )
        @test isempty(log_event_report(calibration_report, event, context))
    end
end

@testset "Events of a calibration" begin
    # A model whose output is its own parameter
    struct ParameterModelInterface <: CAL.AbstractModelInterface
        output_dir::String
        prior::PD.ParameterDistribution
    end

    CAL.forward_model(::ParameterModelInterface, iteration, member) = nothing

    function CAL.observation_map(interface::ParameterModelInterface, iteration)
        (; output_dir, prior) = interface
        ekp = CAL.load_ekp_struct(output_dir, iteration)
        return EKP.get_ϕ_final(prior, ekp)
    end

    # A reporter that records the events it is given and prints nothing
    struct EventRecorder <: Report.AbstractReporter
        events::Vector{Report.AbstractEvent}
    end

    function Report.report(io, recorder::EventRecorder, event, context)
        push!(recorder.events, event)
        return nothing
    end

    prior = PD.constrained_gaussian("a", 10.0, 5.0, 0, Inf)

    function make_calibration_ekp(; kwargs...)
        rng = Random.MersenneTwister(1234)
        return EKP.EnsembleKalmanProcess(
            EKP.construct_initial_ensemble(rng, prior, 20),
            [20.0],
            [0.01;;],
            EKP.Inversion();
            rng,
            kwargs...,
        )
    end

    # Run `n_iterations` of the calibration in `output_dir` and return the
    # events that are reported
    function record_events(ekp, n_iterations, output_dir)
        recorder = EventRecorder(Report.AbstractEvent[])
        CAL.calibrate(
            CAL.JuliaBackend(),
            ekp,
            ParameterModelInterface(output_dir, prior),
            n_iterations,
            prior,
            output_dir;
            calibration_report = Report.CalibrationReport(;
                reporters = (recorder,),
            ),
        )
        return recorder.events
    end

    # Test the events of a calibration of two iterations
    ekp = make_calibration_ekp()
    output_dir = mktempdir()
    @test record_events(ekp, 2, output_dir) == [
        Report.CalibrationStart(1),
        Report.IterationStart(1),
        Report.IterationEnd(1),
        Report.IterationStart(2),
        Report.IterationEnd(2),
        Report.CalibrationEnd(2),
    ]

    # Test that a resumed calibration starts at the first iteration that is not
    # completed
    @test record_events(ekp, 3, output_dir) == [
        Report.CalibrationStart(3),
        Report.IterationStart(3),
        Report.IterationEnd(3),
        Report.CalibrationEnd(3),
    ]

    # Test that nothing is reported when there is nothing left to run
    @test isempty(record_events(ekp, 3, output_dir))

    # Test that the iteration where the scheduler terminates the calibration
    # has no end of iteration, since the ensemble is not updated
    ekp = make_calibration_ekp(
        scheduler = EKP.DataMisfitController(terminate_at = 1),
    )
    output_dir = mktempdir()
    events = record_events(ekp, 20, output_dir)
    stopped_at = CAL.terminated_iteration(output_dir)
    @test !isnothing(stopped_at)
    expected = Report.AbstractEvent[Report.CalibrationStart(1)]
    for iter in 1:(stopped_at - 1)
        push!(expected, Report.IterationStart(iter), Report.IterationEnd(iter))
    end
    push!(
        expected,
        Report.IterationStart(stopped_at),
        Report.CalibrationEnd(stopped_at - 1),
    )
    @test events == expected
end

@testset "Helper functions" begin
    # Test print_aligned
    @test sprint(io -> Report.print_aligned(io, Pair{String, String}[])) == ""
    rows = ["a" => "1", "long label" => "2", "ϕ" => "3"]
    @test sprint(io -> Report.print_aligned(io, rows)) == """
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

    # Test _format_duration
    @test Report._format_duration(0) == "0 seconds"
    @test Report._format_duration(0.4) == "0 seconds"
    @test Report._format_duration(0.6) == "1 second"
    @test Report._format_duration(59) == "59 seconds"
    @test Report._format_duration(3725) == "1 hour, 2 minutes, 5 seconds"
    @test Report._format_duration(90000) == "1 day, 1 hour"

    # Test _parameter_labels
    _, prior = make_ekp(0)
    @test Report._parameter_labels(prior) == [
        "a",
        "b",
        "c[1]",
        "c[2]",
        "c[3]",
        "c[4]",
        "d[1]",
        "d[2]",
        "e[1]",
        "e[2]",
        "e[3]",
    ]
    repeated_prior =
        PD.constrained_gaussian("c", 1.0, 0.5, -Inf, Inf; repeats = 5)
    @test Report._parameter_labels(repeated_prior) == ["c[$k]" for k in 1:5]
    prior = PD.combine_distributions([
        PD.constrained_gaussian("a", 1.0, 0.5, -Inf, Inf),
        PD.constrained_gaussian("c", 0.0, 1.0, -Inf, Inf; repeats = 3),
    ])
    @test Report._parameter_labels(prior) == ["a", "c[1]", "c[2]", "c[3]"]
end

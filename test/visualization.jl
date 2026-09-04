using Test
import Dates
import CairoMakie
import ClimaCalibrate
import ClimaCalibrate: ObservationRecipe, EnsembleBuilder
import ClimaCalibrate.SampleBuilder: build_samples_by_times
import ClimaAnalysis
import ClimaAnalysis.Template:
    TemplateVar,
    make_template_var,
    add_attribs,
    add_dim,
    add_data,
    one_to_n_data,
    initialize
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions

@testset "Visualization of G ensemble matrix and observations" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    neg_var = -2.0 * var
    ClimaAnalysis.set_short_name!(neg_var, "neg_hi")

    covar_estimator = ObservationRecipe.ScalarCovariance(; scalar = 1.0)

    start_date = Dates.DateTime(2007, 12)
    end_date = start_date + Dates.Second(time[end])
    sample_collection =
        build_samples_by_times([var, neg_var], [(start_date, end_date)])
    obs = ObservationRecipe.observation(covar_estimator, sample_collection, 1)

    obs_series = EKP.ObservationSeries(
        Dict(
            "observations" => [obs],
            "names" => ["1"],
            "minibatcher" =>
                ClimaCalibrate.minibatcher_over_samples([1], 1),
        ),
    )

    prior = constrained_gaussian("pi_groups_coeff", 1.0, 0.3, 0, Inf)


    ekp = EKP.EnsembleKalmanProcess(
        obs_series,
        EKP.TransformUnscented(prior, impose_prior = true),
        verbose = true,
        scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    )

    iters = 3
    for i in 1:iters
        g_ens_builder = EnsembleBuilder.GEnsembleBuilder(ekp)
        for j in 1:EKP.get_N_ens(ekp)
            EnsembleBuilder.fill_g_ens_col!(g_ens_builder, j, Float64(j))
        end
        g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
        g_ens .+= i
        EKP.update_ensemble!(ekp, g_ens)
    end

    # Create directory to save plots to
    plot_dir = mktempdir()

    # Test non mutating version
    fig = CairoMakie.Figure(; size = (750, 1000))
    plot_fns = [
        ClimaCalibrate.Visualization.plot_g,
        ClimaCalibrate.Visualization.plot_g_mean,
        ClimaCalibrate.Visualization.plot_obs,
    ]
    colors = [:tomato, :lime, :skyblue]
    for (i, (plot_fn, color)) in enumerate(zip(plot_fns, colors))
        plot_fn(fig[i, 1], ekp; iter = 2, color)
        plot_fn(fig[i, 2], ekp; color)
    end
    plot_path = joinpath(plot_dir, "g_ensemble_and_obs_plot.png")
    CairoMakie.save(plot_path, fig)
    @test isfile(plot_path)

    # Test mutating version
    fig = CairoMakie.Figure(; size = (1000, 500))
    ax1 = CairoMakie.Axis(
        fig[1, 1],
        xlabel = "Index",
        ylabel = "Value",
        title = "G ensemble members and observations for 2nd iteration",
    )
    ax2 = CairoMakie.Axis(
        fig[1, 2],
        xlabel = "Index",
        ylabel = "Value",
        title = "G ensemble members and observations for last iteration",
    )
    mutating_plot_fns = [
        ClimaCalibrate.Visualization.plot_g!,
        ClimaCalibrate.Visualization.plot_g_mean!,
        ClimaCalibrate.Visualization.plot_obs!,
    ]
    colors = [:tomato, :lime, :skyblue]
    for (i, (mutating_plot_fn, color)) in
        enumerate(zip(mutating_plot_fns, colors))
        mutating_plot_fn(ax1, ekp; iter = 2, color)
        mutating_plot_fn(ax2, ekp; color)
    end
    mutating_plot_path =
        joinpath(plot_dir, "g_ensemble_and_obs_plot_from_mutating_axes.png")
    CairoMakie.save(mutating_plot_path, fig)
    @test isfile(mutating_plot_path)

    # Saving a figure only checks that plotting does not throw. These check
    # that each recipe draws the data it was asked for
    @testset "What is plotted" begin
        # Each of these is a Makie recipe, so the lines are the children of the
        # single recipe plot on the axis
        function plotted_lines(plot_fn, ekp; kwargs...)
            ax = CairoMakie.Axis(CairoMakie.Figure()[1, 1])
            plot_fn(ax, ekp; kwargs...)
            recipe = only(ax.scene.plots)
            return filter(p -> p isa CairoMakie.Lines, recipe.plots)
        end
        # A `Lines` holds a vector of points; take their y coordinates
        values_of(line) = last.(line[1][])

        obs_length = length(EKP.get_obs(ekp))

        # One line per ensemble member, each holding that member's column
        lines =
            plotted_lines(ClimaCalibrate.Visualization.plot_g!, ekp; iter = 2)
        @test length(lines) == EKP.get_N_ens(ekp)
        expected = EKP.get_g(ekp, 2)
        for (j, line) in enumerate(lines)
            @test values_of(line) ≈ expected[:, j]
        end

        lines = plotted_lines(
            ClimaCalibrate.Visualization.plot_g_mean!,
            ekp;
            iter = 2,
        )
        @test length(lines) == 1
        @test values_of(only(lines)) ≈ vec(EKP.get_g_mean(ekp, 2))

        lines =
            plotted_lines(ClimaCalibrate.Visualization.plot_obs!, ekp; iter = 2)
        @test length(lines) == 1
        @test length(values_of(only(lines))) == obs_length

        # Omitting `iter` should plot the last completed iteration
        lines = plotted_lines(ClimaCalibrate.Visualization.plot_g_mean!, ekp)
        @test values_of(only(lines)) ≈
              vec(EKP.get_g_mean(ekp, EKP.get_N_iterations(ekp)))
    end
end

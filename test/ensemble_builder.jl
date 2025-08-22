using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.EnsembleBuilder
import EnsembleKalmanProcesses
import OrderedCollections: OrderedDict
import Statistics: mean
import LinearAlgebra: Diagonal, I
import Statistics
import NaNStatistics: nanvar, nanmean

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions

# Since functions not defined in ext are not exported, we need to access
# them like this
ext = Base.get_extension(ClimaCalibrate, :ClimaAnalysisExt)

import ClimaAnalysis.Template:
    TemplateVar,
    make_template_var,
    add_attribs,
    add_dim,
    add_time_dim,
    add_lon_dim,
    add_lat_dim,
    add_data,
    ones_data,
    zeros_data,
    one_to_n_data,
    initialize

@testset "Is compatible with metadata" begin
    make_metadata(var) = ClimaAnalysis.flatten(var).metadata
    lat = [-90.0, 0.0, 90.0]
    var =
        TemplateVar() |>
        add_dim("lat", lat, units = "degrees_east") |>
        add_attribs(
            short_name = "hey",
            long_name = "hello",
            start_date = "2007-12-1",
            units = "m",
        ) |>
        initialize

    # Check short names
    attribs = Dict("short_name" => "hi")
    diff_short_name_var = ClimaAnalysis.remake(var, attributes = attribs)
    @test !ext._same_short_names(var, make_metadata(diff_short_name_var))

    # Check set of dimensions are the same
    dims = Dict("lon" => lat)
    diff_dim_var = ClimaAnalysis.remake(var, dims = dims)
    @test !ext._same_dim_names(var, make_metadata(diff_dim_var))

    # Check units are the same
    attribs = Dict("short_name" => "hi", "units" => "m^2")
    diff_units_var = ClimaAnalysis.remake(var, attributes = attribs)
    @test !ext._same_units(var, make_metadata(diff_units_var))

    # Check units of dimensions are the same
    diff_dim_units_var =
        TemplateVar() |>
        add_dim("lat", lat, units = "degrees_west") |>
        add_attribs(short_name = "hey") |>
        initialize
    @test !ext._same_dim_units(var, make_metadata(diff_dim_units_var))

    # Check values of nontemporal dimensions are the same
    diff_lat = [-80.0, 0.0, 80.0]
    var =
        TemplateVar() |>
        add_dim("lat", diff_lat, units = "degrees_west") |>
        add_attribs(short_name = "hey") |>
        initialize
    @test !ext._compatible_dims_values(var, make_metadata(diff_dim_units_var))

    # Check temporal dimension
    date_var1 =
        TemplateVar() |>
        add_dim("time", [0.0, 1.0, 2.0], units = "s") |>
        add_attribs(start_date = "2010-12-01T00:00:42") |>
        initialize
    date_var2 =
        TemplateVar() |>
        add_dim("time", [-2.0, -1.0, 0.0], units = "s") |>
        add_attribs(start_date = "2010-12-01T00:00:44") |>
        initialize
    @test ext._compatible_dims_values(date_var1, make_metadata(date_var2))
end

@testset "Match dates" begin
    time = [0.0, 1.0, 2.0, 3.0]
    lat = [-90.0, 0.0, 90.0]
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lat", lat, units = "degrees_east") |>
        add_attribs(
            short_name = "hey",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        initialize

    # Match dates with itself
    metadata = ClimaAnalysis.flatten(var).metadata
    matched_var = ext._match_dates(var, metadata)
    @test ClimaAnalysis.dates(matched_var) == ClimaAnalysis.dates(var)
    @test matched_var.data == var.data
    @test matched_var.dims == var.dims
    @test matched_var.attributes == var.attributes

    # Match dates when dates in var is a superset of the dates in metadata
    less_time_var =
        TemplateVar() |>
        add_dim("time", [0.0, 3.0], units = "s") |>
        add_attribs(
            short_name = "hey",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        initialize
    metadata = ClimaAnalysis.flatten(less_time_var).metadata
    matched_var = ext._match_dates(var, metadata)
    @test ClimaAnalysis.dates(matched_var) == ClimaAnalysis.dates(less_time_var)
    @test matched_var.data == var.data[[1, 4], :]
    @test all(
        matched_var.dims[dim] == var.dims[dim] for
        dim in keys(matched_var.dims) if dim != "time"
    )
    @test matched_var.attributes == var.attributes

    # Match dates when dates in var is a subset of the dates in metadata
    more_time_var =
        TemplateVar() |>
        add_dim("time", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], units = "s") |>
        add_attribs(
            short_name = "hey",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        initialize
    metadata = ClimaAnalysis.flatten(more_time_var).metadata
    @test_throws ErrorException ext._match_dates(var, metadata)

    # Times in not sorted order
    odd_time_var =
        TemplateVar() |>
        add_dim("time", [2.0, 0.0, 3.0, 1.0], units = "s") |>
        add_attribs(
            short_name = "hey",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        initialize
    metadata = ClimaAnalysis.flatten(odd_time_var).metadata
    matched_var = ext._match_dates(var, metadata)
    @test ClimaAnalysis.dates(matched_var) == ClimaAnalysis.dates(odd_time_var)
    @test matched_var.data == var.data[[3, 1, 4, 2], :]
    @test all(
        matched_var.dims[dim] == var.dims[dim] for
        dim in keys(matched_var.dims) if dim != "time"
    )
    @test matched_var.attributes == var.attributes
end

@testset "Use GEnsembleBuilder for a fake calibration" begin
    pkgversion(EnsembleKalmanProcesses) > v"2.4.3" || return

    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(3 * i) for i in 0:11],
        )
    time_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hey",
            long_name = "hello",
            start_date = "2007-12-1",
            units = "m",
        ) |>
        one_to_n_data() |>
        initialize

    lon = [-45.0, 0.0, 45.0]
    lon_var =
        TemplateVar() |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
            super = "fun",
            units = "m",
        ) |>
        one_to_n_data() |>
        initialize

    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()

    obs_vec = [
        ObservationRecipe.observation(
            covar_estimator,
            vars,
            start_date,
            end_date,
        ) for (vars, start_date, end_date) in [
            ((time_var, lon_var), "2007-12-1", "2008-9-1"),
            ((lon_var, time_var), "2008-12-1", "2009-9-1"),
            ((time_var,), "2009-12-1", "2010-9-1"),
        ]
    ]
    obs_series = EKP.ObservationSeries(
        Dict(
            "observations" => obs_vec,
            "names" => ["1", "2", "3"],
            "minibatcher" =>
                ClimaCalibrate.minibatcher_over_samples([1, 2, 3], 1),
        ),
    )

    prior = constrained_gaussian("pi_groups_coeff", 1.0, 0.3, 0, Inf)
    eki = EKP.EnsembleKalmanProcess(
        obs_series,
        EKP.TransformUnscented(prior, impose_prior = true),
        verbose = true,
        scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    )

    g_ens_builder = EnsembleBuilder.GEnsembleBuilder(eki)
    g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
    @test size(g_ens) == (16, EKP.get_N_ens(eki))
    @test size(g_ens_builder.completed) == (2, EKP.get_N_ens(eki))
    @test keys(g_ens_builder.metadata_by_short_name) == Set(["hi", "hey"])

    function window_and_flatten(var, start_date, end_date)
        return ClimaAnalysis.flatten(
            ClimaAnalysis.window(
                var,
                "time",
                left = start_date,
                right = end_date,
            ),
        )
    end

    function one_to_n_matrix(matrix)
        return reshape(1:prod(size(matrix)), size(matrix))
    end

    # First minibatch
    for i in 1:3
        @test !EnsembleBuilder.is_complete(g_ens_builder)
        # Permute dimensions to test that the dimensions are permuted correctly
        # before flattening
        EnsembleBuilder.fill_g_ens_col!(
            g_ens_builder,
            i,
            time_var,
            permutedims(lon_var, ("time", "lon")),
        )
    end
    @test EnsembleBuilder.is_complete(g_ens_builder)
    flat_time_var = window_and_flatten(
        time_var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )
    flat_lon_var = window_and_flatten(
        lon_var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )
    g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
    for i in 1:EKP.get_N_ens(eki)
        @test g_ens[:, i] == vcat(flat_time_var.data, flat_lon_var.data)
    end
    EKP.update_ensemble!(eki, g_ens + one_to_n_matrix(g_ens))

    # Second minibatch
    g_ens_builder = EnsembleBuilder.GEnsembleBuilder(eki)
    for i in 1:3
        @test !EnsembleBuilder.is_complete(g_ens_builder)
        EnsembleBuilder.fill_g_ens_col!(g_ens_builder, i, time_var, lon_var)
    end
    @test EnsembleBuilder.is_complete(g_ens_builder)
    flat_time_var = window_and_flatten(
        time_var,
        Dates.DateTime(2008, 12),
        Dates.DateTime(2009, 9),
    )
    flat_lon_var = window_and_flatten(
        lon_var,
        Dates.DateTime(2008, 12),
        Dates.DateTime(2009, 9),
    )
    g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
    for i in 1:EKP.get_N_ens(eki)
        @test g_ens[:, i] == vcat(flat_lon_var.data, flat_time_var.data)
    end
    EKP.update_ensemble!(eki, g_ens + one_to_n_matrix(g_ens))

    # Third minibatch
    g_ens_builder = EnsembleBuilder.GEnsembleBuilder(eki)
    for i in 1:3
        @test !EnsembleBuilder.is_complete(g_ens_builder)
        EnsembleBuilder.fill_g_ens_col!(g_ens_builder, i, time_var)
    end
    @test EnsembleBuilder.is_complete(g_ens_builder)
    flat_time_var = window_and_flatten(
        time_var,
        Dates.DateTime(2009, 12),
        Dates.DateTime(2010, 9),
    )
    g_ens = EnsembleBuilder.get_g_ensemble(g_ens_builder)
    for i in 1:EKP.get_N_ens(eki)
        @test g_ens[:, i] == flat_time_var.data
    end

    # Test warning and errors
    # OutputVar does not has a short name
    no_short_name_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(start_date = "2007-12-1") |>
        initialize
    @test_throws ErrorException EnsembleBuilder.fill_g_ens_col!(
        g_ens_builder,
        1,
        no_short_name_var,
    )

    # OutputVar has a short name, but the short name does not match with
    # anything
    attribs = Dict("start_date" => "2007-12-1", "short_name" => "no short name")
    short_name_var =
        ClimaAnalysis.remake(no_short_name_var, attributes = attribs)
    @test_logs (:warn, r"did not match") EnsembleBuilder.fill_g_ens_col!(
        g_ens_builder,
        1,
        short_name_var,
    )

    # Overriding the contents of a OutputVar
    @test_logs (:warn, r"Replacing the contents") EnsembleBuilder.fill_g_ens_col!(
        g_ens_builder,
        1,
        time_var,
    )
end

@testset "Error handling when constructing GEnsembleBuilder" begin
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(3 * i) for i in 0:4],
        )
    time_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(start_date = "2007-12-1") |>
        one_to_n_data() |>
        initialize
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()
    obs_vec = [
        ObservationRecipe.observation(
            covar_estimator,
            time_var,
            "2007-12-1",
            "2008-9-1",
        ),
    ]
    obs_series = EKP.ObservationSeries(
        Dict(
            "observations" => obs_vec,
            "names" => ["1"],
            "minibatcher" =>
                ClimaCalibrate.minibatcher_over_samples([1], 1),
        ),
    )
    prior = constrained_gaussian("pi_groups_coeff", 1.0, 0.3, 0, Inf)
    eki = EKP.EnsembleKalmanProcess(
        obs_series,
        EKP.TransformUnscented(prior, impose_prior = true),
        verbose = true,
        scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    )
    @test_throws ErrorException EnsembleBuilder.GEnsembleBuilder(eki)
end

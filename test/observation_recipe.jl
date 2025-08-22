using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import EnsembleKalmanProcesses
import OrderedCollections: OrderedDict
import Statistics: mean
import LinearAlgebra: Diagonal, I
import Statistics
import NaNStatistics: nanvar, nanmean

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions

# Since functions not defined in ext.jl are not exported, we need to access
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

@testset "Utils" begin
    @testset "Dates in var" begin
        time =
            ClimaAnalysis.Utils.date_to_time.(
                Dates.DateTime(2007, 12),
                [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:3],
            )
        var =
            TemplateVar() |>
            add_dim("time", time, units = "s") |>
            add_attribs(
                short_name = "hi",
                long_name = "hello",
                start_date = "2007-12-1",
            ) |>
            one_to_n_data(collected = false) |>
            initialize
        @test isnothing(
            ext._check_dates_in_var(
                (var,),
                ClimaAnalysis.dates(var)[1],
                ClimaAnalysis.dates(var)[2],
            ),
        )
        @test_throws ErrorException ext._check_dates_in_var(
            (var,),
            ClimaAnalysis.dates(var)[1],
            Dates.DateTime(2010, 1),
        )
        @test_throws ErrorException ext._check_dates_in_var(
            (var,),
            Dates.DateTime(2005),
            ClimaAnalysis.dates(var)[2],
        )
    end

    @testset "Var to iterable" begin
        time = [0.0, 1.0, 2.0]
        var =
            TemplateVar() |>
            add_dim("time", time, units = "s") |>
            add_attribs(
                short_name = "hi",
                long_name = "hello",
                start_date = "2007-12-1",
            ) |>
            one_to_n_data(collected = false) |>
            initialize

        vars = ext._vars_to_iterable(var)

        @test Base.isiterable(typeof(vars))
        @test eltype(vars) <: ClimaAnalysis.OutputVar
    end

    @testset "Split by season from seconds" begin
        dates = [
            Dates.DateTime(2015, 1, 13),
            Dates.DateTime(2018, 2, 13),
            Dates.DateTime(1981, 7, 6),
            Dates.DateTime(1993, 11, 19),
            Dates.DateTime(2040, 4, 1),
            Dates.DateTime(2000, 8, 18),
        ]
        expected_dates = (
            [Dates.DateTime(2040, 4, 1)],
            [Dates.DateTime(1981, 7, 6), Dates.DateTime(2000, 8, 18)],
            [Dates.DateTime(1993, 11, 19)],
            [Dates.DateTime(2015, 1, 13), Dates.DateTime(2018, 2, 13)],
        )

        seconds =
            ClimaAnalysis.Utils.date_to_time.(
                Dates.DateTime(2015, 1, 13),
                dates,
            )
        expected_seconds = [
            ClimaAnalysis.Utils.date_to_time.(
                Dates.DateTime(2015, 1, 13),
                dates,
            ) for dates in expected_dates
        ]

        seasons = ext.split_by_season_from_seconds(
            seconds,
            Dates.DateTime(2015, 1, 13),
        )

        @test seasons == expected_seconds
    end

    @testset "Find seasons" begin
        start_date = Dates.DateTime(2010, 12, 1)
        end_date = Dates.DateTime(2011, 9, 1)
        @test ext.find_seasons(start_date, end_date) ==
              ["DJF", "MAM", "JJA", "SON"]
        @test ext.find_seasons(start_date, start_date) == ["DJF"]
        @test ext.find_seasons(end_date, end_date) == ["SON"]
    end

    @testset "Check time dim" begin
        # No time dim
        var =
            TemplateVar() |>
            add_dim("lat", [0.0, 1.0], units = "degrees") |>
            add_attribs(short_name = "hi") |>
            one_to_n_data() |>
            initialize
        @test_throws ErrorException ext._check_time_dim(var)

        # Wrong units
        var =
            TemplateVar() |>
            add_dim("time", [0.0, 1.0, 2.0], units = "min") |>
            add_attribs(short_name = "hi") |>
            one_to_n_data(collected = false) |>
            initialize
        @test_throws ErrorException ext._check_time_dim(var)

        # Missing start date
        ClimaAnalysis.set_dim_units!(var, "time", "s")
        @test_throws ErrorException ext._check_time_dim(var)
    end
end

@testset "Seasonally aligned year sample dates" begin
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:13],
        )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        one_to_n_data(collected = false) |>
        initialize
    sample_date_ranges =
        ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges(var)

    @test sample_date_ranges == [
        [
            Dates.DateTime("2007-12-01T00:00:00"),
            Dates.DateTime("2008-11-01T00:00:00"),
        ],
        [
            Dates.DateTime("2008-12-01T00:00:00"),
            Dates.DateTime("2009-01-01T00:00:00"),
        ],
    ]

    # Time dimension with a single element
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12)],
        )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        one_to_n_data(collected = false) |>
        initialize
    sample_date_ranges =
        ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges(var)

    @test sample_date_ranges == [[
        Dates.DateTime("2007-12-01T00:00:00"),
        Dates.DateTime("2007-12-01T00:00:00"),
    ]]

    # Error handling
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2008, 12), Dates.DateTime(2007, 12)],
        )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        one_to_n_data(collected = false) |>
        initialize

    @test_throws ErrorException ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges(
        var,
    )
end

@testset "Change data type of OutputVar" begin
    time = [0.0, 1.0, 2.0]
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        one_to_n_data(data_type = Float32) |>
        initialize
    var64 = ObservationRecipe.change_data_type(var, Float64)
    @test eltype(var64.data) == Float64
    var32 = ObservationRecipe.change_data_type(var, Float32)
    @test eltype(var32.data) == Float32
    var32 = ObservationRecipe.change_data_type(var, Float16)
    @test eltype(var32.data) == Float16
end

@testset "Group and reduce by" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = [0.0, 1.0, 5.0]

    var =
        TemplateVar() |>
        add_dim("lat", lat, units = "degrees") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            long_name = "hi",
            start_date = "2008-1-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    dim_attribs = deepcopy(var.dim_attributes)

    # Equivalent to getting the first time slice
    reduce_by(A; dims) = A
    grp_agg_var =
        ext.group_and_reduce_by(var, "time", x -> [[first(x)]], reduce_by)
    @test grp_agg_var.data == reshape(var.data[:, :, 1], (4, 5, 1))
    @test grp_agg_var.attributes ==
          Dict("long_name" => "hi", "start_date" => "2008-1-1")
    @test grp_agg_var.dim_attributes == dim_attribs
    @test grp_agg_var.dims ==
          OrderedDict("lat" => lat, "lon" => lon, "time" => [0.0])

    # Seasonal averages across time
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2008),
            [Dates.DateTime(2008, i) for i in 1:12],
        )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            long_name = "hi",
            start_date = "2008-1-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize

    function group_by(dim)
        grouped_dates =
            ClimaAnalysis.Utils.time_to_date.(Dates.DateTime(2008), dim) |>
            ClimaAnalysis.Utils.split_by_season_across_time
        grouped_times = [
            ClimaAnalysis.Utils.date_to_time.(Dates.DateTime(2008), x) for
            x in grouped_dates
        ]
        return grouped_times
    end
    avg_season_var = ClimaAnalysis.average_season_across_time(var)
    grp_agg_var = ext.group_and_reduce_by(var, "time", group_by, mean)

    @test avg_season_var.data == grp_agg_var.data
    @test grp_agg_var.attributes ==
          Dict("long_name" => "hi", "start_date" => "2008-1-1")
    @test grp_agg_var.dim_attributes == dim_attribs
    @test grp_agg_var.dims == avg_season_var.dims

    # Error handling
    @test_throws ErrorException ext.group_and_reduce_by(
        var,
        "pfull",
        group_by,
        mean,
    )
end

@testset "Stacked samples" begin
    lat = [-90.0, -45.0, -30.0, 0.0, 30.0, 45.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:35],
        )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        one_to_n_data(collected = false) |>
        initialize
    var = ClimaAnalysis.average_season_across_time(var)

    vars = (var, -var)
    sample_date_ranges = [
        (Dates.DateTime(i, 12, 1), Dates.DateTime(i + 1, 9, 1)) for
        i in 2007:2009
    ]
    stacked_samples = ext._stacked_samples(vars, sample_date_ranges)

    @test length(stacked_samples) == length(sample_date_ranges)

    for ((start_date, end_date), stacked_sample) in
        zip(sample_date_ranges, stacked_samples)
        window_dates =
            var -> ClimaAnalysis.window(
                var,
                "time",
                left = start_date,
                right = end_date,
            )
        data_var = (var |> window_dates |> ClimaAnalysis.flatten).data
        data_minus_var = (-var |> window_dates |> ClimaAnalysis.flatten).data
        @test stacked_sample == vcat(data_var, data_minus_var)
    end
end

@testset "Covariance constructors" begin
    # Negative value for scalar
    @test_throws ErrorException ObservationRecipe.ScalarCovariance(;
        scalar = -1.0,
        use_latitude_weights = false,
        min_cosd_lat = 0.1,
    )

    # Zero value for scalar
    @test_throws ErrorException ObservationRecipe.ScalarCovariance(;
        scalar = 0.0,
        use_latitude_weights = false,
        min_cosd_lat = 0.1,
    )

    # Negative cosine weights
    @test_throws ErrorException ObservationRecipe.ScalarCovariance(;
        scalar = 1.0,
        use_latitude_weights = true,
        min_cosd_lat = -0.1,
    )

    # Negative values for model_error_scale and regularization
    @test_throws ErrorException ObservationRecipe.SeasonalDiagonalCovariance(
        model_error_scale = -2.0,
    )
    @test_throws ErrorException ObservationRecipe.SeasonalDiagonalCovariance(
        regularization = -2.0,
    )

    # Dates in the wrong order
    @test_throws ErrorException ObservationRecipe.SVDplusDCovariance([[
        Dates.DateTime(2011),
        Dates.DateTime(2010),
    ]])

    # Negative values for regularization and model_error_scale
    @test_throws ErrorException ObservationRecipe.SVDplusDCovariance(
        [[Dates.DateTime(2010), Dates.DateTime(2011)]],
        regularization = -2.0,
    )
    @test_throws ErrorException ObservationRecipe.SVDplusDCovariance(
        [[Dates.DateTime(2010), Dates.DateTime(2011)]],
        model_error_scale = -2.0,
    )
end

@testset "Lat weights" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    var =
        TemplateVar() |>
        add_dim("lat", lat, units = "deg") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        one_to_n_data() |>
        initialize

    lat_weights = ext._lat_weights_var(var)
    @test lat_weights.data == 1.0 ./ max.(cosd.(lat), 0.1)

    lat_weights = ext._lat_weights_var(var, min_cosd_lat = 3.0)
    @test lat_weights.data == 1.0 ./ max.(cosd.(lat), 3.0)

    # 3D case
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = [0.0, 1.0, 2.0]
    var =
        TemplateVar() |>
        add_dim("lat", lat, units = "deg") |>
        add_dim("lon", lon, units = "deg") |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        one_to_n_data() |>
        initialize

    lat_weights = ext._lat_weights_var(var)
    @test lat_weights.data ==
          repeat(1.0 ./ max.(cosd.(lat), 0.1), outer = (1, 5, 3))

    lat_weights = ext._lat_weights_var(var, min_cosd_lat = 3.0)
    @test lat_weights.data ==
          repeat(1.0 ./ max.(cosd.(lat), 3.0), outer = (1, 5, 3))

    # 3D NaN case
    data = collect(var.data)
    data[1, 1, 1] = NaN
    data[3, 2, 1] = NaN
    var = ClimaAnalysis.remake(var, data = data)
    lat_weights = ext._lat_weights_var(var)
    correct_lat_weights =
        repeat(1.0 ./ max.(cosd.(lat), 0.1), outer = (1, 5, 3))
    correct_lat_weights[1, 1, 1] = NaN
    correct_lat_weights[3, 2, 1] = NaN
    @test isequal(lat_weights.data, correct_lat_weights)

    # Error handling
    # No latitude dimension
    var = make_template_var("lon", "time") |> initialize
    @test_throws ErrorException ext._lat_weights_var(var)

    # Latitude dimension is not degrees
    var = make_template_var("lat", "time") |> initialize
    ClimaAnalysis.set_dim_units!(var, "lat", "rad")
    @test_throws ErrorException ext._lat_weights_var(var)
end

@testset "ScalarCovariance" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:35],
        )
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

    sample_start_date = Dates.DateTime(2007, 12)
    sample_end_date = Dates.DateTime(2008, 9)
    window_var = ClimaAnalysis.window(
        var,
        "time",
        left = sample_start_date,
        right = sample_end_date,
    )
    data_length = length(window_var.data)

    # Test default constructor
    covar_estimator = ObservationRecipe.ScalarCovariance()
    scalar_covar = ObservationRecipe.covariance(
        covar_estimator,
        var,
        sample_start_date,
        sample_end_date,
    )
    @test Diagonal(ones(data_length)) == scalar_covar

    covar_estimator = ObservationRecipe.ScalarCovariance(scalar = 10.0)
    scalar_covar = ObservationRecipe.covariance(
        covar_estimator,
        var,
        sample_start_date,
        sample_end_date,
    )
    @test Diagonal(10.0 * ones(data_length)) == scalar_covar

    # Latitude weights
    covar_estimator = ObservationRecipe.ScalarCovariance(
        scalar = 10.0,
        use_latitude_weights = true,
        min_cosd_lat = 0.2,
    )
    scalar_covar = ObservationRecipe.covariance(
        covar_estimator,
        var,
        sample_start_date,
        sample_end_date,
    )
    @test scalar_covar ==
          10.0 .* Diagonal(
        ClimaAnalysis.flatten(
            ext._lat_weights_var(window_var, min_cosd_lat = 0.2),
        ).data,
    )

    # With NaNs
    data = copy(var.data)
    data[1, 1, 1] = NaN
    data[3, 2, 4] = NaN
    data[21, 3, 2]
    nan_var = ClimaAnalysis.remake(var, data = data)
    window_nan_var = ClimaAnalysis.window(
        nan_var,
        "time",
        left = sample_start_date,
        right = sample_end_date,
    )

    covar_estimator = ObservationRecipe.ScalarCovariance(scalar = 10.0)
    scalar_covar = ObservationRecipe.covariance(
        covar_estimator,
        nan_var,
        sample_start_date,
        sample_end_date,
    )
    @test scalar_covar == 10.0 .* Diagonal(ones(data_length - 2))

    # Latitude weights with NaNs
    covar_estimator = ObservationRecipe.ScalarCovariance(
        scalar = 10.0,
        use_latitude_weights = true,
        min_cosd_lat = 0.2,
    )
    scalar_covar = ObservationRecipe.covariance(
        covar_estimator,
        nan_var,
        sample_start_date,
        sample_end_date,
    )
    @test scalar_covar ==
          10.0 .* Diagonal(
        ClimaAnalysis.flatten(
            ext._lat_weights_var(window_nan_var, min_cosd_lat = 0.2),
        ).data,
    )
end

@testset "SVDplusDCovariance" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:35],
        )
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
    var = ClimaAnalysis.average_season_across_time(var)

    sample_date_ranges = [
        (Dates.DateTime(i, 12, 1), Dates.DateTime(i + 1, 9, 1)) for
        i in 2007:2009
    ]

    # There is no need to test the SVD part of the matrix as that is constructed
    # by EKP

    # Test only regularization
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        sample_date_ranges;
        model_error_scale = 0.0,
        regularization = 1e-3,
    )
    svd_plus_d_covar = ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )
    sample_size = length(lon) * length(lat) * 4
    @test svd_plus_d_covar.diag_cov == Diagonal([0.001 for _ in 1:sample_size])

    # Test only model error scale
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        sample_date_ranges;
        model_error_scale = 1.0,
        regularization = 0.0,
    )
    svd_plus_d_covar = ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )
    @test svd_plus_d_covar.diag_cov == Diagonal(
        vec(
            mean(
                hcat(ext._stacked_samples((var,), sample_date_ranges)...),
                dims = 2,
            ) .^ 2,
        ),
    )

    # Test regularization + model_error_scale
    model_error_scale = 2.0
    regularization = 1e-3
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        sample_date_ranges;
        model_error_scale = model_error_scale,
        regularization = regularization,
    )
    svd_plus_d_covar = ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )
    @test svd_plus_d_covar.diag_cov ==
          Diagonal([regularization for _ in 1:sample_size]) + Diagonal(
        vec(
            model_error_scale .* mean(
                hcat(ext._stacked_samples((var,), sample_date_ranges)...),
                dims = 2,
            ),
        ) .^ 2,
    )

    # Test float type
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        sample_date_ranges,
        model_error_scale = 2.0f0,
        regularization = 1.0f0,
    )
    var32 = ObservationRecipe.change_data_type(var, Float32)
    svd_plus_d_covar = ObservationRecipe.covariance(
        covar_estimator,
        var32,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )

    # See this issue for the broken tests:
    # https://github.com/CliMA/EnsembleKalmanProcesses.jl/issues/504
    @test_broken eltype(svd_plus_d_covar.svd_cov.S) == Float32
    @test_broken eltype(svd_plus_d_covar.svd_cov.U) == Float32
    @test_broken eltype(svd_plus_d_covar.svd_cov.V) == Float32
    @test_broken eltype(svd_plus_d_covar.svd_cov.Vt) == Float32
    @test eltype(svd_plus_d_covar.diag_cov.diag) == Float32

    # Error handling
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:2],
        )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        initialize
    covar_estimator = ObservationRecipe.SVDplusDCovariance([(
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 2),
    )])

    # Check start and end dates provided are valid
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 10),
    )

    # Check start date is earlier than end date
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2008, 2),
        Dates.DateTime(2007, 12),
    )

    # Check dates in sample_date_ranges are valid
    covar_estimator = ObservationRecipe.SVDplusDCovariance([(
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 3),
    )])
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 3),
    )
end

@testset "Seasonal diagonal covariance" begin
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(3 * i) for i in 0:11],
        )
    data = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, NaN]
    DJF = [data[i] for i in 1:4:length(data)]
    MAM = [data[i] for i in 2:4:length(data)]
    JJA = [data[i] for i in 3:4:length(data)]
    SON = [data[i] for i in 4:4:length(data)]
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        add_data(data = data) |>
        initialize

    # No regularization and model scale, but ignore NaN
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()

    seasonal_covariance = ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )

    @test seasonal_covariance ==
          Diagonal([nanvar(season) for season in (DJF, MAM, JJA, SON)])

    # No model scale, but include regularization and ignore NaN
    regularization = 2.0
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance(
        regularization = regularization,
    )

    seasonal_covariance = ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )

    @test seasonal_covariance ==
          Diagonal([nanvar(season) for season in (DJF, MAM, JJA, SON)]) + 2 * I

    # No regularization, but include model scale and ignore NaN
    model_error_scale = 3.0
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance(
        model_error_scale = model_error_scale,
    )

    seasonal_covariance = ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )

    @test seasonal_covariance ==
          Diagonal([nanvar(season) for season in (DJF, MAM, JJA, SON)]) + Diagonal(
        (
            model_error_scale .*
            [nanmean(season) for season in (DJF, MAM, JJA, SON)]
        ) .^ 2,
    )

    # Enable lat weights, but use default settings for everything else
    lats = [-90.0, -85.0, -20.0, 0.0, 20.0, 85.0, 90.0]
    time1 =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Year(i) for i in 0:1],
        )
    data = ones(Float64, 2, 7)
    data[2, :] .*= 2
    lat_var =
        TemplateVar() |>
        add_dim("time", time1, units = "s") |>
        add_dim("latitude", lats, units = "deg") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        add_data(data = data) |>
        initialize
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance(
        use_latitude_weights = true,
    )
    seasonal_covariance = ObservationRecipe.covariance(
        covar_estimator,
        lat_var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2007, 12),
    )
    sliced_var = ClimaAnalysis.slice(lat_var, time = 0.0)
    @test seasonal_covariance ==
          (1 / 2) *
          Diagonal(ClimaAnalysis.flatten(ext._lat_weights_var(sliced_var)).data)

    # Lat weight with NaN in data
    data = lat_var.data
    data[:, 3] .= NaN
    data[:, 4] .= NaN
    lat_var = ClimaAnalysis.remake(lat_var, data = data)
    seasonal_covariance = ObservationRecipe.covariance(
        covar_estimator,
        lat_var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2007, 12),
    )
    sliced_var = ClimaAnalysis.slice(lat_var, time = 0.0)
    @test seasonal_covariance ==
          (1 / 2) *
          Diagonal(ClimaAnalysis.flatten(ext._lat_weights_var(sliced_var)).data)

    # Ignore NaN set to false
    covar_estimator =
        ObservationRecipe.SeasonalDiagonalCovariance(ignore_nan = false)

    seasonal_covariance = ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )
    # Use approx because of floating point errors
    @test seasonal_covariance ≈ Diagonal([
        Statistics.var(DJF),
        Statistics.var(MAM),
        Statistics.var(JJA),
    ])

    # Check float type
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()

    var32 = ObservationRecipe.change_data_type(var, Float32)
    seasonal_covariance = ObservationRecipe.covariance(
        covar_estimator,
        var32,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )

    @test eltype(seasonal_covariance) == Float32

    # Error handling
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:2],
        )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
        ) |>
        initialize
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()

    # Check multiple time slices in a single season
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 2),
    )

    # Check start and end dates exist
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 10),
    )

    # Check start date is earlier than end date
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        var,
        Dates.DateTime(2008, 2),
        Dates.DateTime(2007, 12),
    )
end

@testset "Observation" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:35],
        )
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
    var = ClimaAnalysis.average_season_across_time(var)
    neg_var = -2.0 * var
    neg_var.attributes["hi"] = "hello"
    neg_var.dim_attributes["lon"]["a dim"] = "attribute"

    sample_date_ranges = [
        (Dates.DateTime(i, 12, 1), Dates.DateTime(i + 1, 9, 1)) for
        i in 2007:2009
    ]

    # It doesn't matter which covariance estimator is being used
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        sample_date_ranges;
        model_error_scale = 1e-2,
        regularization = 1e-3,
    )

    start_date = Dates.DateTime(2007, 12)
    end_date = Dates.DateTime(2008, 9)
    obs = ObservationRecipe.observation(
        covar_estimator,
        (var, neg_var),
        start_date,
        end_date,
    )

    # We already test the covariance matrix, so we test if the flattened
    # sample is formed correctly and if the metadata is correct
    # Test if flattened sample is correct
    window_dates =
        var -> ClimaAnalysis.window(
            var,
            "time",
            left = start_date,
            right = end_date,
        )
    data1 = ClimaAnalysis.flatten(window_dates(var)).data
    data2 = ClimaAnalysis.flatten(window_dates(neg_var)).data

    flattened_data = vcat(data1, data2)
    @test obs.samples[1] == flattened_data

    # Also check the observation name
    @test obs.names == ["hi;-2.0 * hi"]

    if pkgversion(EnsembleKalmanProcesses) > v"2.4.2"
        # Test metadata in observation is there in the EKP object
        @test obs.metadata isa Vector{T} where {T <: ClimaAnalysis.Var.Metadata}
        @test length(obs.metadata) == 2

        # Test if metadata is correct by unflattening back to an OutputVar
        unflattened_var =
            ClimaAnalysis.unflatten(obs.metadata[1], obs.samples[1][1:80])
        neg_unflattened_var =
            ClimaAnalysis.unflatten(obs.metadata[2], obs.samples[1][81:end])

        windowed_var = window_dates(var)
        neg_windowed_var = window_dates(neg_var)

        # Test if the two OutputVars are equivalent
        @test unflattened_var.data == windowed_var.data
        @test unflattened_var.attributes == windowed_var.attributes
        @test unflattened_var.dim_attributes == windowed_var.dim_attributes
        @test unflattened_var.dims == windowed_var.dims

        @test neg_unflattened_var.data == neg_windowed_var.data
        @test neg_unflattened_var.attributes == neg_windowed_var.attributes
        @test neg_unflattened_var.dim_attributes ==
              neg_windowed_var.dim_attributes
        @test neg_unflattened_var.dims == neg_windowed_var.dims
    end

    # Error handling
    covar_estimator = ObservationRecipe.SVDplusDCovariance(sample_date_ranges)

    # Check start date is before end date
    @test_throws ErrorException ObservationRecipe.observation(
        covar_estimator,
        (var, neg_var),
        Dates.DateTime(2008, 9),
        Dates.DateTime(2007, 12),
    )

    # Check start date and end date exist in vars
    @test_throws ErrorException ObservationRecipe.observation(
        covar_estimator,
        (var, neg_var),
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 8),
    )

end

@testset "Short names of observation" begin
    if pkgversion(EnsembleKalmanProcesses) > v"2.4.2"
        time =
            ClimaAnalysis.Utils.date_to_time.(
                Dates.DateTime(2007, 12),
                [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:2],
            )
        var1 =
            TemplateVar() |>
            add_dim("time", time, units = "s") |>
            add_attribs(short_name = "Hello", start_date = "2007-12-1") |>
            initialize

        var2 =
            TemplateVar() |>
            add_dim("time", time, units = "s") |>
            add_attribs(short_name = "world!", start_date = "2007-12-1") |>
            initialize

        var3 =
            TemplateVar() |>
            add_dim("time", time, units = "s") |>
            add_attribs(no_short_name = "no", start_date = "2007-12-1") |>
            initialize

        covar_estimator = ObservationRecipe.ScalarCovariance()
        obs = ObservationRecipe.observation(
            covar_estimator,
            (var1, var2, var3),
            Dates.DateTime(2007, 12, 1),
            Dates.DateTime(2008, 1, 1),
        )
        @test isequal(
            ObservationRecipe.short_names(obs),
            ["Hello", "world!", nothing],
        )
    end
end

@testset "Get information about metadata from nth iteration" begin
    pkgversion(EnsembleKalmanProcesses) > v"2.4.3" || return
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(3 * i) for i in 0:47],
        )
    time_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "time",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data() |>
        initialize

    lon = [-45.0, 0.0, 45.0]
    lon_var =
        TemplateVar() |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "lon",
            start_date = "2007-12-1",
            super = "fun",
        ) |>
        one_to_n_data() |>
        initialize

    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()
    obs1 = ObservationRecipe.observation(
        covar_estimator,
        (time_var, lon_var),
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )
    obs2 = ObservationRecipe.observation(
        covar_estimator,
        time_var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )
    obs3 = ObservationRecipe.observation(
        covar_estimator,
        lon_var,
        Dates.DateTime(2007, 12),
        Dates.DateTime(2008, 9),
    )

    # Test with fixed minibatcher with no randomness
    obs_series = EKP.ObservationSeries(
        Dict(
            "observations" => [obs1, obs2, obs3, obs1],
            "names" => ["1", "2", "3", "4"],
            "minibatcher" =>
                ClimaCalibrate.minibatcher_over_samples([1, 2, 3, 4], 2),
        ),
    )

    metadata1 = ObservationRecipe.get_metadata_for_nth_iteration(obs_series, 1)
    metadata2 = ObservationRecipe.get_metadata_for_nth_iteration(obs_series, 2)
    metadata3 = ObservationRecipe.get_metadata_for_nth_iteration(obs_series, 3)
    metadata_indices1 =
        ext._get_minibatch_indices_for_nth_iteration(obs_series, 1)
    metadata_indices2 =
        ext._get_minibatch_indices_for_nth_iteration(obs_series, 2)
    metadata_indices3 =
        ext._get_minibatch_indices_for_nth_iteration(obs_series, 3)

    @test length(metadata1) == 3
    @test metadata1[1].attributes["short_name"] == "time"
    @test metadata1[2].attributes["short_name"] == "lon"
    @test metadata1[3].attributes["short_name"] == "time"

    @test length(metadata_indices1) == 3
    @test metadata_indices1[1] == 1:4
    @test metadata_indices1[2] == 5:16
    @test metadata_indices1[3] == 17:20

    @test length(metadata2) == 3
    @test metadata2[1].attributes["short_name"] == "lon"
    @test metadata2[2].attributes["short_name"] == "time"
    @test metadata2[3].attributes["short_name"] == "lon"

    @test length(metadata_indices1) == 3
    @test metadata_indices2[1] == 1:12
    @test metadata_indices2[2] == 13:16
    @test metadata_indices2[3] == 17:28

    @test length(metadata3) == 3
    @test metadata3[1].attributes["short_name"] == "time"
    @test metadata3[2].attributes["short_name"] == "lon"
    @test metadata3[3].attributes["short_name"] == "time"

    @test length(metadata_indices1) == 3
    @test metadata_indices1[1] == 1:4
    @test metadata_indices1[2] == 5:16
    @test metadata_indices1[3] == 17:20
end

@testset "Reconstruct mean g ens final" begin
    if pkgversion(EnsembleKalmanProcesses) > v"2.4.3"
        # Test with a two OutputVars and two iterations with a single
        # observation (no observation series)
        time =
            ClimaAnalysis.Utils.date_to_time.(
                Dates.DateTime(2007, 12),
                [Dates.DateTime(2007, 12) + Dates.Month(3 * i) for i in 0:47],
            )
        time_var =
            TemplateVar() |>
            add_dim("time", time, units = "s") |>
            add_attribs(
                short_name = "hi",
                long_name = "hello",
                start_date = "2007-12-1",
                blah = "blah2",
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
            ) |>
            one_to_n_data() |>
            initialize

        covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()
        obs = ObservationRecipe.observation(
            covar_estimator,
            (time_var, lon_var),
            Dates.DateTime(2007, 12),
            Dates.DateTime(2008, 9),
        )

        prior = constrained_gaussian("pi_groups_coeff", 1.0, 0.3, 0, Inf)

        eki = EKP.EnsembleKalmanProcess(
            obs,
            EKP.TransformUnscented(prior, impose_prior = true),
            verbose = true,
            scheduler = EKP.DataMisfitController(on_terminate = "continue"),
        )
        G_ens = reshape(collect(1.0:48.0), 16, 3)
        EKP.update_ensemble!(eki, G_ens)
        vars = ObservationRecipe.reconstruct_g_mean_final(eki)

        # Test OutputVar is constructed correctly
        @test length(vars) == 2
        for (i, var) in enumerate((time_var, lon_var))
            @test vars[i].attributes == var.attributes
            @test vars[i].dim_attributes == var.dim_attributes
            @test vars[i].dims["time"] == first(var.dims["time"], 4)
        end
        @test vars[1].data == vec(mean(G_ens[1:4, :], dims = 2))
        @test vars[2].data == reshape(mean(G_ens[5:end, :], dims = 2), 3, 4)

        # Another iteration
        G_ens = reshape(collect(100.0:147.0), 16, 3)
        EKP.update_ensemble!(eki, G_ens)
        vars = ObservationRecipe.reconstruct_g_mean_final(eki)

        # Test OutputVar is reconstructed correctly
        @test length(vars) == 2
        for (i, var) in enumerate((time_var, lon_var))
            @test vars[i].attributes == var.attributes
            @test vars[i].dim_attributes == var.dim_attributes
            @test vars[i].dims["time"] == first(var.dims["time"], 4)
        end

        @test vars[1].data == vec(mean(G_ens[1:4, :], dims = 2))
        @test vars[2].data == reshape(mean(G_ens[5:end, :], dims = 2), 3, 4)

        # Test with multiple OutputVars with two iterations with a minibatch of
        # size 1 and an observation series
        obs1 = ObservationRecipe.observation(
            covar_estimator,
            (time_var, lon_var),
            Dates.DateTime(2007, 12),
            Dates.DateTime(2008, 9),
        )
        obs2 = ObservationRecipe.observation(
            covar_estimator,
            (time_var, lon_var),
            Dates.DateTime(2008, 12),
            Dates.DateTime(2009, 9),
        )

        obs_series = EKP.ObservationSeries(
            Dict(
                "observations" => [obs1, obs2],
                "names" => ["1", "2"],
                "minibatcher" =>
                    ClimaCalibrate.minibatcher_over_samples([1, 2], 1),
            ),
        )

        eki = EKP.EnsembleKalmanProcess(
            obs_series,
            EKP.TransformUnscented(prior, impose_prior = true),
            verbose = true,
            scheduler = EKP.DataMisfitController(on_terminate = "continue"),
        )

        G_ens = reshape(collect(100.0:147.0), 16, 3)
        EKP.update_ensemble!(eki, G_ens)
        vars = ObservationRecipe.reconstruct_g_mean_final(eki)

        # Test OutputVar is constructed correctly
        @test length(vars) == 2
        for (i, var) in enumerate((time_var, lon_var))
            @test vars[i].attributes == var.attributes
            @test vars[i].dim_attributes == var.dim_attributes
            @test vars[i].dims["time"] == first(var.dims["time"], 4)
        end
        @test vars[1].data == vec(mean(G_ens[1:4, :], dims = 2))
        @test vars[2].data == reshape(mean(G_ens[5:end, :], dims = 2), 3, 4)

        G_ens = reshape(collect(200.0:247.0), 16, 3)
        EKP.update_ensemble!(eki, G_ens)
        vars = ObservationRecipe.reconstruct_g_mean_final(eki)

        # Test OutputVar is constructed correctly
        @test length(vars) == 2
        for (i, var) in enumerate((time_var, lon_var))
            @test vars[i].attributes == var.attributes
            @test vars[i].dim_attributes == var.dim_attributes
            @test vars[i].dims["time"] == var.dims["time"][5:8]
        end

        @test vars[1].data == vec(mean(G_ens[1:4, :], dims = 2))
        @test vars[2].data == reshape(mean(G_ens[5:end, :], dims = 2), 3, 4)

        # Test with multiple OutputVars with two iterations with a minibatch of
        # size 2 and an observation series
        obs3 = ObservationRecipe.observation(
            covar_estimator,
            (time_var, lon_var),
            Dates.DateTime(2009, 12),
            Dates.DateTime(2010, 9),
        )
        obs4 = ObservationRecipe.observation(
            covar_estimator,
            (time_var, lon_var),
            Dates.DateTime(2010, 12),
            Dates.DateTime(2011, 9),
        )
        obs_series = EKP.ObservationSeries(
            Dict(
                "observations" => [obs1, obs2, obs3, obs4],
                "names" => ["1", "2", "3", "4"],
                "minibatcher" => ClimaCalibrate.minibatcher_over_samples(
                    [1, 2, 3, 4],
                    2,
                ),
            ),
        )

        eki = EKP.EnsembleKalmanProcess(
            obs_series,
            EKP.TransformUnscented(prior, impose_prior = true),
            verbose = true,
            scheduler = EKP.DataMisfitController(on_terminate = "continue"),
        )

        G_ens = reshape(collect(1.0:96.0), 32, 3)
        EKP.update_ensemble!(eki, G_ens)
        vars = ObservationRecipe.reconstruct_g_mean_final(eki)

        # Test OutputVar is constructed correctly
        @test length(vars) == 4
        for (i, var) in enumerate((time_var, lon_var, time_var, lon_var))
            @test vars[i].attributes == var.attributes
            @test vars[i].dim_attributes == var.dim_attributes
            if i in (1, 2)
                @test vars[i].dims["time"] == var.dims["time"][1:4]
            elseif i in (3, 4)
                @test vars[i].dims["time"] == var.dims["time"][5:8]
            else
                error("You are not supposed to be here!")
            end
        end

        @test vars[1].data == vec(mean(G_ens[1:4, :], dims = 2))
        @test vars[2].data == reshape(mean(G_ens[5:16, :], dims = 2), 3, 4)
        @test vars[3].data == vec(mean(G_ens[17:20, :], dims = 2))
        @test vars[4].data == reshape(mean(G_ens[21:end, :], dims = 2), 3, 4)

        # Another iteration
        G_ens = reshape(collect(100.0:195.0), 32, 3)
        EKP.update_ensemble!(eki, G_ens)
        vars = ObservationRecipe.reconstruct_g_mean_final(eki)

        # Test OutputVar is constructed correctly
        @test length(vars) == 4
        for (i, var) in enumerate((time_var, lon_var, time_var, lon_var))
            @test vars[i].attributes == var.attributes
            @test vars[i].dim_attributes == var.dim_attributes
            if i in (1, 2)
                @test vars[i].dims["time"] == var.dims["time"][9:12]
            elseif i in (3, 4)
                @test vars[i].dims["time"] == var.dims["time"][13:16]
            else
                error("You are not supposed to be here!")
            end
        end

        @test vars[1].data == vec(mean(G_ens[1:4, :], dims = 2))
        @test vars[2].data == reshape(mean(G_ens[5:16, :], dims = 2), 3, 4)
        @test vars[3].data == vec(mean(G_ens[17:20, :], dims = 2))
        @test vars[4].data == reshape(mean(G_ens[21:end, :], dims = 2), 3, 4)
    end
end

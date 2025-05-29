using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.Pipeline as Pipeline
import OrderedCollections: OrderedDict
import Statistics: mean

# Since functions not defined in ext.jl are not exported, we need to access
# them like this
ext = Base.get_extension(ClimaCalibrate, :ClimaAnalysisExt)

# TODO: I might need NaNStatistics as well...

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

@testset "group and reduce by" begin
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
    time = ClimaAnalysis.Utils.date_to_time.(
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

@testset "Seasonal sample" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = ClimaAnalysis.Utils.date_to_time.(
        Dates.DateTime(2007, 12),
        [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:11],
    )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            long_name = "hi",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize

    sample_config = Pipeline.SeasonalSample(
        ignore_nan_in_average = true,
        ignore_nan_in_sample = true,
    )

    data, metadata = Pipeline.sample(sample_config, var, metadata = true)

    # Unflatten data to make it easier to test
    unflatten_var = ClimaAnalysis.unflatten(metadata, data)

    average_season_var = ClimaAnalysis.average_season_across_time(var)

    @test unflatten_var.data == average_season_var.data
    @test unflatten_var.attributes == average_season_var.attributes
    @test unflatten_var.dim_attributes == average_season_var.dim_attributes
    @test unflatten_var.dims == average_season_var.dims

    # Test if taking seasonal sample is idempotent. If the OutputVar are already
    # seasonal averages, then the data should be flattened
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = ClimaAnalysis.Utils.date_to_time.(
        Dates.DateTime(2007, 12),
        [Dates.DateTime(2007, 12) + Dates.Month(3 * i) for i in 0:3],
    )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            long_name = "hi",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize

    sample_config = Pipeline.SeasonalSample(
        ignore_nan_in_average = true,
        ignore_nan_in_sample = true,
    )
    data, metadata = Pipeline.sample(sample_config, var, metadata = true)
    unflatten_var = ClimaAnalysis.unflatten(metadata, data)

    @test unflatten_var.data == var.data
    @test unflatten_var.attributes["long_name"] ==
          "hi season averaged over time (0.0 to 2.376e7s)"
    @test unflatten_var.attributes["start_date"] == "2007-12-1"
    @test unflatten_var.attributes["blah"] == "blah2"
    @test unflatten_var.dim_attributes == var.dim_attributes
    @test unflatten_var.dims == var.dims

    # TODO: Test with start and end dates
    # TODO: Test other configurations (with and without nans)
end

@testset "Covariance" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = ClimaAnalysis.Utils.date_to_time.(
        Dates.DateTime(2007, 12),
        [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:36],
    )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            long_name = "hi",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize

    covar_config = Pipeline.NoiseCovariance()
    sample_config = Pipeline.SeasonalSample(
        ignore_nan_in_average = true,
        ignore_nan_in_sample = true,
    )
    covar = Pipeline.covariance(covar_config, sample_config, var)
end

@testset "Observation" begin
    # TODO: Make an OutputVar of 3 years
    # TODO: Make observation for a single year
    # TODO: Make covariance for all three years
    # Test that it match the result from sample and covariance
    # Test that the ordering is correct (should be because of flattening)

    # Test reconstruct and deconstruct
end

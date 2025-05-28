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

using ClimaCalibrate, OrderedCollections, Test, Dates
import Random: MersenneTwister

import Dates, ClimaAnalysis

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
    one_to_n_data

@testset "ClimaAnalysisExt" begin
    dates = [(Dates.DateTime(2010, 12, 1) + Dates.Month(i) for i in 1:24)...]
    times = ClimaAnalysis.Utils.date_to_time.(Dates.DateTime(2010, 12, 1), dates)
    lon = collect(range(-180.0, 180.0, 10))
    lat = collect(range(-90.0, 90.0, 5))

    var = TemplateVar() |>
    add_attribs(start_date = "2010-12-1") |>
    add_dim("time", times, units = "s") |>
    add_dim("lon", lon) |>
    add_dim("lat", lat) |>
    ClimaAnalysis.Template.initialize

    ClimaCalibrate.year_of_seasonal_observations(var, 2011)
end

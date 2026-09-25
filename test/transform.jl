using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.SampleBuilder
import ClimaCalibrate.SampleBuilder:
    AbstractSampleCollection,
    AbstractTransform,
    LatitudeWeighting,
    apply_transform,
    base
import EnsembleKalmanProcesses as EKP

import ClimaAnalysis.Template:
    TemplateVar, add_dim, add_attribs, one_to_n_data, initialize

# Since functions defined in ext are not exported, we need to access them like
# this
ext = Base.get_extension(ClimaCalibrate, :ClimaCalibrateClimaAnalysisExt)

@testset "Latitude weights to matrix of samples" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = ClimaAnalysis.Utils.date_to_time.(
        Dates.DateTime(2007, 12),
        [Dates.DateTime(i, 12, 1) for i in 2007:2009],
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

    sample_date_ranges = [
        (Dates.DateTime(i, 12, 1), Dates.DateTime(i, 12, 1)) for i in 2007:2009
    ]

    sc = SampleBuilder.build_samples_by_times(
        [var],
        sample_date_ranges;
        FT = Float64,
    )
    weighted_sc = apply_transform(LatitudeWeighting(min_cosd_lat = 0.15), sc)

    time_slice = ClimaAnalysis.slice(var, time = Dates.DateTime(2007, 12, 1))
    lat_weights_per_column = sqrt.(
        ClimaAnalysis.flatten(
            ext._lat_weights_var(time_slice, min_cosd_lat = 0.15),
        ).data,
    )
    @test isequal(
        SampleBuilder.get_samples(weighted_sc),
        SampleBuilder.get_samples(sc) .* lat_weights_per_column,
    )
end

using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.SampleBuilder
import EnsembleKalmanProcesses
import OrderedCollections: OrderedDict
import Statistics: mean
import LinearAlgebra: Diagonal, I
import Statistics
import NaNStatistics: nanvar, nanmean

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions

# Since functions defined in ext are not exported, we need to access them like
# this
ext = Base.get_extension(ClimaCalibrate, :ClimaCalibrateClimaAnalysisExt)

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

    # Negative values for regularization and model_error_scale
    @test_throws ErrorException ObservationRecipe.SVDplusDCovariance(
        regularization = -2.0,
    )
    @test_throws ErrorException ObservationRecipe.SVDplusDCovariance(
        model_error_scale = -2.0,
    )
    @test_throws ErrorException ObservationRecipe.SVDplusDCovariance(
        use_latitude_weights = true,
        min_cosd_lat = -0.1,
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

    osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            var,
            [(sample_start_date, sample_end_date)];
            FT = Float64,
        ),
        1,
    )

    # Test default constructor
    covar_estimator = ObservationRecipe.ScalarCovariance()
    scalar_covar = ObservationRecipe.covariance(covar_estimator, osc)
    @test Diagonal(ones(data_length)) == scalar_covar

    covar_estimator = ObservationRecipe.ScalarCovariance(scalar = 10.0)
    scalar_covar = ObservationRecipe.covariance(covar_estimator, osc)
    @test Diagonal(10.0 * ones(data_length)) == scalar_covar

    # Latitude weights
    covar_estimator = ObservationRecipe.ScalarCovariance(
        scalar = 10.0,
        use_latitude_weights = true,
        min_cosd_lat = 0.2,
    )
    scalar_covar = ObservationRecipe.covariance(covar_estimator, osc)
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
    nan_var = ClimaAnalysis.remake(var, data = data)
    window_nan_var = ClimaAnalysis.window(
        nan_var,
        "time",
        left = sample_start_date,
        right = sample_end_date,
    )

    nan_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            nan_var,
            [(sample_start_date, sample_end_date)];
            FT = Float64,
        ),
        1,
    )

    covar_estimator = ObservationRecipe.ScalarCovariance(scalar = 10.0)
    scalar_covar = ObservationRecipe.covariance(covar_estimator, nan_osc)
    @test scalar_covar == 10.0 .* Diagonal(ones(data_length - 2))

    # Latitude weights with NaNs
    covar_estimator = ObservationRecipe.ScalarCovariance(
        scalar = 10.0,
        use_latitude_weights = true,
        min_cosd_lat = 0.2,
    )
    scalar_covar = ObservationRecipe.covariance(covar_estimator, nan_osc)
    @test scalar_covar ==
          10.0 .* Diagonal(
        ClimaAnalysis.flatten(
            ext._lat_weights_var(window_nan_var, min_cosd_lat = 0.2),
        ).data,
    )
end

@testset "ScalarCovariance for OutputVars with no time dimension" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    var =
        TemplateVar() |>
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

    data_length = length(var.data)

    osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples(var; FT = Float64),
        1,
    )

    # Only scalar
    covar_estimator = ObservationRecipe.ScalarCovariance(scalar = 100.0)
    scalar_covar = ObservationRecipe.covariance(covar_estimator, osc)
    @test Diagonal(ones(data_length) .* 100) == scalar_covar

    # Scalar and latitude weights
    covar_estimator = ObservationRecipe.ScalarCovariance(
        scalar = 10.0,
        use_latitude_weights = true,
        min_cosd_lat = 0.2,
    )
    scalar_covar = ObservationRecipe.covariance(covar_estimator, osc)
    @test scalar_covar ==
          10.0 .* Diagonal(
        ClimaAnalysis.flatten(
            ext._lat_weights_var(var, min_cosd_lat = 0.2),
        ).data,
    )
end

@testset "Latitude weights to matrix of samples" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
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
    stacked_sample_matrix_no_lat_weights = copy(sc.samples)
    stacked_sample_matrix_with_lat_weights = copy(sc.samples)

    ext._apply_lat_weights_to_samples!(
        stacked_sample_matrix_with_lat_weights,
        sc.metadata[:, 1],
        min_cosd_lat = 0.15,
    )
    time_slice = ClimaAnalysis.slice(var, time = Dates.DateTime(2007, 12, 1))
    lat_weights_per_column =
        sqrt.(
            ClimaAnalysis.flatten(
                ext._lat_weights_var(time_slice, min_cosd_lat = 0.15),
            ).data,
        )
    lat_weights_per_column =
        reshape(lat_weights_per_column, length(lat_weights_per_column), 1)
    @test isequal(
        stacked_sample_matrix_with_lat_weights,
        stacked_sample_matrix_no_lat_weights .* lat_weights_per_column,
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

    osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [var],
            sample_date_ranges;
            FT = Float64,
        ),
        1,
    )

    # There is no need to test the SVD part of the matrix as that is constructed
    # by EKP

    # Test only regularization
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        model_error_scale = 0.0,
        regularization = 1e-3,
    )
    svd_plus_d_covar = ObservationRecipe.covariance(covar_estimator, osc)
    sample_size = length(lon) * length(lat) * 4
    @test svd_plus_d_covar.diag_cov == Diagonal([0.001 for _ in 1:sample_size])

    # Test only model error scale
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        model_error_scale = 1.0,
        regularization = 0.0,
    )
    svd_plus_d_covar = ObservationRecipe.covariance(covar_estimator, osc)
    @test svd_plus_d_covar.diag_cov ==
          Diagonal(vec(mean(SampleBuilder.get_samples(osc), dims = 2) .^ 2))

    # Test regularization + model_error_scale
    model_error_scale = 2.0
    regularization = 1e-3
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        model_error_scale = model_error_scale,
        regularization = regularization,
    )
    svd_plus_d_covar = ObservationRecipe.covariance(covar_estimator, osc)
    @test svd_plus_d_covar.diag_cov ==
          Diagonal([regularization for _ in 1:sample_size]) + Diagonal(
        vec(
            model_error_scale .* mean(SampleBuilder.get_samples(osc), dims = 2),
        ) .^ 2,
    )

    # Test latitude weights
    covar_estimator_lat_weights = ObservationRecipe.SVDplusDCovariance(
        use_latitude_weights = true,
        min_cosd_lat = 0.2,
    )
    covar_estimator_no_lat_weights =
        ObservationRecipe.SVDplusDCovariance(use_latitude_weights = false)
    svd_plus_d_covar_with_lat_weights =
        ObservationRecipe.covariance(covar_estimator_lat_weights, osc)
    svd_plus_d_covar_with_no_lat_weights =
        ObservationRecipe.covariance(covar_estimator_no_lat_weights, osc)
    # Note: Computing the SVD part of the covariance matrix is random, so it is
    # difficult to test. As such, we check that the result is different when
    # latitude weights is not applied
    function reconstruct(A)
        U, S, V = A
        return U * Diagonal(S) * V'
    end
    @test any(
        .!isapprox.(
            reconstruct(svd_plus_d_covar_with_lat_weights.svd_cov),
            reconstruct(svd_plus_d_covar_with_no_lat_weights.svd_cov),
        ),
    )

    # Test rank
    covar_estimator_rank0 = ObservationRecipe.SVDplusDCovariance(; rank = 0)
    svd_plus_d_covar_rank0 =
        ObservationRecipe.covariance(covar_estimator_rank0, osc)
    # It is silly to use a rank of 0 for SVD, but it simplifies testing
    # There are no eigenvalues for a rank 0 SVD
    @test isempty(svd_plus_d_covar_rank0.svd_cov.S)

    covar_estimator_rank10 = ObservationRecipe.SVDplusDCovariance(; rank = 10)
    @test_logs (:warn, r"rank") ObservationRecipe.covariance(
        covar_estimator_rank10,
        osc,
    )

    # Test float type
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        model_error_scale = 2.0f0,
        regularization = 1.0f0,
    )
    osc32 = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [var],
            sample_date_ranges;
            FT = Float32,
        ),
        1,
    )
    svd_plus_d_covar = ObservationRecipe.covariance(covar_estimator, osc32)

    if pkgversion(EnsembleKalmanProcesses) >= v"2.5.0"
        # See https://github.com/CliMA/EnsembleKalmanProcesses.jl/issues/504
        @test eltype(svd_plus_d_covar.svd_cov.S) == Float32
        @test eltype(svd_plus_d_covar.svd_cov.U) == Float32
        @test eltype(svd_plus_d_covar.svd_cov.V) == Float32
        @test eltype(svd_plus_d_covar.svd_cov.Vt) == Float32
        @test eltype(svd_plus_d_covar.diag_cov.diag) == Float32
    end

    # Float32 samples must stay Float32 even when model_error_scale and
    # regularization are given as Float64 literals (the diagonal part should not
    # silently widen to Float64)
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        model_error_scale = 1.0,
        regularization = 1e-3,
    )
    svd_plus_d_covar = ObservationRecipe.covariance(covar_estimator, osc32)
    @test eltype(svd_plus_d_covar.diag_cov.diag) == Float32
    if pkgversion(EnsembleKalmanProcesses) >= v"2.5.0"
        @test eltype(svd_plus_d_covar.svd_cov.S) == Float32
    end

    # Error handling: negative rank
    @test_throws ErrorException ObservationRecipe.SVDplusDCovariance(rank = -1)
end

@testset "Quantile regularization" begin
    # Test constructor
    q_reg = ObservationRecipe.QuantileRegularization(0.05)
    @test q_reg.qtl == 0.05

    @test_throws ErrorException ObservationRecipe.QuantileRegularization(0.0)
    @test_throws ErrorException ObservationRecipe.QuantileRegularization(-0.05)
    @test_throws ErrorException ObservationRecipe.QuantileRegularization(1.1)

    function compute_model_error_scale_vec(osc, model_error_scale)
        return vec(
            model_error_scale .* mean(SampleBuilder.get_samples(osc), dims = 2),
        ) .^ 2
    end

    # Computing with quantile regularization and no regularization and
    # extracting the quantile value
    # Single OutputVar
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

    model_error_scale = 0.05
    covar_estimator_with_q_reg = ObservationRecipe.SVDplusDCovariance(;
        regularization = q_reg,
        model_error_scale,
    )
    covar_estimator_with_no_reg = ObservationRecipe.SVDplusDCovariance(;
        regularization = 0.0,
        model_error_scale,
    )

    osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [var],
            sample_date_ranges;
            FT = Float64,
        ),
        1,
    )

    svd_plus_d_q_reg =
        ObservationRecipe.covariance(covar_estimator_with_q_reg, osc)

    svd_plus_d_no_reg =
        ObservationRecipe.covariance(covar_estimator_with_no_reg, osc)

    q_reg_of_model_error_scale_vec =
        svd_plus_d_q_reg.diag_cov.diag .- svd_plus_d_no_reg.diag_cov.diag

    model_error_scale_vec =
        compute_model_error_scale_vec(osc, model_error_scale)
    five_percentile =
        first(Statistics.quantile(model_error_scale_vec, [q_reg.qtl]))

    @test all(q_reg_of_model_error_scale_vec .≈ five_percentile)

    # Multiple OutputVars
    var2 =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", [0.0, 1.0, 10.0], units = "degrees") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    var2 = ClimaAnalysis.average_season_across_time(var2)
    # Want the quantile to be different
    @. var2.data = sin(var2.data)

    osc12 = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [var, var2],
            sample_date_ranges;
            FT = Float64,
        ),
        1,
    )

    svd_plus_d_q_reg =
        ObservationRecipe.covariance(covar_estimator_with_q_reg, osc12)

    svd_plus_d_no_reg =
        ObservationRecipe.covariance(covar_estimator_with_no_reg, osc12)

    q_reg_of_model_error_scale_vec =
        svd_plus_d_q_reg.diag_cov.diag .- svd_plus_d_no_reg.diag_cov.diag

    i = 0
    for v in (var, var2)
        osc_v = SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times(
                [v],
                sample_date_ranges;
                FT = Float64,
            ),
            1,
        )
        model_error_scale_vec =
            compute_model_error_scale_vec(osc_v, model_error_scale)
        five_percentile =
            first(Statistics.quantile(model_error_scale_vec, [q_reg.qtl]))
        n = length(model_error_scale_vec)
        @test all(
            q_reg_of_model_error_scale_vec[(i + 1):(i + n)] .≈ five_percentile,
        )
        i += n
    end

    # Error handling
    small_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    small_var = ClimaAnalysis.average_season_across_time(small_var)
    small_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times([small_var], sample_date_ranges),
        1,
    )
    @test_throws r"Insufficient samples for computing quantile" ObservationRecipe.covariance(
        covar_estimator_with_q_reg,
        small_osc,
    )
    covar_estimator_no_model_error_scale = ObservationRecipe.SVDplusDCovariance(
        regularization = q_reg,
        model_error_scale = 0.0,
    )
    @test_throws r"Zero found for the quantile" ObservationRecipe.covariance(
        covar_estimator_no_model_error_scale,
        osc,
    )
end

@testset "Seasonal diagonal covariance" begin
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(3 * i) for i in 0:11],
        )
    data = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, 233.0]
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

    # Each sample is one year of seasonal statistics; the variance is across the
    # years (columns) of the sample collection
    sample_date_ranges = [
        (Dates.DateTime(i, 12, 1), Dates.DateTime(i + 1, 9, 1)) for
        i in 2007:2009
    ]
    osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [var],
            sample_date_ranges;
            FT = Float64,
        ),
        1,
    )

    # No regularization and model scale
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()
    seasonal_covariance = ObservationRecipe.covariance(covar_estimator, osc)
    @test seasonal_covariance ==
          Diagonal([nanvar(season) for season in (DJF, MAM, JJA, SON)])

    # Include regularization
    regularization = 2.0
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance(
        regularization = regularization,
    )
    seasonal_covariance = ObservationRecipe.covariance(covar_estimator, osc)
    @test seasonal_covariance ==
          Diagonal([nanvar(season) for season in (DJF, MAM, JJA, SON)]) + 2 * I

    # Include model scale
    model_error_scale = 3.0
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance(
        model_error_scale = model_error_scale,
    )
    seasonal_covariance = ObservationRecipe.covariance(covar_estimator, osc)
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
    # One DJF sample per year
    time_ranges = [
        (Dates.DateTime(i, 12, 1), Dates.DateTime(i, 12, 1)) for i in 2007:2008
    ]
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance(
        use_latitude_weights = true,
    )
    lat_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [lat_var],
            time_ranges;
            FT = Float64,
        ),
        1,
    )
    seasonal_covariance = ObservationRecipe.covariance(covar_estimator, lat_osc)
    sliced_var = ClimaAnalysis.slice(lat_var, time = 0.0)
    @test seasonal_covariance ==
          (1 / 2) *
          Diagonal(ClimaAnalysis.flatten(ext._lat_weights_var(sliced_var)).data)

    # Lat weight with NaN in data (NaN positions consistent across years)
    data = lat_var.data
    data[:, 3] .= NaN
    data[:, 4] .= NaN
    lat_var = ClimaAnalysis.remake(lat_var, data = data)
    lat_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [lat_var],
            time_ranges;
            FT = Float64,
        ),
        1,
    )
    seasonal_covariance = ObservationRecipe.covariance(covar_estimator, lat_osc)
    sliced_var = ClimaAnalysis.slice(lat_var, time = 0.0)
    @test seasonal_covariance ==
          (1 / 2) *
          Diagonal(ClimaAnalysis.flatten(ext._lat_weights_var(sliced_var)).data)

    # Check float type
    covar_estimator = ObservationRecipe.SeasonalDiagonalCovariance()
    osc32 = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [var],
            sample_date_ranges;
            FT = Float32,
        ),
        1,
    )
    seasonal_covariance = ObservationRecipe.covariance(covar_estimator, osc32)
    @test eltype(seasonal_covariance) == Float32

    # Error handling
    # Fewer than two samples
    one_sample_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [var],
            [first(sample_date_ranges)],
        ),
        1,
    )
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        one_sample_osc,
    )

    seasonal_time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2006, 12),
            [Dates.DateTime(2006, 12) + Dates.Month(3 * i) for i in 0:20],
        )
    seasonal_var =
        TemplateVar() |>
        add_dim("time", seasonal_time, units = "s") |>
        add_attribs(short_name = "hi", start_date = "2006-12-1") |>
        one_to_n_data() |>
        initialize

    # Every variable must have a time dimension
    no_time_var =
        TemplateVar() |>
        add_dim("latitude", [-90.0, 0.0, 90.0], units = "deg") |>
        add_attribs(short_name = "hi") |>
        one_to_n_data() |>
        initialize
    no_time_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples(reshape([no_time_var, no_time_var], 1, :)),
        1,
    )
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        no_time_osc,
    )

    # Multiple time points map to the same season and year (monthly data, so
    # each window has both December and January of the same DJF)
    monthly_time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(i) for i in 0:14],
        )
    monthly_var =
        TemplateVar() |>
        add_dim("time", monthly_time, units = "s") |>
        add_attribs(short_name = "hi", start_date = "2007-12-1") |>
        one_to_n_data() |>
        initialize
    duplicate_season_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [monthly_var],
            [
                (Dates.DateTime(2007, 12), Dates.DateTime(2008, 1)),
                (Dates.DateTime(2008, 12), Dates.DateTime(2009, 1)),
            ],
        ),
        1,
    )
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        duplicate_season_osc,
    )

    # More than four season and year combinations for a single variable (each
    # window spans five seasons)
    five_season_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [seasonal_var],
            [
                (Dates.DateTime(2006, 12), Dates.DateTime(2007, 12)),
                (Dates.DateTime(2007, 12), Dates.DateTime(2008, 12)),
            ],
        ),
        1,
    )
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        five_season_osc,
    )

    # Seasons do not come from the same year (SON of one year and DJF of the
    # next year, which find_season_and_year assigns to different years)
    multiple_year_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [seasonal_var],
            [
                (Dates.DateTime(2007, 9), Dates.DateTime(2007, 12)),
                (Dates.DateTime(2008, 9), Dates.DateTime(2008, 12)),
            ],
        ),
        1,
    )
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        multiple_year_osc,
    )

    # Order of seasons differs across samples (one column is [DJF, MAM] and the
    # other is [MAM, JJA])
    misaligned_season_osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [seasonal_var],
            [
                (Dates.DateTime(2007, 12), Dates.DateTime(2008, 3)),
                (Dates.DateTime(2008, 3), Dates.DateTime(2008, 6)),
            ],
        ),
        1,
    )
    @test_throws ErrorException ObservationRecipe.covariance(
        covar_estimator,
        misaligned_season_osc,
    )
end

@testset "Observation" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    x = [1.0, 2.0]
    y = [3.0]
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
        add_dim("x", x, units = "m") |>
        add_dim("y", y, units = "m") |>
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
        model_error_scale = 1e-2,
        regularization = 1e-3,
    )

    start_date = Dates.DateTime(2007, 12)
    end_date = Dates.DateTime(2008, 9)
    osc = SampleBuilder.choose_obs(
        SampleBuilder.build_samples_by_times(
            [var, neg_var],
            sample_date_ranges;
            FT = Float64,
        ),
        1,
    )
    obs = ObservationRecipe.observation(covar_estimator, osc)

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
    data1 =
        ClimaAnalysis.flatten(window_dates(var); dims = ext.FLATTENED_DIMS).data
    data2 =
        ClimaAnalysis.flatten(
            window_dates(neg_var);
            dims = ext.FLATTENED_DIMS,
        ).data

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
            ClimaAnalysis.unflatten(obs.metadata[1], obs.samples[1][1:160])
        neg_unflattened_var =
            ClimaAnalysis.unflatten(obs.metadata[2], obs.samples[1][161:end])

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
end

@testset "Observation with no time dimension" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    x = [1.0, 2.0]
    y = [3.0]
    var =
        TemplateVar() |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_dim("x", x, units = "m") |>
        add_dim("y", y, units = "m") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    neg_var = -2.0 * var
    neg_var.attributes["hi"] = "hello"
    neg_var.dim_attributes["lon"]["a dim"] = "attribute"

    covar_estimator = ObservationRecipe.ScalarCovariance(scalar = 2.0)

    osc =
        SampleBuilder.choose_obs(SampleBuilder.build_samples([var, neg_var]), 1)
    obs = ObservationRecipe.observation(covar_estimator, osc)

    # We already test the covariance matrix, so we test if the flattened
    # sample is formed correctly and if the metadata is correct
    # Test if flattened sample is correct
    data1 = ClimaAnalysis.flatten(var; dims = ext.FLATTENED_DIMS).data
    data2 = ClimaAnalysis.flatten(neg_var; dims = ext.FLATTENED_DIMS).data

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
            ClimaAnalysis.unflatten(obs.metadata[1], obs.samples[1][1:40])
        neg_unflattened_var =
            ClimaAnalysis.unflatten(obs.metadata[2], obs.samples[1][41:end])

        # Test if the two OutputVars are equivalent
        @test unflattened_var.data == var.data
        @test unflattened_var.attributes == var.attributes
        @test unflattened_var.dim_attributes == var.dim_attributes
        @test unflattened_var.dims == var.dims

        @test neg_unflattened_var.data == neg_var.data
        @test neg_unflattened_var.attributes == neg_var.attributes
        @test neg_unflattened_var.dim_attributes == neg_var.dim_attributes
        @test neg_unflattened_var.dims == neg_var.dims
    end
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
        osc = SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times(
                [var1, var2, var3],
                [(Dates.DateTime(2007, 12, 1), Dates.DateTime(2008, 1, 1))],
            ),
            1,
        )
        obs = ObservationRecipe.observation(covar_estimator, osc)
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
    sample_date_ranges = [
        (Dates.DateTime(i, 12, 1), Dates.DateTime(i + 1, 9, 1)) for
        i in 2007:2018
    ]
    obs1 = ObservationRecipe.observation(
        covar_estimator,
        SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times(
                [time_var, lon_var],
                sample_date_ranges,
            ),
            1,
        ),
    )
    obs2 = ObservationRecipe.observation(
        covar_estimator,
        SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times(
                [time_var],
                sample_date_ranges,
            ),
            1,
        ),
    )
    obs3 = ObservationRecipe.observation(
        covar_estimator,
        SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times([lon_var], sample_date_ranges),
            1,
        ),
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

    metadata1 = ClimaCalibrate.get_metadata_for_nth_iteration(obs_series, 1)
    metadata2 = ClimaCalibrate.get_metadata_for_nth_iteration(obs_series, 2)
    metadata3 = ClimaCalibrate.get_metadata_for_nth_iteration(obs_series, 3)
    metadata_indices1 =
        ObservationRecipe._get_minibatch_indices_for_nth_iteration(
            obs_series,
            1,
        )
    metadata_indices2 =
        ObservationRecipe._get_minibatch_indices_for_nth_iteration(
            obs_series,
            2,
        )
    metadata_indices3 =
        ObservationRecipe._get_minibatch_indices_for_nth_iteration(
            obs_series,
            3,
        )
    obs1 = ClimaCalibrate.get_observations_for_nth_iteration(obs_series, 1)
    obs2 = ClimaCalibrate.get_observations_for_nth_iteration(obs_series, 2)
    obs3 = ClimaCalibrate.get_observations_for_nth_iteration(obs_series, 3)

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

    @test length(metadata_indices2) == 3
    @test metadata_indices2[1] == 1:12
    @test metadata_indices2[2] == 13:16
    @test metadata_indices2[3] == 17:28

    @test length(metadata3) == 3
    @test metadata3[1].attributes["short_name"] == "time"
    @test metadata3[2].attributes["short_name"] == "lon"
    @test metadata3[3].attributes["short_name"] == "time"

    @test length(metadata_indices3) == 3
    @test metadata_indices3[1] == 1:4
    @test metadata_indices3[2] == 5:16
    @test metadata_indices3[3] == 17:20

    @test obs1 == [obs_series.observations[1], obs_series.observations[2]]
    @test obs2 == [obs_series.observations[3], obs_series.observations[4]]
    @test obs3 == [obs_series.observations[1], obs_series.observations[2]]
end

@testset "Reconstruct g and g mean" begin
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
            long_name = "Time",
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
            long_name = "Longitude",
            start_date = "2007-12-1",
            super = "fun",
        ) |>
        one_to_n_data() |>
        initialize

    trim_var(var, start_date, end_date) =
        ClimaAnalysis.window(var, "time", left = start_date, right = end_date)

    time_var_length = length(
        trim_var(
            time_var,
            Dates.DateTime(2007, 12),
            Dates.DateTime(2008, 9),
        ).data,
    )
    lon_var_length = length(
        trim_var(
            lon_var,
            Dates.DateTime(2007, 12),
            Dates.DateTime(2008, 9),
        ).data,
    )

    covar_estimator = ObservationRecipe.ScalarCovariance()

    observations = [
        ObservationRecipe.observation(
            covar_estimator,
            SampleBuilder.choose_obs(
                SampleBuilder.build_samples_by_times(
                    [time_var, lon_var],
                    [(Dates.DateTime(year, 12), Dates.DateTime(year + 1, 9))],
                ),
                1,
            ),
        ) for year in (2007, 2008, 2009, 2010)
    ]

    obs_series = EKP.ObservationSeries(
        Dict(
            "observations" => observations,
            "names" => ["1", "2", "3", "4"],
            "minibatcher" =>
                ClimaCalibrate.minibatcher_over_samples([1, 2, 3, 4], 2),
        ),
    )

    prior = constrained_gaussian("pi_groups_coeff", 1.0, 0.3, 0, Inf)

    eki = EKP.EnsembleKalmanProcess(
        obs_series,
        EKP.TransformUnscented(prior, impose_prior = true),
        verbose = true,
        scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    )
    G_ens = ClimaCalibrate.g_ens_matrix(eki)
    col = vcat(
        fill(1, time_var_length),
        fill(2, lon_var_length),
        fill(3, time_var_length),
        fill(4, lon_var_length),
    )
    M = hcat(col, 2 * col, 3 * col)
    G_ens .= M
    EKP.update_ensemble!(eki, G_ens)

    g_ens_as_vars = ObservationRecipe.reconstruct_g(eki, 1)
    @test size(g_ens_as_vars) == (4, 3)
    @test ClimaAnalysis.short_name.(g_ens_as_vars[:, 1]) ==
          ["time", "lon", "time", "lon"]
    for I in CartesianIndices(g_ens_as_vars)
        i, j = Tuple(I)
        @test all(g_ens_as_vars[i, j].data .== i * j)
    end
    # This is an error thrown by EKP which is a BoundsError, but we only care
    # that some kind of error is thrown here
    @test_throws Any ObservationRecipe.reconstruct_g(eki, 2)

    g_mean_as_vars = ObservationRecipe.reconstruct_g_mean(eki, 1)
    @test length(g_mean_as_vars) == 4
    @test ClimaAnalysis.short_name.(g_mean_as_vars) ==
          ["time", "lon", "time", "lon"]
    for (i, var) in enumerate(g_mean_as_vars)
        # For i = 1, it is the average of 1, 2, and 3.
        # For i = 2, it is the average of 2, 4, and 6.
        # This pattern continues for i = 3 and i = 4.
        @test all(var.data .== 2 * i)
    end
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
        sample_date_ranges = [
            (Dates.DateTime(i, 12, 1), Dates.DateTime(i + 1, 9, 1)) for
            i in 2007:2010
        ]
        sc = SampleBuilder.build_samples_by_times(
            [time_var, lon_var],
            sample_date_ranges,
        )
        obs = ObservationRecipe.observation(
            covar_estimator,
            SampleBuilder.choose_obs(sc, 1),
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
            SampleBuilder.choose_obs(sc, 1),
        )
        obs2 = ObservationRecipe.observation(
            covar_estimator,
            SampleBuilder.choose_obs(sc, 2),
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
            SampleBuilder.choose_obs(sc, 3),
        )
        obs4 = ObservationRecipe.observation(
            covar_estimator,
            SampleBuilder.choose_obs(sc, 4),
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

@testset "Reconstruct diagonal of covariance and OutputVars from observations" begin
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2007, 12),
            [Dates.DateTime(2007, 12) + Dates.Month(3 * i) for i in 0:47],
        )
    time_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(short_name = "time", start_date = "2007-12-1") |>
        one_to_n_data() |>
        initialize

    lon = [-45.0, 0.0, 45.0]
    lon_var =
        TemplateVar() |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("time", time, units = "s") |>
        add_attribs(short_name = "lon", start_date = "2007-12-1") |>
        one_to_n_data() |>
        initialize

    covar_estimator = ObservationRecipe.ScalarCovariance()
    # Each sample spans two years (8 seasons); two samples are enough for a
    # variance
    sample_date_ranges = [
        (Dates.DateTime(2007, 12), Dates.DateTime(2009, 9)),
        (Dates.DateTime(2009, 12), Dates.DateTime(2011, 9)),
    ]
    obs1 = ObservationRecipe.observation(
        covar_estimator,
        SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times(
                [time_var, lon_var],
                sample_date_ranges,
            ),
            1,
        ),
    )
    obs2 = ObservationRecipe.observation(
        covar_estimator,
        SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times(
                [time_var],
                sample_date_ranges,
            ),
            1,
        ),
    )
    obs3 = ObservationRecipe.observation(
        covar_estimator,
        SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times(
                [permutedims(lon_var, ("time", "lon"))],
                sample_date_ranges,
            ),
            1,
        ),
    )

    covs1 = ObservationRecipe.reconstruct_diag_cov(obs1)
    covs2 = ObservationRecipe.reconstruct_diag_cov(obs2)
    covs3 = ObservationRecipe.reconstruct_diag_cov(obs3)

    @test length(covs1) == 2
    @test length(covs2) == 1
    @test length(covs3) == 1

    time_var_from_covs1 = covs1[1]
    lat_var_from_covs2 = covs1[2]
    time_var_from_covs2 = covs2[1]
    lat_var_from_covs3 = permutedims(covs3[1], ("lon", "time"))

    @test isequal(time_var_from_covs1.data, time_var_from_covs2.data)
    @test isequal(lat_var_from_covs2.data, lat_var_from_covs3.data)

    # Reconstruct OutputVars from observations
    vars1 = ObservationRecipe.reconstruct_vars(obs1)
    vars2 = ObservationRecipe.reconstruct_vars(obs2)
    vars3 = ObservationRecipe.reconstruct_vars(obs3)

    @test length(vars1) == 2
    @test length(vars2) == 1
    @test length(vars3) == 1

    # Check dimensions
    @test vars1[1].dims["time"] == time[1:8]
    @test vars1[2].dims["lon"] == lon
    @test vars1[2].dims["time"] == time[1:8]
    @test vars2[1].dims["time"] == time[1:8]
    @test vars3[1].dims["time"] == time[1:8]
    @test vars3[1].dims["lon"] == lon

    # Check attributes and dimension attributes
    for (reconstructed_var, original_var) in zip(
        (vars1[1], vars1[2], vars2[1], vars3[1]),
        (time_var, lon_var, time_var, lon_var),
    )
        @test reconstructed_var.attributes == original_var.attributes
        @test reconstructed_var.dim_attributes == original_var.dim_attributes
    end

    # Check data
    @test vars1[1].data == time_var.data[1:8]
    @test vars1[2].data == lon_var.data[:, 1:8]
    @test vars2[1].data == time_var.data[1:8]
    @test vars3[1].data == permutedims(lon_var.data[:, 1:8], (2, 1))

    var3d =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lat", [-45.0, 10.0, 42.0], units = "degrees") |>
        add_dim("lon", [0.0, 1.0], units = "degrees") |>
        add_attribs(short_name = "time", start_date = "2007-12-1") |>
        ones_data() |>
        initialize

    covar_estimator = ObservationRecipe.ScalarCovariance(
        scalar = 5.0,
        use_latitude_weights = true,
        min_cosd_lat = 0.01,
    )
    obs4 = ObservationRecipe.observation(
        covar_estimator,
        SampleBuilder.choose_obs(
            SampleBuilder.build_samples_by_times(
                var3d,
                [(Dates.DateTime(2007, 12), Dates.DateTime(2007, 12))],
            ),
            1,
        ),
    )
    covs4 = ObservationRecipe.reconstruct_diag_cov(obs4)
    var_from_covs4 = covs4[1]
    # The second index is the latitude dimension and with ScalarCovariance with
    # latitude weights, the weights should all be the same along a particular
    # latitude
    for i in 1:3
        @test allequal(var_from_covs4.data[:, i, :])
    end

    # Test that none of them are equal to any of the other ones
    @test !isequal(var_from_covs4.data[:, 1, :], var_from_covs4.data[:, 2, :])
    @test !isequal(var_from_covs4.data[:, 2, :], var_from_covs4.data[:, 3, :])
    @test !isequal(var_from_covs4.data[:, 1, :], var_from_covs4.data[:, 3, :])

    # Test reconstruction of obs4
    vars4 = ObservationRecipe.reconstruct_vars(obs4)
    @test length(vars4) == 1
    @test vars4[1].attributes == var3d.attributes
    @test vars4[1].dim_attributes == var3d.dim_attributes
    @test vars4[1].dims["time"] == [0.0]
    @test vars4[1].dims["lat"] == var3d.dims["lat"]
    @test vars4[1].dims["lon"] == var3d.dims["lon"]
    @test vars4[1].data == var3d.data[[1], :, :]
end

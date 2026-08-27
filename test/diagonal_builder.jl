using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.SampleBuilder
import Statistics: mean
import Statistics
import LinearAlgebra: Diagonal

import ClimaAnalysis.Template:
    TemplateVar, add_dim, add_attribs, one_to_n_data, initialize

# Diagonal builders used to test that build_diagonal can be extended (structs
# cannot be defined inside a testset)
struct VarianceDiagonal <: ObservationRecipe.AbstractDiagonalBuilder end

function ObservationRecipe.build_diagonal(::VarianceDiagonal, sample_collection)
    samples = SampleBuilder.get_samples(sample_collection)
    return Diagonal(vec(Statistics.var(samples, dims = 2)))
end

struct WrongSizeDiagonal <: ObservationRecipe.AbstractDiagonalBuilder end

ObservationRecipe.build_diagonal(::WrongSizeDiagonal, sample_collection) =
    Diagonal([1.0, 2.0])

struct NotADiagonal <: ObservationRecipe.AbstractDiagonalBuilder end

function ObservationRecipe.build_diagonal(::NotADiagonal, sample_collection)
    n = size(SampleBuilder.get_samples(sample_collection), 1)
    return ones(n, n)
end

struct DenseDiagonal <: ObservationRecipe.AbstractDiagonalBuilder end

function ObservationRecipe.build_diagonal(::DenseDiagonal, sample_collection)
    n = size(SampleBuilder.get_samples(sample_collection), 1)
    return collect(Diagonal(fill(2.0, n)))
end

@testset "Diagonal builders" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = ClimaAnalysis.Utils.date_to_time.(
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

    sample_collection = SampleBuilder.build_samples_by_times(
        [var],
        sample_date_ranges;
        FT = Float64,
    )
    samples = SampleBuilder.get_samples(sample_collection)
    n = size(samples, 1)

    # ScalarDiagonal
    scalar_diagonal = ObservationRecipe.ScalarDiagonal(1e-3)
    @test ObservationRecipe.build_diagonal(
        scalar_diagonal,
        sample_collection,
    ) == Diagonal(fill(1e-3, n))

    # ModelErrorScaleDiagonal
    mes_diagonal = ObservationRecipe.ModelErrorScaleDiagonal(2.0)
    @test ObservationRecipe.build_diagonal(mes_diagonal, sample_collection) ==
          Diagonal(vec((2.0 .* mean(samples, dims = 2)) .^ 2))

    # Summing diagonal builders builds the sum of the diagonal matrices
    sum_diagonal = mes_diagonal + scalar_diagonal
    @test sum_diagonal isa ObservationRecipe.SumDiagonal
    @test ObservationRecipe.build_diagonal(sum_diagonal, sample_collection) ==
          ObservationRecipe.build_diagonal(mes_diagonal, sample_collection) +
          ObservationRecipe.build_diagonal(scalar_diagonal, sample_collection)

    # Sums of diagonal builders are flattened
    a = ObservationRecipe.ScalarDiagonal(1.0)
    b = ObservationRecipe.ScalarDiagonal(2.0)
    c = ObservationRecipe.ScalarDiagonal(3.0)
    d = ObservationRecipe.ScalarDiagonal(4.0)
    @test (a + b + c).builders == (a, b, c)
    @test (a + (b + c)).builders == (a, b, c)
    @test ((a + b) + (c + d)).builders == (a, b, c, d)
    @test ObservationRecipe.build_diagonal(a + b + c + d, sample_collection) ==
          Diagonal(fill(10.0, n))

    # QuantileDiagonal computes the quantile of the diagonal built from the
    # wrapped diagonal builder (there is a single variable, so there is a
    # single block along the diagonal)
    qtl_diagonal = ObservationRecipe.QuantileDiagonal(0.3, mes_diagonal)
    mes_diag_vec =
        ObservationRecipe.build_diagonal(mes_diagonal, sample_collection).diag
    @test ObservationRecipe.build_diagonal(qtl_diagonal, sample_collection) ==
          Diagonal(fill(Statistics.quantile(mes_diag_vec, 0.3), n))

    # QuantileDiagonal can wrap a SumDiagonal
    qtl_of_sum = ObservationRecipe.QuantileDiagonal(0.3, sum_diagonal)
    sum_diag_vec =
        ObservationRecipe.build_diagonal(sum_diagonal, sample_collection).diag
    @test ObservationRecipe.build_diagonal(qtl_of_sum, sample_collection) ==
          Diagonal(fill(Statistics.quantile(sum_diag_vec, 0.3), n))

    # Float32 samples must stay Float32 even when the diagonal builders are
    # given Float64 literals
    sample_collection32 = SampleBuilder.build_samples_by_times(
        [var],
        sample_date_ranges;
        FT = Float32,
    )
    for diagonal_builder in (
        ObservationRecipe.ScalarDiagonal(1e-3),
        ObservationRecipe.ModelErrorScaleDiagonal(2.0),
        ObservationRecipe.QuantileDiagonal(
            0.3,
            ObservationRecipe.ModelErrorScaleDiagonal(2.0),
        ),
        ObservationRecipe.ModelErrorScaleDiagonal(2.0) +
        ObservationRecipe.ScalarDiagonal(1e-3),
    )
        @test eltype(
            ObservationRecipe.build_diagonal(
                diagonal_builder,
                sample_collection32,
            ).diag,
        ) == Float32
    end

    # A user-defined diagonal builder can be used with SVDplusDCovariance and
    # composed with the built-in diagonal builders
    covar_estimator = ObservationRecipe.SVDplusDCovariance(
        diagonal = VarianceDiagonal() + ObservationRecipe.ScalarDiagonal(1.0),
    )
    svd_plus_d_covar =
        ObservationRecipe.covariance(covar_estimator, sample_collection)
    @test svd_plus_d_covar.diag_cov ==
          Diagonal(vec(Statistics.var(samples, dims = 2)) .+ 1.0)

    # A diagonal matrix that is not a Diagonal is converted to a Diagonal
    covar_estimator =
        ObservationRecipe.SVDplusDCovariance(diagonal = DenseDiagonal())
    svd_plus_d_covar =
        ObservationRecipe.covariance(covar_estimator, sample_collection)
    @test svd_plus_d_covar.diag_cov == Diagonal(fill(2.0, n))

    # Error handling: matrix from build_diagonal with the wrong size or that
    # is not a diagonal matrix
    @test_throws ErrorException ObservationRecipe.covariance(
        ObservationRecipe.SVDplusDCovariance(diagonal = WrongSizeDiagonal()),
        sample_collection,
    )
    @test_throws ErrorException ObservationRecipe.covariance(
        ObservationRecipe.SVDplusDCovariance(diagonal = NotADiagonal()),
        sample_collection,
    )
end

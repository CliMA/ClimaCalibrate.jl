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

# Diagonal terms used to test that compute_diagonal can be extended (structs
# cannot be defined inside a testset)
struct VarianceDiagonal <: ObservationRecipe.AbstractDiagonalTerm end

function ObservationRecipe.compute_diagonal(
    ::VarianceDiagonal,
    sample_collection,
)
    samples = SampleBuilder.get_samples(sample_collection)
    return Diagonal(vec(Statistics.var(samples, dims = 2)))
end

struct WrongSizeDiagonal <: ObservationRecipe.AbstractDiagonalTerm end

ObservationRecipe.compute_diagonal(::WrongSizeDiagonal, sample_collection) =
    Diagonal([1.0, 2.0])

struct NotADiagonal <: ObservationRecipe.AbstractDiagonalTerm end

function ObservationRecipe.compute_diagonal(::NotADiagonal, sample_collection)
    n = size(SampleBuilder.get_samples(sample_collection), 1)
    return ones(n, n)
end

struct DenseDiagonal <: ObservationRecipe.AbstractDiagonalTerm end

function ObservationRecipe.compute_diagonal(::DenseDiagonal, sample_collection)
    n = size(SampleBuilder.get_samples(sample_collection), 1)
    return collect(Diagonal(fill(2.0, n)))
end

@testset "Diagonal terms" begin
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
    @test ObservationRecipe.compute_diagonal(
        scalar_diagonal,
        sample_collection,
    ) == Diagonal(fill(1e-3, n))

    # ModelErrorScaleDiagonal
    mes_diagonal = ObservationRecipe.ModelErrorScaleDiagonal(2.0)
    @test ObservationRecipe.compute_diagonal(mes_diagonal, sample_collection) ==
          Diagonal(vec((2.0 .* mean(samples, dims = 2)) .^ 2))

    # Summing diagonal terms builds the sum of the diagonal matrices
    sum_diagonal = mes_diagonal + scalar_diagonal
    @test sum_diagonal isa ObservationRecipe.SumDiagonal
    @test ObservationRecipe.compute_diagonal(sum_diagonal, sample_collection) ==
          ObservationRecipe.compute_diagonal(mes_diagonal, sample_collection) +
          ObservationRecipe.compute_diagonal(scalar_diagonal, sample_collection)

    # Sums of diagonal terms are flattened
    a = ObservationRecipe.ScalarDiagonal(1.0)
    b = ObservationRecipe.ScalarDiagonal(2.0)
    c = ObservationRecipe.ScalarDiagonal(3.0)
    d = ObservationRecipe.ScalarDiagonal(4.0)
    @test (a + b + c).terms == (a, b, c)
    @test (a + (b + c)).terms == (a, b, c)
    @test ((a + b) + (c + d)).terms == (a, b, c, d)
    @test ObservationRecipe.compute_diagonal(
        a + b + c + d,
        sample_collection,
    ) == Diagonal(fill(10.0, n))

    # QuantileDiagonal computes the quantile of the diagonal built from the
    # wrapped diagonal term (there is a single variable, so there is a
    # single block along the diagonal)
    qtl_diagonal = ObservationRecipe.QuantileDiagonal(0.3, mes_diagonal)
    mes_diag_vec =
        ObservationRecipe.compute_diagonal(mes_diagonal, sample_collection).diag
    @test ObservationRecipe.compute_diagonal(qtl_diagonal, sample_collection) ==
          Diagonal(fill(Statistics.quantile(mes_diag_vec, 0.3), n))

    # QuantileDiagonal can wrap a SumDiagonal
    qtl_of_sum = ObservationRecipe.QuantileDiagonal(0.3, sum_diagonal)
    sum_diag_vec =
        ObservationRecipe.compute_diagonal(sum_diagonal, sample_collection).diag
    @test ObservationRecipe.compute_diagonal(qtl_of_sum, sample_collection) ==
          Diagonal(fill(Statistics.quantile(sum_diag_vec, 0.3), n))

    # Float32 samples must stay Float32 even when the diagonal terms are
    # given Float64 literals
    sample_collection32 = SampleBuilder.build_samples_by_times(
        [var],
        sample_date_ranges;
        FT = Float32,
    )
    for diagonal_term in (
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
            ObservationRecipe.compute_diagonal(
                diagonal_term,
                sample_collection32,
            ).diag,
        ) == Float32
    end

    # A user-defined diagonal term can be used with SVDplusDCovariance and
    # composed with the built-in diagonal terms
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

    # Error handling: matrix from compute_diagonal with the wrong size or that
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

@testset "QuantileDiagonal with multiple variables" begin
    pr_time = [0.0, 1.0, 2.0]
    pr_lat = [-90.0, 0.0, 90.0]
    rsut_time = [0.0, 1.0, 2.0]
    rsut_lat = [-45.0, 45.0]
    pr_var =
        TemplateVar() |>
        add_dim("time", pr_time, units = "s") |>
        add_dim("lat", pr_lat, units = "degrees") |>
        add_attribs(
            short_name = "pr",
            start_date = "2008-1-1",
            units = "mm/day",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    rsut_var =
        TemplateVar() |>
        add_dim("time", rsut_time, units = "s") |>
        add_dim("lat", rsut_lat, units = "degrees") |>
        add_attribs(
            short_name = "rsut",
            start_date = "2008-1-1",
            units = "W m-2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize

    sample_collection =
        SampleBuilder.build_samples([pr_var, rsut_var]; FT = Float64)
    samples = SampleBuilder.get_samples(sample_collection)
    metadata = SampleBuilder.get_metadata(sample_collection)[:, 1]
    pr_length = ClimaAnalysis.flattened_length(metadata[1])
    pr_indices = 1:pr_length
    rsut_indices = (pr_length + 1):size(samples, 1)

    mes_diagonal = ObservationRecipe.ModelErrorScaleDiagonal(2.0)
    mes_diag_vec =
        ObservationRecipe.compute_diagonal(mes_diagonal, sample_collection).diag

    # Each variable's block of the diagonal is filled with the quantile of
    # that block, not a single quantile across the whole diagonal
    qtl_diag_vec = similar(mes_diag_vec)
    qtl_diag_vec[pr_indices] .=
        Statistics.quantile(mes_diag_vec[pr_indices], 0.5)
    qtl_diag_vec[rsut_indices] .=
        Statistics.quantile(mes_diag_vec[rsut_indices], 0.5)
    @test qtl_diag_vec[first(pr_indices)] != qtl_diag_vec[first(rsut_indices)]

    qtl_diagonal = ObservationRecipe.QuantileDiagonal(0.5, mes_diagonal)
    @test ObservationRecipe.compute_diagonal(qtl_diagonal, sample_collection) ==
          Diagonal(qtl_diag_vec)

    # Error when a variable's block of the diagonal has fewer entries than
    # 1 / qtl
    @test_throws ErrorException ObservationRecipe.compute_diagonal(
        ObservationRecipe.QuantileDiagonal(0.05, mes_diagonal),
        sample_collection,
    )

    # Error when the quantile of a variable's block of the diagonal is zero
    @test_throws ErrorException ObservationRecipe.compute_diagonal(
        ObservationRecipe.QuantileDiagonal(
            0.5,
            ObservationRecipe.ScalarDiagonal(0.0),
        ),
        sample_collection,
    )
end

using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import ClimaCalibrate.ObservationRecipe:
    AbstractDiagonalTerm,
    ScalarDiagonal,
    ModelErrorScaleDiagonal,
    VarianceDiagonal,
    QuantileDiagonal,
    SumDiagonal,
    compute_diagonal
import ClimaCalibrate.SampleBuilder
import EnsembleKalmanProcesses as EKP
import LinearAlgebra: Diagonal

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

@testset "ScalarDiagonal" begin
    # Scalar is wrapped into a vector
    term = ScalarDiagonal(1.0)
    @test term isa AbstractDiagonalTerm
    @test term.scalars == [1.0]

    # Vector constructor and eltype
    term = ScalarDiagonal([1.0, 3.0])
    @test term.scalars == [1.0, 3.0]
    @test term isa ScalarDiagonal{Float64}
    @test ScalarDiagonal(Float32[1.0]) isa ScalarDiagonal{Float32}

    # Non-float values are not accepted
    @test_throws MethodError ScalarDiagonal([1, 3])

    # Empty vectors are not accepted
    @test_throws "should not be empty" ScalarDiagonal(Float64[])

    # Variables with flattened lengths of 10 (pr) and 3 (ta)
    pr_var =
        TemplateVar() |>
        add_dim("time", collect(0.0:9.0), units = "s") |>
        add_attribs(short_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    ta_var =
        TemplateVar() |>
        add_dim("time", collect(0.0:2.0), units = "s") |>
        add_attribs(short_name = "ta", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize

    # Sample collection with a single variable and a single sample
    single_var_collection = SampleBuilder.build_samples(pr_var)
    @test compute_diagonal(ScalarDiagonal(2.0), single_var_collection) ==
          Diagonal(fill(2.0, 10))
    @test compute_diagonal(ScalarDiagonal([3.0]), single_var_collection) ==
          Diagonal(fill(3.0, 10))

    # Sample collection with two variables and a single sample
    sample_collection =
        SampleBuilder.build_samples(reshape([pr_var, ta_var], 2, 1))

    # A single scalar fills the whole diagonal
    @test compute_diagonal(ScalarDiagonal(2.0), sample_collection) ==
          Diagonal(fill(2.0, 13))

    # One scalar per variable fills the corresponding block
    @test compute_diagonal(ScalarDiagonal([1.0, 3.0]), sample_collection) ==
          Diagonal(vcat(fill(1.0, 10), fill(3.0, 3)))

    # Number of scalars must be one or match the number of variables
    @test_throws "The number of scalars" compute_diagonal(
        ScalarDiagonal([1.0, 2.0, 3.0]),
        sample_collection,
    )

    # The eltype of the diagonal matches the eltype of the samples not the
    # eltype of the term
    @test compute_diagonal(ScalarDiagonal(2.0), sample_collection) isa
          Diagonal{Float32}
    sample_collection64 = SampleBuilder.build_samples(
        reshape([pr_var, ta_var], 2, 1);
        FT = Float64,
    )
    @test compute_diagonal(ScalarDiagonal(2.0f0), sample_collection64) isa
          Diagonal{Float64}
end

@testset "ModelErrorScaleDiagonal" begin
    # Scalar is wrapped into a vector
    term = ModelErrorScaleDiagonal(0.05)
    @test term isa AbstractDiagonalTerm
    @test term.model_error_scales == [0.05]

    # Vector constructor and eltype
    term = ModelErrorScaleDiagonal([0.05, 0.1])
    @test term.model_error_scales == [0.05, 0.1]
    @test ModelErrorScaleDiagonal(Float32[0.05]) isa
          ModelErrorScaleDiagonal{Float32}

    # Zero is allowed
    @test ModelErrorScaleDiagonal(0.0).model_error_scales == [0.0]

    # Negative model error scales are not allowed
    @test_throws "should not be negative" ModelErrorScaleDiagonal(-0.05)
    @test_throws "should not be negative" ModelErrorScaleDiagonal([0.05, -0.1])

    # Non-float values are not accepted
    @test_throws MethodError ModelErrorScaleDiagonal([1, 3])

    # Empty vectors are not accepted
    @test_throws "should not be empty" ModelErrorScaleDiagonal(Float64[])

    # Two variables with flattened lengths of 10 and 3, and three samples with
    # constant data, so the mean across samples is 2.0 for pr and 5.0 for ta
    function make_var(short_name, time_len, value)
        var =
            TemplateVar() |>
            add_dim("time", collect(1.0:time_len), units = "s") |>
            add_attribs(short_name = short_name, start_date = "2008-1-1") |>
            one_to_n_data(collected = true) |>
            initialize
        var.data .= value
        return var
    end
    vars_matrix = [
        make_var("pr", 10, 1.0) make_var("pr", 10, 2.0) make_var("pr", 10, 3.0)
        make_var("ta", 3, 4.0) make_var("ta", 3, 5.0) make_var("ta", 3, 6.0)
    ]
    sample_collection = SampleBuilder.build_samples(vars_matrix)

    # Diagonal is (scale * mean)^2, where the mean is taken across samples
    @test compute_diagonal(ModelErrorScaleDiagonal(0.5), sample_collection) ==
          Diagonal(vcat(fill((0.5 * 2.0)^2, 10), fill((0.5 * 5.0)^2, 3)))

    # One scale per variable applies to the corresponding block
    @test compute_diagonal(
        ModelErrorScaleDiagonal([0.5, 2.0]),
        sample_collection,
    ) == Diagonal(vcat(fill((0.5 * 2.0)^2, 10), fill((2.0 * 5.0)^2, 3)))

    # The eltype of the diagonal matches the eltype of the samples
    @test compute_diagonal(ModelErrorScaleDiagonal(0.5), sample_collection) isa
          Diagonal{Float32}
    sample_collection64 = SampleBuilder.build_samples(vars_matrix; FT = Float64)
    @test compute_diagonal(
        ModelErrorScaleDiagonal(0.5f0),
        sample_collection64,
    ) isa Diagonal{Float64}

    # Number of scales must be one or match the number of variables
    @test_throws "The number of model error scales" compute_diagonal(
        ModelErrorScaleDiagonal([0.5, 1.0, 2.0]),
        sample_collection,
    )
end

@testset "VarianceDiagonal" begin
    term = VarianceDiagonal()
    @test term isa AbstractDiagonalTerm

    # Two variables with flattened lengths of 10 and 3, and three samples with
    # constant data, so the variance across samples is 1.0 for pr ((1, 2, 3))
    # and 4.0 for ta ((1, 3, 5))
    function make_var(short_name, time_len, value)
        var =
            TemplateVar() |>
            add_dim("time", collect(1.0:time_len), units = "s") |>
            add_attribs(short_name = short_name, start_date = "2008-1-1") |>
            one_to_n_data(collected = true) |>
            initialize
        var.data .= value
        return var
    end
    vars_matrix = [
        make_var("pr", 10, 1.0) make_var("pr", 10, 2.0) make_var("pr", 10, 3.0)
        make_var("ta", 3, 1.0) make_var("ta", 3, 3.0) make_var("ta", 3, 5.0)
    ]
    sample_collection = SampleBuilder.build_samples(vars_matrix)

    @test compute_diagonal(VarianceDiagonal(), sample_collection) ==
          Diagonal(vcat(fill(1.0, 10), fill(4.0, 3)))

    # The eltype of the diagonal matches the eltype of the samples
    @test compute_diagonal(VarianceDiagonal(), sample_collection) isa
          Diagonal{Float32}
    sample_collection64 = SampleBuilder.build_samples(vars_matrix; FT = Float64)
    @test compute_diagonal(VarianceDiagonal(), sample_collection64) isa
          Diagonal{Float64}
end

@testset "QuantileDiagonal" begin
    inner_term = ScalarDiagonal(1.0)

    # Scalar quantile is wrapped into a vector
    term = QuantileDiagonal(0.05, inner_term)
    @test term isa AbstractDiagonalTerm
    @test term.quantiles == [0.05]
    @test term.diag_term === inner_term

    # Vector of quantiles; 1.0 is included in (0, 1]
    term = QuantileDiagonal([0.05, 1.0], inner_term)
    @test term.quantiles == [0.05, 1.0]

    # Quantiles outside (0, 1] are not allowed
    @test_throws "Quantiles must be in (0, 1]" QuantileDiagonal(0.0, inner_term)
    @test_throws "Quantiles must be in (0, 1]" QuantileDiagonal(
        -0.05,
        inner_term,
    )
    @test_throws "Quantiles must be in (0, 1]" QuantileDiagonal(1.5, inner_term)
    @test_throws "Quantiles must be in (0, 1]" QuantileDiagonal(
        [0.05, 1.5],
        inner_term,
    )

    # Second argument must be a diagonal term
    @test_throws MethodError QuantileDiagonal(0.05, 1.0)

    # Empty vectors are not accepted
    @test_throws "should not be empty" QuantileDiagonal(Float64[], inner_term)

    # Two variables with flattened lengths of 10 and 3, and a single sample
    # with data 1 to n, so the inner ModelErrorScaleDiagonal diagonal is
    # (1:10).^2 for pr and (1:3).^2 for ta
    pr_var =
        TemplateVar() |>
        add_dim("time", collect(1.0:10.0), units = "s") |>
        add_attribs(short_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    ta_var =
        TemplateVar() |>
        add_dim("time", collect(1.0:3.0), units = "s") |>
        add_attribs(short_name = "ta", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    sample_collection =
        SampleBuilder.build_samples(reshape([pr_var, ta_var], 2, 1))
    mes_term = ModelErrorScaleDiagonal(1.0)

    # The quantile of each block fills that block; the median of (1:10).^2 is
    # (25 + 36) / 2 and the median of (1:3).^2 is 4
    @test compute_diagonal(
        QuantileDiagonal(0.5, mes_term),
        sample_collection,
    ) == Diagonal(vcat(fill(30.5, 10), fill(4.0, 3)))

    # One quantile per variable; the 1.0 quantile is the maximum of the block
    @test compute_diagonal(
        QuantileDiagonal([1.0, 0.5], mes_term),
        sample_collection,
    ) == Diagonal(vcat(fill(100.0, 10), fill(4.0, 3)))

    # Number of quantiles must be one or match the number of variables
    @test_throws "The number of quantiles" compute_diagonal(
        QuantileDiagonal([0.5, 0.5, 0.5], mes_term),
        sample_collection,
    )

    # Blocks are too small for a meaningful quantile (1 / 0.05 = 20 entries
    # needed)
    @test_throws "Insufficient samples" compute_diagonal(
        QuantileDiagonal(0.05, mes_term),
        sample_collection,
    )

    # Quantile of a zero diagonal block is not allowed, and the error names
    # the variable of the offending block (only the ta block is zero)
    @test_throws r"Zero found.*variable \(ta\)" compute_diagonal(
        QuantileDiagonal(0.5, ScalarDiagonal([1.0, 0.0])),
        sample_collection,
    )

    # The eltype of the diagonal matches the eltype of the samples
    @test compute_diagonal(
        QuantileDiagonal(0.5, mes_term),
        sample_collection,
    ) isa Diagonal{Float32}

    # Three samples where sample j has data j .* (1:n), so the mean across
    # samples is 2 .* (1:n) and the inner ModelErrorScaleDiagonal diagonal is
    # 4 .* (1:10).^2 for pr and 4 .* (1:3).^2 for ta
    function make_scaled_var(short_name, time_len, factor)
        var =
            TemplateVar() |>
            add_dim("time", collect(1.0:time_len), units = "s") |>
            add_attribs(short_name = short_name, start_date = "2008-1-1") |>
            one_to_n_data(collected = true) |>
            initialize
        var.data .*= factor
        return var
    end
    multi_sample_collection = SampleBuilder.build_samples(
        [
            make_scaled_var("pr", 10, 1.0) make_scaled_var("pr", 10, 2.0) make_scaled_var(
                "pr",
                10,
                3.0,
            )
            make_scaled_var("ta", 3, 1.0) make_scaled_var("ta", 3, 2.0) make_scaled_var("ta", 3, 3.0)
        ],
    )

    # The medians are 4 * 30.5 for pr and 4 * 4 for ta
    @test compute_diagonal(
        QuantileDiagonal(0.5, mes_term),
        multi_sample_collection,
    ) == Diagonal(vcat(fill(4.0 * 30.5, 10), fill(4.0 * 4.0, 3)))
end

@testset "SumDiagonal" begin
    term1 = ScalarDiagonal(1.0)
    term2 = ModelErrorScaleDiagonal(0.05)
    term3 = ScalarDiagonal(2.0)

    # Adding two terms makes a SumDiagonal
    sum_term = term1 + term2
    @test sum_term isa SumDiagonal
    @test sum_term isa AbstractDiagonalTerm
    @test sum_term.diagonal_terms === (term1, term2)

    # Nested sums are flattened (term + sum, sum + term, sum + sum)
    @test (term1 + term2 + term3).diagonal_terms === (term1, term2, term3)
    @test (term3 + (term1 + term2)).diagonal_terms === (term3, term1, term2)
    @test ((term1 + term2) + (term3 + term1)).diagonal_terms ===
          (term1, term2, term3, term1)

    # Sums can be nested in other terms
    @test QuantileDiagonal(0.05, sum_term).diag_term === sum_term

    # Only tuples of diagonal terms are accepted
    @test_throws MethodError SumDiagonal((1.0, 2.0))

    # Empty tuples are not accepted
    @test_throws "should not be empty" SumDiagonal(())

    # Two variables with flattened lengths of 10 and 3, and a single sample
    # with data 1 to n
    pr_var =
        TemplateVar() |>
        add_dim("time", collect(1.0:10.0), units = "s") |>
        add_attribs(short_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    ta_var =
        TemplateVar() |>
        add_dim("time", collect(1.0:3.0), units = "s") |>
        add_attribs(short_name = "ta", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    sample_collection =
        SampleBuilder.build_samples(reshape([pr_var, ta_var], 2, 1))

    # The diagonal of a sum is the elementwise sum of the diagonals
    @test compute_diagonal(
        ScalarDiagonal(2.0) + ScalarDiagonal([1.0, 3.0]),
        sample_collection,
    ) == Diagonal(vcat(fill(3.0, 10), fill(5.0, 3)))

    # Sum of different term types; the ModelErrorScaleDiagonal diagonal is
    # (1:10).^2 for pr and (1:3).^2 for ta
    @test compute_diagonal(
        ScalarDiagonal(1.0) + ModelErrorScaleDiagonal(1.0),
        sample_collection,
    ) == Diagonal(vcat((1.0:10.0) .^ 2 .+ 1.0, (1.0:3.0) .^ 2 .+ 1.0))

    # A flattened three-term sum
    @test compute_diagonal(
        ScalarDiagonal(1.0) + ScalarDiagonal(2.0) + ScalarDiagonal(3.0),
        sample_collection,
    ) == Diagonal(fill(6.0, 13))

    # Model error scale + quantile regularization
    model_error_scale_term = ModelErrorScaleDiagonal(1.0)
    @test compute_diagonal(
        model_error_scale_term + QuantileDiagonal(0.5, model_error_scale_term),
        sample_collection,
    ) == Diagonal(vcat((1.0:10.0) .^ 2 .+ 30.5, (1.0:3.0) .^ 2 .+ 4.0))

    # Errors from terms inside the sum propagate (three scalars for two
    # variables)
    @test_throws "The number of scalars" compute_diagonal(
        ScalarDiagonal(1.0) + ScalarDiagonal([1.0, 2.0, 3.0]),
        sample_collection,
    )

    # The eltype of the diagonal matches the eltype of the samples, even for
    # sums of terms with mixed eltypes
    @test compute_diagonal(
        ScalarDiagonal(1.0f0) + ModelErrorScaleDiagonal(0.5),
        sample_collection,
    ) isa Diagonal{Float32}

    function make_scaled_var(short_name, time_len, factor)
        var =
            TemplateVar() |>
            add_dim("time", collect(1.0:time_len), units = "s") |>
            add_attribs(short_name = short_name, start_date = "2008-1-1") |>
            one_to_n_data(collected = true) |>
            initialize
        var.data .*= factor
        return var
    end
    # Three samples where sample j has data j .* (1:n), so the mean across
    # samples is 2 .* (1:n) and the ModelErrorScaleDiagonal diagonal is
    # 4 .* (1:n).^2
    multi_sample_collection = SampleBuilder.build_samples(
        [
            make_scaled_var("pr", 10, 1.0) make_scaled_var("pr", 10, 2.0) make_scaled_var(
                "pr",
                10,
                3.0,
            )
            make_scaled_var("ta", 3, 1.0) make_scaled_var("ta", 3, 2.0) make_scaled_var("ta", 3, 3.0)
        ],
    )
    @test compute_diagonal(
        ScalarDiagonal(1.0) + ModelErrorScaleDiagonal(1.0),
        multi_sample_collection,
    ) == Diagonal(
        vcat(4.0 .* (1.0:10.0) .^ 2 .+ 1.0, 4.0 .* (1.0:3.0) .^ 2 .+ 1.0),
    )
end

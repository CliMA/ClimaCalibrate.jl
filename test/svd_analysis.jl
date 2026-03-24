using Test
using LinearAlgebra
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate as CAL
using LinearMaps

@testset "create_compact_linear_map" begin
    @testset "SVD input" begin
        # Use a square symmetric matrix
        A = randn(8, 6)
        cov_A = A * A'  # 8×8 symmetric PSD
        svd_A = svd(cov_A)
        lmap = CAL._create_compact_linear_map(svd_A)
        @test lmap isa LinearMap
        @test size(lmap) == (8, 8)
        x = randn(8)
        expected = svd_A.U * (svd_A.S .* (svd_A.Vt * x))
        @test lmap * x ≈ expected
    end

    @testset "Diagonal input" begin
        d = rand(6) .+ 0.1
        diag_mat = Diagonal(d)
        lmap = CAL._create_compact_linear_map(diag_mat)
        @test lmap isa LinearMap
        @test size(lmap) == (6, 6)
        x = randn(6)
        @test lmap * x ≈ d .* x
    end

    @testset "EKP.SVDplusD input" begin
        samples = randn(8, 20)
        svd_cov = EKP.tsvd_cov_from_samples(samples)
        d = 0.01 * ones(8)
        svdplusd = EKP.SVDplusD(svd_cov, Diagonal(d))
        lmap = CAL._create_compact_linear_map(svdplusd)
        @test lmap isa LinearMap
        @test size(lmap) == (8, 8)
        # Spot-check: applying Γ = U*S*Vt + D to a vector
        x = randn(8)
        svda = svdplusd.svd_cov
        diaga = svdplusd.diag_cov.diag
        expected = svda.U * (svda.S .* (svda.Vt * x)) + diaga .* x
        @test lmap * x ≈ expected
    end

    @testset "UniformScaling throws" begin
        @test_throws ArgumentError CAL._create_compact_linear_map(I)
    end

    @testset "Unsupported type throws" begin
        @test_throws ArgumentError CAL._create_compact_linear_map(randn(5, 5))
    end

    @testset "Vector of SVDs (block-diagonal)" begin
        A1 = randn(5, 3)
        cov1 = A1 * A1'
        A2 = randn(4, 2)
        cov2 = A2 * A2'
        lmap = CAL._create_compact_linear_map([svd(cov1), svd(cov2)])
        @test size(lmap) == (9, 9)
        # Each block acts independently
        x = randn(9)
        s1, s2 = svd(cov1), svd(cov2)
        expected = vcat(
            s1.U * (s1.S .* (s1.Vt * x[1:5])),
            s2.U * (s2.S .* (s2.Vt * x[6:9])),
        )
        @test lmap * x ≈ expected
    end
end

@testset "compute_structured_energy" begin
    @testset "zero projections → zero energy" begin
        @test CAL.compute_structured_energy(zeros(3, 2)) == 0.0
    end

    @testset "consistent with manual calculation" begin
        proj = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3 eigvecs × 2 vars
        z = vec(sum(proj, dims = 2))           # [3.0, 7.0, 11.0]
        @test CAL.compute_structured_energy(proj) ≈ sum(z .^ 2) / length(z)
    end

    @testset "unit noise → energy ≈ 1 in expectation" begin
        # z[i] = sum_v(proj[i,v]); if proj[i,v] ~ N(0,1) with many draws,
        # energy = sum(z^2)/n_eig concentrates around n_vars
        proj = ones(5, 1)  # z[i] = 1 for all i → energy = 1
        @test CAL.compute_structured_energy(proj) ≈ 1.0
    end
end

@testset "compute_structured_energy_by_variable" begin
    @testset "returns one value per variable" begin
        proj = randn(3, 4)
        @test length(CAL.compute_structured_energy_by_variable(proj)) == 4
    end

    @testset "consistent with manual calculation" begin
        proj = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3 eigvecs × 2 vars
        result = CAL.compute_structured_energy_by_variable(proj)
        @test result[1] ≈ (1^2 + 3^2 + 5^2) / 3
        @test result[2] ≈ (2^2 + 4^2 + 6^2) / 3
    end

    @testset "consistent with global energy when one variable" begin
        proj = randn(4, 1)
        @test only(CAL.compute_structured_energy_by_variable(proj)) ≈
              CAL.compute_structured_energy(proj)
    end
end

@testset "compute_normalized_projections" begin
    n_obs, n_eig = 10, 3
    Q, _ = qr(randn(n_obs, n_obs))
    eigvecs = Matrix(Q)[:, 1:n_eig]
    eigvalues = [4.0, 2.0, 1.0]
    diff = randn(n_obs)

    # Mock metadata_vec using named tuples — only the `range` field is accessed
    metadata_vec = [1:6, 7:10]
    proj = CAL.compute_normalized_projections(
        diff,
        eigvecs,
        eigvalues,
        metadata_vec,
    )

    @test size(proj) == (n_eig, 2)

    # Compare against manual calculation
    @test proj[1, 1] ≈ dot(eigvecs[1:6, 1], diff[1:6]) / sqrt(4.0)
    @test proj[2, 1] ≈ dot(eigvecs[1:6, 2], diff[1:6]) / sqrt(2.0)
    @test proj[1, 2] ≈ dot(eigvecs[7:10, 1], diff[7:10]) / sqrt(4.0)
    @test proj[3, 2] ≈ dot(eigvecs[7:10, 3], diff[7:10]) / sqrt(1.0)
end

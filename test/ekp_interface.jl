import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
import ClimaCalibrate as CAL
import ClimaParams as CP
import LinearAlgebra: I
using Test
import Random

rng_seed = 1234
Random.seed!(rng_seed)
rng_ekp = Random.MersenneTwister(rng_seed)

FT = Float64
prior_path = joinpath(pkgdir(CAL), "test", "test_case_inputs", "prior.toml")
param_names = ["one", "two"]

prior = CAL.get_prior(prior_path)
noise = 0.1 * I
observations = zeros(Float64, 1)
ensemble_size = 10

"""
    member_parameters(output_dir, iteration, member)

Read back the parameters written for an ensemble member.
"""
function member_parameters(output_dir, iteration, member)
    override_file = CAL.parameter_path(output_dir, iteration, member)
    td = CP.create_toml_dict(FT; override_file)
    return CP.get_parameter_values(td, param_names)
end

@testset "Test loading latest EKP struct" begin
    output_dir = mktempdir(cleanup = true)

    # Test loading from directory with no completed iterations
    empty_dir = joinpath(output_dir, "empty")
    mkpath(empty_dir)
    @test isnothing(CAL.load_latest_ekp(empty_dir))

    ekp = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        observations,
        noise,
        EKP.Inversion();
        scheduler = EKP.MutableScheduler(2),
        accelerator = EKP.NesterovAccelerator(),
        rng = rng_ekp,
    )
    # Iterations are 1-indexed, which is what `initialize` writes
    CAL.initialize(ekp, prior, output_dir)

    latest_ekp = CAL.load_latest_ekp(output_dir)
    @test !isnothing(latest_ekp)
    @test latest_ekp isa EKP.EnsembleKalmanProcess
    for prop in propertynames(latest_ekp)
        prop in [:u, :accelerator, :localizer] && continue
        @test getproperty(latest_ekp, prop) == getproperty(ekp, prop)
    end
    @test latest_ekp.u[1].data == ekp.u[1].data

    # A later iteration takes precedence over an earlier one. The two are
    # distinguishable only if what is saved to iteration 2 differs
    later_ekp = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        observations,
        noise,
        EKP.Inversion();
        rng = rng_ekp,
    )
    CAL.save_eki_and_parameters(later_ekp, output_dir, 2, prior)
    @test CAL.load_latest_ekp(output_dir).u[1].data == later_ekp.u[1].data
    @test CAL.load_latest_ekp(output_dir).u[1].data != ekp.u[1].data
    @test CAL.last_completed_iteration(output_dir) == 0
end

@testset "Initialized parameter values" begin
    output_dir = mktempdir(cleanup = true)
    ekp = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        observations,
        noise,
        EKP.Inversion(),
        EKP.default_options_dict(EKP.Inversion());
        rng = rng_ekp,
    )
    CAL.initialize(ekp, prior, output_dir)

    # The parameters on disk are the constrained initial ensemble. Asserting
    # that keeps the test independent of how much randomness EKP consumes
    # while constructing the object
    ϕ = EKP.get_ϕ_final(prior, ekp)
    names = EKP.get_name(prior)
    for member in 1:ensemble_size
        params = member_parameters(output_dir, 1, member)
        for (i, name) in enumerate(names)
            @test getproperty(params, Symbol(name)) ≈ ϕ[i, member]
        end
    end
end

@testset "`initialize` does not clobber an existing calibration" begin
    output_dir = mktempdir(cleanup = true)
    ekp = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        observations,
        noise,
        EKP.Inversion(),
    )
    CAL.initialize(ekp, prior, output_dir)
    original = member_parameters(output_dir, 1, 1)

    # Restarting typically rebuilds the EKP object from scratch, which draws a
    # new initial ensemble. The parameters that the completed forward models
    # were run with have to survive that
    restarted_ekp = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        observations,
        noise,
        EKP.Inversion(),
    )
    returned = CAL.initialize(restarted_ekp, prior, output_dir)

    @test member_parameters(output_dir, 1, 1) == original
    @test EKP.get_u_final(returned) == EKP.get_u_final(ekp)
    @test EKP.get_u_final(returned) != EKP.get_u_final(restarted_ekp)

    # A restart with a different ensemble size is a mistake, not something to
    # silently accommodate
    mismatched = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size + 1),
        observations,
        noise,
        EKP.Inversion(),
    )
    @test_throws ErrorException CAL.initialize(mismatched, prior, output_dir)

    # So is a restart against observations the user has edited since: the
    # stored process is the one the completed members were run with
    other_observations = observations .+ 1
    edited = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        other_observations,
        noise,
        EKP.Inversion(),
    )
    @test_throws ErrorException CAL.initialize(edited, prior, output_dir)

    wider_noise = 2 .* noise
    renoised = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        observations,
        wider_noise,
        EKP.Inversion(),
    )
    # A covariance of the same shape is not compared entry by entry, so this
    # one is accepted
    @test CAL.initialize(renoised, prior, output_dir) isa
          EKP.EnsembleKalmanProcess

    unscented = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        observations,
        noise,
        EKP.TransformInversion(),
    )
    @test_throws ErrorException CAL.initialize(unscented, prior, output_dir)
end

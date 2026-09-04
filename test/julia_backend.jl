using Test

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaParams as CP
import Random
import Statistics

import ClimaCalibrate as CAL
import JLD2

# Experiment Info
output_file = "model_output.jld2"
prior = constrained_gaussian("test_param", 10, 5, 0, Inf)
n_iterations = 1
ensemble_size = 20
observations = [20.0]
noise = [0.01;;]
output_dir = mktempdir()

struct DummyModelInterface <: CAL.AbstractModelInterface
    output_dir::String
end

"""
    run_calibration(output_dir)

Run the whole calibration in `output_dir` and return the resulting EKP object.

The `rng` is passed to `EnsembleKalmanProcess` as well as to
`construct_initial_ensemble`. Without it the ensemble update draws from the
global RNG, and the run is only as reproducible as the state of that RNG.
"""
function run_calibration(output_dir)
    rng = Random.MersenneTwister(1234)
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, ensemble_size)
    eki = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        observations,
        noise,
        EKP.Inversion();
        rng,
    )
    return CAL.calibrate(
        CAL.JuliaBackend(),
        eki,
        DummyModelInterface(output_dir),
        n_iterations,
        prior,
        output_dir,
    )
end

# Model interface
# This "model" just samples parameters and returns them, we are checking that
# the results are reproducible
function CAL.forward_model(interface::DummyModelInterface, iteration, member)
    (; output_dir) = interface
    member_path = CAL.path_to_ensemble_member(output_dir, iteration, member)
    param_path = CAL.parameter_path(output_dir, iteration, member)
    toml_dict = CP.create_toml_dict(Float64; override_file = param_path)
    (; test_param) = CP.get_parameter_values(toml_dict, "test_param")
    JLD2.save_object(joinpath(member_path, output_file), test_param)
end

function CAL.observation_map(interface::DummyModelInterface, iteration)
    (; output_dir) = interface
    dims = 1
    G_ensemble = Array{Float64}(undef, dims..., ensemble_size)
    for m in 1:ensemble_size
        member_path =
            TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
        output = JLD2.load_object(joinpath(member_path, output_file))
        G_ensemble[:, m] .= output
    end
    return G_ensemble
end

function CAL.analyze_iteration(
    ::DummyModelInterface,
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration,
)
    @info "Analyzing iteration"
    @info "Iteration $iteration"
    @info "Current mean constrained parameter: $(EKP.get_ϕ_mean_final(prior, ekp))"
    @info "g_ensemble: $g_ensemble"
    @info "output_dir: $output_dir"
    return nothing
end

ekp = run_calibration(output_dir)

@testset "Test end-to-end calibration" begin
    parameter_values =
        [EKP.get_ϕ_mean(prior, ekp, it) for it in 1:(n_iterations + 1)]
    @test length(parameter_values) == n_iterations + 1

    # This "model" returns its own parameter, so the ensemble update should
    # move the mean parameter towards the observation. Asserting that keeps the
    # test from going stale when EKP changes how it consumes randomness
    initial_error = abs(parameter_values[1][1] - only(observations))
    final_error = abs(parameter_values[end][1] - only(observations))
    @test final_error < initial_error

    # The spread should contract as the ensemble learns
    ϕ = [EKP.get_ϕ(prior, ekp, it) for it in 1:(n_iterations + 1)]
    @test Statistics.var(vec(ϕ[end])) < Statistics.var(vec(ϕ[1]))
end

@testset "Test reproducibility" begin
    # The same inputs give the same answer
    rerun = run_calibration(mktempdir())
    @test EKP.get_ϕ_mean_final(prior, rerun) == EKP.get_ϕ_mean_final(prior, ekp)
    @test EKP.get_g_mean_final(rerun) == EKP.get_g_mean_final(ekp)
end

@testset "Rerunning a finished calibration" begin
    # The loop has nothing to run, and the caller gets the process the run
    # ended with. Returning the stored first iteration would look like a
    # calibration that never updated
    rerun = run_calibration(output_dir)
    @test EKP.get_ϕ_mean_final(prior, rerun) == EKP.get_ϕ_mean_final(prior, ekp)
    @test EKP.get_N_iterations(rerun) == EKP.get_N_iterations(ekp)
    @test CAL.last_completed_iteration(output_dir) == n_iterations
end

@testset "A terminated calibration stays terminated" begin
    terminated_dir = mktempdir()
    rng = Random.MersenneTwister(1234)
    # `DataMisfitController` stops the calibration once the accumulated misfit
    # reaches its threshold, and `update_ensemble!` still writes the next
    # iteration's parameters
    eki = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng, prior, ensemble_size),
        observations,
        noise,
        EKP.Inversion();
        rng,
        scheduler = EKP.DataMisfitController(terminate_at = 1),
    )
    ekp_terminated = CAL.calibrate(
        CAL.JuliaBackend(),
        eki,
        DummyModelInterface(terminated_dir),
        20,
        prior,
        terminated_dir,
    )
    stopped_at = CAL.terminated_iteration(terminated_dir)
    @test !isnothing(stopped_at)
    @test stopped_at < 20

    # A rerun would otherwise run the remaining iterations on the parameters
    # the terminated iteration left behind
    resumed = CAL.calibrate(
        CAL.JuliaBackend(),
        eki,
        DummyModelInterface(terminated_dir),
        20,
        prior,
        terminated_dir,
    )
    @test CAL.last_completed_iteration(terminated_dir) == stopped_at
    @test EKP.get_ϕ_mean_final(prior, resumed) ==
          EKP.get_ϕ_mean_final(prior, ekp_terminated)
end

@testset "Calibration output layout" begin
    @test CAL.last_completed_iteration(output_dir) == n_iterations
    @test isfile(CAL.ekp_path(output_dir, n_iterations + 1))
    @test isfile(joinpath(CAL.path_to_iteration(output_dir, 1), "prior.jld2"))
    for m in 1:ensemble_size
        @test isfile(CAL.parameter_path(output_dir, 1, m))
    end
end

@testset "JuliaBackend checkpoints its members" begin
    # A member that recorded a completed forward model is not rerun. Without
    # these checkpoints an interrupted iteration would rerun the whole ensemble,
    # and the output directory could not be resumed by another backend
    for m in 1:ensemble_size
        @test isfile(CAL.checkpoint_path(output_dir, 1, m))
        @test CAL.model_completed(output_dir, 1, m)
    end

    runs = Ref(0)
    struct CountingInterface <: CAL.AbstractModelInterface end
    CAL.forward_model(::CountingInterface, iteration, member) = (runs[] += 1)

    # Every member of iteration 1 is checkpointed, so nothing should run
    CAL.Calibration.run_iteration(
        CAL.JuliaBackend(),
        CountingInterface(),
        1,
        ensemble_size,
        output_dir,
    )
    @test runs[] == 0
end

@testset "Failure rate halts the calibration" begin
    struct FailingInterface <: CAL.AbstractModelInterface end
    CAL.forward_model(::FailingInterface, iteration, member) =
        error("member $member failed")

    fresh = mktempdir()

    # Above the failure-rate threshold the run stops
    @test_throws ErrorException CAL.Calibration.run_iteration(
        CAL.JuliaBackend(),
        FailingInterface(),
        1,
        4,
        fresh,
    )

    # A backend that tolerates any failure rate keeps going
    tolerant = CAL.JuliaBackend(failure_rate = 1.0)
    @test isnothing(
        CAL.Calibration.run_iteration(
            tolerant,
            FailingInterface(),
            1,
            4,
            mktempdir(),
        ),
    )
end

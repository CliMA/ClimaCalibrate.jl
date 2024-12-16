# Tests for SurfaceFluxes example calibration on HPC, used in buildkite testing
# To run, open the REPL: julia --project=experiments/surface_fluxes_perfect_model test/slurm_workers.jl

using ClimaCalibrate
using Distributed
using Test
import EnsembleKalmanProcesses: get_ϕ, get_g_mean_final
import Statistics: var

function test_sf_calibration_output(eki, prior, observation)
    @testset "End to end test using file config (surface fluxes perfect model)" begin
        params = get_ϕ(prior, eki)
        spread = map(var, params)
        
        # Spread should be heavily decreased as particles have converged
        @test last(spread) / first(spread) < 0.15

        forward_model_output = get_g_mean_final(eki)
        @show forward_model_output
        @test all(
            isapprox.(forward_model_output, observation; rtol = 1e-2),
        )
    end
end

experiment_dir = dirname(Base.active_project())
addprocs(ClimaCalibrate.SlurmManager(10))
include(joinpath(experiment_dir, "generate_data.jl"))

@everywhere begin
    using ClimaCalibrate
    experiment_dir = dirname(Base.active_project())
    output_dir = joinpath("output", "surface_fluxes_perfect_model")
    prior = get_prior(joinpath(experiment_dir, "prior.toml"))
    ensemble_size = 20
    n_iterations = 6
end

@everywhere begin
    include(joinpath(experiment_dir, "observation_map.jl"))
    ustar = JLD2.load_object(
        joinpath(experiment_dir, "data", "synthetic_ustar_array_noisy.jld2"),
    )
    (; observation, variance) =
        process_member_data(ustar; output_variance = true)

    model_interface = joinpath(experiment_dir, "model_interface.jl")
    include(model_interface)
end

eki = worker_calibrate(
    ensemble_size,
    n_iterations,
    observation,
    variance,
    prior,
    output_dir,
)

test_sf_calibration_output(eki, prior, observation)

include(joinpath(experiment_dir, "postprocessing.jl"))

# Slurm Worker Unit Tests
@testset "Slurm Worker Unit Tests" begin
    out_file = "my_slurm_job.out"
    p = addprocs(ClimaCalibrate.SlurmManager(1); o = out_file)
    @test nprocs() == 2
    @test workers() == p
    @test fetch(@spawnat :any myid()) == p[1]
    @test remotecall_fetch(+, p[1], 1, 1) == 2
    rmprocs(p)
    @test nprocs() == 1
    @test workers() == [1]

    # Check output file creation
    @test isfile(out_file)
    rm(out_file)

    @test_throws TaskFailedException p = addprocs(
        ClimaCalibrate.SlurmManager(1);
        o = out_file,
        output = out_file,
    )
end

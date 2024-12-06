# Tests for SurfaceFluxes example calibration on HPC, used in buildkite testing
# To run, open the REPL: julia --project=experiments/surface_fluxes_perfect_model test/hpc_backend_e2e.jl

using ClimaCalibrate
using Distributed
using Test
import EnsembleKalmanProcesses: get_ϕ_mean_final, get_g_mean_final

function test_sf_calibration_output(eki, prior)
    @testset "End to end test using file config (surface fluxes perfect model)" begin
        parameter_values = get_ϕ_mean_final(prior, eki)
        test_parameter_values = [4.778584250117946, 3.7295665619234697]
        @test all(
            isapprox.(parameter_values, test_parameter_values; rtol = 1e-3),
        )

        forward_model_output = get_g_mean_final(eki)
        test_model_output = [0.05228473730385304]
        @test all(
            isapprox.(forward_model_output, test_model_output; rtol = 1e-3),
        )
    end
end

experiment_dir = dirname(Base.active_project())
addprocs(
    ClimaCalibrate.SlurmManager(10);
    exeflags = "--project=$(dirname(Base.active_project()))",
)
include(joinpath(experiment_dir, "generate_data.jl"))

@everywhere begin
    using ClimaCalibrate
    experiment_dir = dirname(Base.active_project())
    output_dir = joinpath("output", "surface_fluxes_perfect_model")
    prior = get_prior(joinpath(experiment_dir, "prior.toml"))
    ensemble_size = 10
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

test_sf_calibration_output(eki, prior)

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

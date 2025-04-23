using Test
using ClimaCalibrate
import EnsembleKalmanProcesses as EKP

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)

experiment_config = ExperimentConfig(
    n_iterations,
    ensemble_size,
    observation,
    variance,
    prior,
    output_dir,
)
hpc_kwargs = kwargs(time = 5, ntasks = 1, cpus_per_task = 1)

backend = get_backend()
@assert backend <: HPCBackend
hpc_kwargs = kwargs(time = 5, ntasks = 1, cpus_per_task = 1)
if get_backend() == DerechoBackend
    hpc_kwargs[:queue] = "preempt"
    hpc_kwargs[:gpus_per_task] = 1
end

model_interface, io = mktemp()
write(io, "import ClimaCalibrate\n")
write(io, "ClimaCalibrate.forward_model(iter, member) = member > 2 && exit()")
close(io)


eki = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, ensemble_size),
    observation,
    variance,
    EKP.Inversion(),
    verbose = true,
)

ClimaCalibrate.initialize(eki, prior, output_dir)

ClimaCalibrate.run_hpc_iteration(
    backend,
    eki,
    0,
    ensemble_size,
    output_dir,
    experiment_dir,
    model_interface,
    ClimaCalibrate.module_load_string(backend),
    prior;
    hpc_kwargs,
)
@testset "Test model checkpoints with interruptions" begin
    for m in 1:ensemble_size
        checkpoint_file = joinpath(
            ClimaCalibrate.path_to_ensemble_member(output_dir, 0, m),
            "checkpoint.txt",
        )
        expected = m <= 2 ? "completed" : "started"
        @test read(checkpoint_file, String) == expected
    end
end

model_interface, io = mktemp()
write(io, "import ClimaCalibrate\n")
write(io, "ClimaCalibrate.forward_model(iter, member) = return")
close(io)

ClimaCalibrate.run_hpc_iteration(
    backend,
    eki,
    0,
    ensemble_size,
    output_dir,
    experiment_dir,
    model_interface,
    ClimaCalibrate.module_load_string(backend),
    prior;
    hpc_kwargs,
)

@testset "Test model checkpoints for completion" begin
    for m in 1:ensemble_size
        checkpoint_file = joinpath(
            ClimaCalibrate.path_to_ensemble_member(output_dir, 0, m),
            "checkpoint.txt",
        )
        @test read(checkpoint_file, String) == "completed"
    end
end


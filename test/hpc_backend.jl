using ClimaCalibrate

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)
backend = get_backend()
@assert backend <: HPCBackend
hpc_kwargs = kwargs(time = 5, ntasks = 1, cpus_per_task = 1)
if backend == DerechoBackend
    hpc_kwargs[:queue] = "preempt"
    hpc_kwargs[:gpus_per_task] = 1
end

original_model_interface = model_interface
interruption_model_interface, io = mktemp(@__DIR__)
model_interface_str = """
import ClimaCalibrate
ClimaCalibrate.forward_model(iter, member) = member == 1 && exit()
"""
write(io, model_interface_str)
close(io)

eki = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, ensemble_size),
    observation,
    variance,
    EKP.Inversion(),
    verbose = true,
)

ClimaCalibrate.initialize(eki, prior, output_dir)

backend = backend(
    hpc_kwargs = hpc_kwargs,
    experiment_dir = experiment_dir,
    model_interface = interruption_model_interface,
)
ClimaCalibrate.run_hpc_iteration(
    backend,
    0,
    ensemble_size,
    output_dir,
    ClimaCalibrate.module_load_string(backend),
)

@testset "Test model checkpoints with interruptions" begin
    for m in 1:ensemble_size
        @test m == 1 ? ClimaCalibrate.model_started(output_dir, 0, m) :
              ClimaCalibrate.model_completed(output_dir, 0, m)
        rm(ClimaCalibrate.checkpoint_path(output_dir, 0, m))
    end
end

eki = calibrate(
    ensemble_size,
    n_iterations,
    observation,
    variance,
    prior,
    output_dir;
    backend_kwargs = (; model_interface, hpc_kwargs),
    verbose = true,
    localization_method = EKP.Localizers.NoLocalization(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DefaultScheduler(),
)

@test ClimaCalibrate.last_completed_iteration(output_dir) == n_iterations - 1

@testset "Test model checkpoints for completion" begin
    for m in 1:ensemble_size
        @test ClimaCalibrate.model_completed.(output_dir, 0, m)
    end
end

test_sf_calibration_output(eki, prior, observation)

# Remove previous output - this is not necessary but safe for tests
rm(output_dir, recursive = true)
# Pure Julia calibration, this should run anywhere
julia_eki = calibrate(
    JuliaBackend(),
    ensemble_size,
    n_iterations,
    observation,
    variance,
    prior,
    output_dir;
    localization_method = EKP.Localizers.NoLocalization(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DefaultScheduler(),
)
test_sf_calibration_output(julia_eki, prior, observation)

compare_g_ensemble(eki, julia_eki)

theta_star_vec =
    (; coefficient_a_m_businger = 4.7, coefficient_a_h_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger", "coefficient_a_h_businger"],
)

g_vs_iter_plot(eki)

import ClimaCalibrate
import Random

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)
backend = ClimaCalibrate.get_backend()
@assert backend <: ClimaCalibrate.HPCBackend
hpc_kwargs = Dict{Symbol, Any}(:time => 5, :ntasks => 1, :cpus_per_task => 1)
if backend == ClimaCalibrate.DerechoBackend
    hpc_kwargs[:queue] = "preempt"
end

original_model_interface = model_interface
interruption_model_interface, io = mktemp(@__DIR__)
model_interface_str = """
import ClimaCalibrate
ClimaCalibrate.forward_model(iter, member) = member == 1 && exit()
"""
write(io, model_interface_str)
close(io)

"""
    make_ekp(
    rng_seed,
    prior,
    ensemble_size,
    observation,
    variance;
    ekp_kwargs...,
)

A convenience constructor for making a `EnsembleKalmanProcess` object.
"""
function make_ekp(
    rng_seed,
    prior,
    ensemble_size,
    observation,
    variance;
    ekp_kwargs...,
)
    Random.seed!(rng_seed)
    rng_ekp = Random.MersenneTwister(rng_seed)
    eki = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng_ekp, prior, ensemble_size),
        observation,
        variance,
        EKP.Inversion();
        rng = rng_ekp,
        ekp_kwargs...,
    )
    return eki
end

rng_seed = 1234
eki = make_ekp(
    rng_seed,
    prior,
    ensemble_size,
    observation,
    variance;
    verbose = true,
)

ClimaCalibrate.initialize(eki, prior, output_dir)

backend = backend(hpc_kwargs = hpc_kwargs)
experiment_dir = dirname(Base.active_project())
ClimaCalibrate.Calibration.run_iteration(
    backend,
    1,
    ensemble_size,
    output_dir,
    interruption_model_interface,
    experiment_dir,
    "",
)

@testset "Test model checkpoints with interruptions" begin
    for m in 1:ensemble_size
        @test m == 1 ? ClimaCalibrate.model_started(output_dir, 1, m) :
              ClimaCalibrate.model_completed(output_dir, 1, m)
        rm(ClimaCalibrate.checkpoint_path(output_dir, 1, m))
    end
end

ekp = make_ekp(
    rng_seed,
    prior,
    ensemble_size,
    observation,
    variance;
    localization_method = EKP.Localizers.NoLocalization(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DefaultScheduler(),
)
backend = ClimaCalibrate.get_backend()
eki = ClimaCalibrate.Calibration.calibrate(
    backend(; hpc_kwargs),
    ekp,
    n_iterations,
    prior,
    output_dir,
    model_interface;
    experiment_dir,
)

@test ClimaCalibrate.last_completed_iteration(output_dir) == n_iterations

@testset "Test model checkpoints for completion" begin
    for m in 1:ensemble_size
        @test ClimaCalibrate.model_completed.(output_dir, 1, m)
    end
end

test_sf_calibration_output(eki, prior, observation)

# Remove previous output - this is not necessary but safe for tests
rm(output_dir, recursive = true)

# Pure Julia calibration, this should run anywhere
ekp = make_ekp(
    rng_seed,
    prior,
    ensemble_size,
    observation,
    variance;
    localization_method = EKP.Localizers.NoLocalization(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DefaultScheduler(),
)
julia_eki = ClimaCalibrate.Calibration.calibrate(
    JuliaBackend(),
    ekp,
    n_iterations,
    prior,
    output_dir,
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
    output_dir,
)

g_vs_iter_plot(eki, output_dir)

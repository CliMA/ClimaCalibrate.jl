using ClimaCalibrate

include(joinpath(pkgdir(ClimaCalibrate), "test", "sf_calibration_utils.jl"))
experiment_config = ExperimentConfig(
    n_iterations,
    ensemble_size,
    observation,
    variance,
    prior,
    output_dir,
)

@assert get_backend() <: HPCBackend
hpc_kwargs = kwargs(time = 5, ntasks = 1, cpus_per_task = 1)
if get_backend() == DerechoBackend
    hpc_kwargs[:queue] = "develop"
end
eki = calibrate(experiment_config; model_interface, hpc_kwargs, verbose = true)
test_sf_calibration_output(eki, prior, experiment_config.observations)

# Remove previous output - this is not necessary but safe for tests
rm(output_dir, recursive = true)
# Pure Julia calibration, this should run anywhere
julia_eki = calibrate(JuliaBackend, experiment_config)
test_sf_calibration_output(julia_eki, prior, experiment_config.observations)

compare_g_ensemble(eki, julia_eki)

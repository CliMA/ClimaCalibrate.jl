struct CalibrationConfig{
    BACKEND <: ClimaCalibrate.Backend.AbstractBackend,
    EKP <: EKP.EnsembleKalmanProcess,
    PRIOR,
}
    backend::BACKEND
    ekp::EKP # This is updated and it is not in-place so this could become stale
    n_iterations::Int64
    prior::PRIOR
    output_dir::String
end

function CalibrationConfig(
    backend::ClimaCalibrate.Backend.AbstractBackend,
    ekp::EKP.EnsembleKalmanProcess,
    n_iterations,
    prior,
    output_dir,
)
    output_dir = abspath(output_dir)
    @assert isdir(output_dir) "Output directory does not exist: $output_dir"

    n_iterations >= 0 && error("Number of iterations ($n_iterations) must be non-negative")

    return CalibrationConfig(backend, ekp, n_iterations, prior, output_dir)
end

# Instead of

# calibrate(
#     backend::WorkerBackend,
#     ekp::EKP.EnsembleKalmanProcess,
#     n_iterations,
#     prior,
#     output_dir,
# )

# it is

# For JuliaBackend or WorkerBackend
# calibrate(config::CalibrationConfig)
# or
# for HPCBackend
# calibrate(
#     config,
#     model_interface;
#     experiment_dir = project_dir(),
#     exeflags = "",
# )

# I am not sure what advantage this brings beside removing a little bit of code
# duplication and removing all the number of arguments that need to be passed
# around

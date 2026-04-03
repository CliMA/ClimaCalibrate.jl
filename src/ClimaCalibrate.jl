module ClimaCalibrate

export project_dir

project_dir() = dirname(Base.active_project())

# TODO: Ask Nat about using Reexport later
# TODO: I don't know what should be exported and what shouldn't be exported
include("EKPUtils.jl")
using .EKPUtils:
    minibatcher_over_samples, observation_series_from_samples, g_ens_matrix
export minibatcher_over_samples, observation_series_from_samples, g_ens_matrix

include("Backend.jl")

using .BetterBackend:
    SlurmBackend,
    JuliaBackend,
    WorkerBackend,
    DerechoBackend,
    GCPBackend,
    ClimaGPUBackend,
    CaltechHPCBackend,
    get_backend,
    HPCBackend,
    SlurmManager,
    PBSManager,
    add_workers,
    get_manager,
    default_worker_pool,
    set_worker_loggers,
    set_worker_logger,
    map_remotecall_fetch,
    foreach_remotecall_wait,
    JobInfo,
    job_status,
    ispending,
    isrunning,
    issuccess,
    isfailed,
    iscompleted,
    submit_job,
    requeue_job,
    kill_job,
    make_job_script

export SlurmBackend,
    JuliaBackend,
    WorkerBackend,
    DerechoBackend,
    GCPBackend,
    ClimaGPUBackend,
    CaltechHPCBackend,
    get_backend,
    HPCBackend,
    SlurmManager,
    PBSManager,
    add_workers,
    get_manager,
    default_worker_pool,
    set_worker_loggers,
    set_worker_logger,
    map_remotecall_fetch,
    foreach_remotecall_wait,
    JobInfo,
    job_status,
    ispending,
    isrunning,
    issuccess,
    isfailed,
    iscompleted,
    submit_job,
    requeue_job,
    kill_job,
    make_job_script


include("Calibration.jl")

using .Calibration:
    calibrate

export calibrate

include("model_interface.jl")
include("observation_recipe.jl")
include("ensemble_builder.jl")
include("checkers.jl")
include("svd_analysis.jl")

end # module ClimaCalibrate

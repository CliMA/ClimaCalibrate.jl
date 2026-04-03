module ClimaCalibrate

export project_dir

project_dir() = dirname(Base.active_project())

include("ekp_utils.jl")
using .EKPUtils:
    minibatcher_over_samples,
    observation_series_from_samples,
    g_ens_matrix,
    get_metadata_for_nth_iteration,
    get_observations_for_nth_iteration
export minibatcher_over_samples,
    observation_series_from_samples,
    g_ens_matrix,
    get_metadata_for_nth_iteration,
    get_observations_for_nth_iteration

include("backend_manager.jl")

using .Backend:
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
    cancel_job,
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
    cancel_job,
    make_job_script


include("calibration.jl")

using .Calibration:
    calibrate,
    get_prior,
    save_eki_and_parameters,
    load_latest_ekp,
    initialize,
    path_to_ensemble_member,
    parameter_path,
    model_started,
    model_completed,
    write_model_completed,
    last_completed_iteration,
    save_G_ensemble,
    checkpoint_path,
    update_ensemble,
    update_ensemble!,
    observation_map_and_update!,
    get_param_dict,
    path_to_iteration,
    path_to_model_log

export calibrate,
    get_prior,
    save_eki_and_parameters,
    load_latest_ekp,
    initialize,
    path_to_ensemble_member,
    parameter_path,
    model_started,
    model_completed,
    write_model_completed,
    last_completed_iteration,
    save_G_ensemble,
    checkpoint_path,
    update_ensemble,
    update_ensemble!,
    observation_map_and_update!,
    get_param_dict,
    path_to_iteration,
    path_to_model_log

include("model_interface.jl")
include("observation_recipe.jl")
include("ensemble_builder.jl")
include("checkers.jl")
include("svd_analysis.jl")

end # module ClimaCalibrate

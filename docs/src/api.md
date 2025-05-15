# API

## Model Interface

```@docs
ClimaCalibrate.forward_model
ClimaCalibrate.observation_map
ClimaCalibrate.analyze_iteration
ClimaCalibrate.postprocess_g_ensemble
```

## Worker Interface
```@docs
ClimaCalibrate.add_workers
ClimaCalibrate.WorkerBackend
ClimaCalibrate.SlurmManager
ClimaCalibrate.PBSManager
ClimaCalibrate.set_worker_loggers
ClimaCalibrate.map_remotecall_fetch
ClimaCalibrate.foreach_remotecall_wait
```

## Backend Interface

```@docs
ClimaCalibrate.calibrate
ClimaCalibrate.JuliaBackend
ClimaCalibrate.DerechoBackend
ClimaCalibrate.CaltechHPCBackend
ClimaCalibrate.ClimaGPUBackend
ClimaCalibrate.get_backend
ClimaCalibrate.model_run
ClimaCalibrate.module_load_string
```

## Job Scheduler
```@docs
ClimaCalibrate.wait_for_jobs
ClimaCalibrate.log_member_error
ClimaCalibrate.kill_job
ClimaCalibrate.job_status
ClimaCalibrate.kwargs
ClimaCalibrate.slurm_model_run
ClimaCalibrate.generate_sbatch_script
ClimaCalibrate.generate_sbatch_directives
ClimaCalibrate.submit_slurm_job
ClimaCalibrate.pbs_model_run
ClimaCalibrate.generate_pbs_script
ClimaCalibrate.submit_pbs_job
```

## EnsembleKalmanProcesses Interface

```@docs
ClimaCalibrate.initialize
ClimaCalibrate.save_G_ensemble
ClimaCalibrate.update_ensemble
ClimaCalibrate.update_ensemble!
ClimaCalibrate.observation_map_and_update!
ClimaCalibrate.get_prior
ClimaCalibrate.get_param_dict
ClimaCalibrate.path_to_iteration
ClimaCalibrate.path_to_ensemble_member
ClimaCalibrate.path_to_model_log
ClimaCalibrate.parameter_path
ClimaCalibrate.minibatcher_over_samples
ClimaCalibrate.observation_series_from_samples
```

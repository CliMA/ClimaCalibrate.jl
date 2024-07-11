# API

## Model Interface

```@docs
ClimaCalibrate.set_up_forward_model
ClimaCalibrate.run_forward_model
ClimaCalibrate.observation_map
```

## Backend Interface

```@docs
ClimaCalibrate.get_backend
ClimaCalibrate.calibrate
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
ClimaCalibrate.ExperimentConfig
ClimaCalibrate.get_prior
ClimaCalibrate.get_param_dict
ClimaCalibrate.path_to_iteration
ClimaCalibrate.path_to_ensemble_member
ClimaCalibrate.path_to_model_log
```

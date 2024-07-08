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
ClimaCalibrate.sbatch_model_run
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

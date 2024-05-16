# API

## Model Interface

```@docs
ClimaCalibrate.set_up_forward_model
ClimaCalibrate.run_forward_model
ClimaCalibrate.observation_map
```

## Backend Interface

```@docs
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
```

# API

## Model Interface
```@docs
CalibrateAtmos.get_config
CalibrateAtmos.run_forward_model
CalibrateAtmos.get_forward_model
CalibrateAtmos.observation_map
```

## Backend Interface
```@docs
CalibrateAtmos.calibrate
CalibrateAtmos.srun_model
```

## EnsembleKalmanProcesses Interface
```@docs
CalibrateAtmos.initialize
CalibrateAtmos.save_G_ensemble
CalibrateAtmos.update_ensemble
CalibrateAtmos.ExperimentConfig
CalibrateAtmos.get_prior
CalibrateAtmos.get_param_dict
CalibrateAtmos.path_to_iteration
```

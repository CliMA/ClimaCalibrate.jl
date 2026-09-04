# API

## Package

```@docs
ClimaCalibrate.ClimaCalibrate
ClimaCalibrate.project_dir
```

## Model Interface

```@docs
ClimaCalibrate.AbstractModelInterface
ClimaCalibrate.forward_model
ClimaCalibrate.observation_map
ClimaCalibrate.analyze_iteration
ClimaCalibrate.postprocess_g_ensemble
ClimaCalibrate.model_interface_filepath
ClimaCalibrate.experiment_dir
ClimaCalibrate.exeflags
```

## Calibration Interface

```@docs
ClimaCalibrate.Calibration
ClimaCalibrate.calibrate
```

## Config Interface

```@docs
ClimaCalibrate.Backend.AbstractHPCConfig
ClimaCalibrate.Backend.SlurmConfig
ClimaCalibrate.Backend.SlurmConfig()
ClimaCalibrate.Backend.PBSConfig
ClimaCalibrate.Backend.PBSConfig()
```

## Backend Interface

```@docs
ClimaCalibrate.Backend
ClimaCalibrate.JuliaBackend
ClimaCalibrate.WorkerBackend
ClimaCalibrate.HPCBackend
ClimaCalibrate.SlurmBackend
ClimaCalibrate.CaltechHPCBackend
ClimaCalibrate.ClimaGPUBackend
ClimaCalibrate.GCPBackend
ClimaCalibrate.DerechoBackend
ClimaCalibrate.Backend.failure_rate
ClimaCalibrate.Backend.job_timeout
ClimaCalibrate.backend_type
ClimaCalibrate.get_backend
```

## Worker Interface
```@docs
ClimaCalibrate.SlurmManager
ClimaCalibrate.PBSManager
ClimaCalibrate.get_manager
ClimaCalibrate.add_workers
ClimaCalibrate.@worker_setup
ClimaCalibrate.calibration_worker_pool
ClimaCalibrate.cancel_worker_jobs
ClimaCalibrate.set_worker_logger
ClimaCalibrate.set_worker_loggers
ClimaCalibrate.map_remotecall_fetch
ClimaCalibrate.foreach_remotecall_wait
```

## Cluster Management Interface

```@docs
ClimaCalibrate.JobInfo
ClimaCalibrate.JobStatus
ClimaCalibrate.job_status
ClimaCalibrate.ispending
ClimaCalibrate.isrunning
ClimaCalibrate.issuccess
ClimaCalibrate.isfailed
ClimaCalibrate.iscompleted
ClimaCalibrate.submit_job
ClimaCalibrate.requeue_job
ClimaCalibrate.cancel_job
ClimaCalibrate.cancel_jobs_at_exit
ClimaCalibrate.job_records
ClimaCalibrate.write_job_script
ClimaCalibrate.make_job_script
```

## EnsembleKalmanProcesses Interface

```@docs
ClimaCalibrate.initialize
ClimaCalibrate.last_completed_iteration
ClimaCalibrate.terminated_iteration
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
ClimaCalibrate.checkpoint_path
ClimaCalibrate.load_latest_ekp
ClimaCalibrate.load_ekp_struct
ClimaCalibrate.ekp_path
ClimaCalibrate.save_eki_and_parameters
ClimaCalibrate.model_started
ClimaCalibrate.model_completed
ClimaCalibrate.write_model_started
ClimaCalibrate.write_model_completed
```

## EKP Utilities

```@docs
ClimaCalibrate.EKPUtils
ClimaCalibrate.EKPUtils.minibatcher_over_samples
ClimaCalibrate.EKPUtils.observation_series_from_samples
ClimaCalibrate.EKPUtils.get_observations_for_nth_iteration
ClimaCalibrate.EKPUtils.get_metadata_for_nth_iteration
ClimaCalibrate.EKPUtils.g_ens_matrix
```

## Sample Builder Interface

```@docs
ClimaCalibrate.SampleBuilder
ClimaCalibrateClimaAnalysisExt.SampleCollection
ClimaCalibrate.SampleBuilder.build_samples
ClimaCalibrate.SampleBuilder.build_samples_by_times
ClimaCalibrate.SampleBuilder.num_samples
ClimaCalibrate.SampleBuilder.reconstruct_col
ClimaCalibrate.SampleBuilder.get_samples
ClimaCalibrate.SampleBuilder.get_metadata
```

## Observation Recipe Interface

```@docs
ClimaCalibrate.ObservationRecipe
ClimaCalibrate.ObservationRecipe.AbstractCovarianceEstimator
ClimaCalibrate.ObservationRecipe.ScalarCovariance
ClimaCalibrate.ObservationRecipe.ScalarCovariance()
ClimaCalibrate.ObservationRecipe.SeasonalDiagonalCovariance
ClimaCalibrate.ObservationRecipe.SeasonalDiagonalCovariance()
ClimaCalibrate.ObservationRecipe.SVDplusDCovariance
ClimaCalibrate.ObservationRecipe.SVDplusDCovariance()
ClimaCalibrate.ObservationRecipe.QuantileRegularization
ClimaCalibrate.ObservationRecipe.covariance
ClimaCalibrate.ObservationRecipe.observation
ClimaCalibrate.ObservationRecipe.short_names
ClimaCalibrate.ObservationRecipe.reconstruct_g
ClimaCalibrate.ObservationRecipe.reconstruct_g_mean
ClimaCalibrate.ObservationRecipe.reconstruct_g_mean_final
ClimaCalibrate.ObservationRecipe.reconstruct_diag_cov
ClimaCalibrate.ObservationRecipe.reconstruct_vars
ClimaCalibrate.ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges
```

## SVD Residual Analysis

```@docs
ClimaCalibrate.analyze_residual
ClimaCalibrate.compute_structured_energy
ClimaCalibrate.compute_structured_energy_by_variable
ClimaCalibrate.compute_normalized_projections
```

## Ensemble Builder Interface

```@docs
ClimaCalibrate.EnsembleBuilder
ClimaCalibrateClimaAnalysisExt.GEnsembleBuilder
ClimaCalibrate.EnsembleBuilder.GEnsembleBuilder
ClimaCalibrate.EnsembleBuilder.fill_g_ens_col!
ClimaCalibrate.EnsembleBuilder.is_complete
ClimaCalibrate.EnsembleBuilder.get_g_ensemble
ClimaCalibrate.EnsembleBuilder.ranges_by_short_name
ClimaCalibrate.EnsembleBuilder.metadata_by_short_name
ClimaCalibrate.EnsembleBuilder.missing_short_names
```

## Checker Interface

```@docs
ClimaCalibrate.Checker
ClimaCalibrate.Checker.AbstractChecker
ClimaCalibrate.Checker.ShortNameChecker
ClimaCalibrate.Checker.DimNameChecker
ClimaCalibrate.Checker.DimUnitsChecker
ClimaCalibrate.Checker.UnitsChecker
ClimaCalibrate.Checker.DimValuesChecker
ClimaCalibrate.Checker.SequentialIndicesChecker
ClimaCalibrate.Checker.SignChecker
ClimaCalibrate.Checker.check
```

## Visualization Interface

```@docs
ClimaCalibrate.Visualization
ClimaCalibrate.Visualization.plot_g
ClimaCalibrate.Visualization.plot_g!
ClimaCalibrate.Visualization.plot_g_mean
ClimaCalibrate.Visualization.plot_g_mean!
ClimaCalibrate.Visualization.plot_obs
ClimaCalibrate.Visualization.plot_obs!
```

# API

## Model Interface

```@docs
ClimaCalibrate.forward_model
ClimaCalibrate.observation_map
ClimaCalibrate.analyze_iteration
ClimaCalibrate.postprocess_g_ensemble
```

## Calibration Interface

```@docs
ClimaCalibrate.calibrate
```

## Backend Interface

```@docs
ClimaCalibrate.JuliaBackend
ClimaCalibrate.HPCBackend
ClimaCalibrate.DerechoBackend
ClimaCalibrate.CaltechHPCBackend
ClimaCalibrate.ClimaGPUBackend
ClimaCalibrate.GCPBackend
ClimaCalibrate.WorkerBackend
ClimaCalibrate.get_backend
```

## Worker Interface
```@docs
ClimaCalibrate.SlurmManager
ClimaCalibrate.PBSManager
ClimaCalibrate.add_workers
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
ClimaCalibrate.make_job_script
```

## EnsembleKalmanProcesses Interface

```@docs
ClimaCalibrate.initialize
ClimaCalibrate.last_completed_iteration
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
ClimaCalibrate.save_eki_and_parameters
```

## EKP Utilities

```@docs
ClimaCalibrate.EKPUtils.minibatcher_over_samples
ClimaCalibrate.EKPUtils.observation_series_from_samples
ClimaCalibrate.EKPUtils.get_observations_for_nth_iteration
ClimaCalibrate.EKPUtils.get_metadata_for_nth_iteration
ClimaCalibrate.EKPUtils.g_ens_matrix
```

## Observation Recipe Interface

```@docs
ClimaCalibrate.ObservationRecipe.AbstractCovarianceEstimator
ClimaCalibrate.ObservationRecipe.ScalarCovariance
ClimaCalibrate.ObservationRecipe.ScalarCovariance()
ClimaCalibrate.ObservationRecipe.SeasonalDiagonalCovariance
ClimaCalibrate.ObservationRecipe.SeasonalDiagonalCovariance()
ClimaCalibrate.ObservationRecipe.SVDplusDCovariance
ClimaCalibrate.ObservationRecipe.SVDplusDCovariance(sample_dates)
ClimaCalibrate.ObservationRecipe.QuantileRegularization
ClimaCalibrate.ObservationRecipe.covariance
ClimaCalibrate.ObservationRecipe.observation
ClimaCalibrate.ObservationRecipe.short_names
ClimaCalibrate.ObservationRecipe.reconstruct_g_mean_final
ClimaCalibrate.ObservationRecipe.reconstruct_diag_cov
ClimaCalibrate.ObservationRecipe.reconstruct_vars
ClimaCalibrate.ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges
ClimaCalibrate.ObservationRecipe.change_data_type
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
ClimaAnalysisExt.GEnsembleBuilder
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

# API

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
ClimaCalibrate.JuliaBackend
ClimaCalibrate.HPCBackend
ClimaCalibrate.DerechoBackend
ClimaCalibrate.DerechoBackend(config::PBSConfig)
ClimaCalibrate.DerechoBackend(; )
ClimaCalibrate.CaltechHPCBackend
ClimaCalibrate.CaltechHPCBackend(config::SlurmConfig)
ClimaCalibrate.CaltechHPCBackend(; )
ClimaCalibrate.ClimaGPUBackend
ClimaCalibrate.ClimaGPUBackend(config::SlurmConfig)
ClimaCalibrate.ClimaGPUBackend(; )
ClimaCalibrate.GCPBackend
ClimaCalibrate.GCPBackend(config::SlurmConfig)
ClimaCalibrate.GCPBackend(; )
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
ClimaCalibrate.load_ekp_struct
ClimaCalibrate.ekp_path
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

## Sample Builder Interface

```@docs
ClimaCalibrateClimaAnalysisExt.SampleCollection
ClimaCalibrate.SampleBuilder.build_samples
ClimaCalibrate.SampleBuilder.build_samples_by_times
ClimaCalibrate.SampleBuilder.num_samples
ClimaCalibrate.SampleBuilder.reconstruct_col
ClimaCalibrateClimaAnalysisExt.ObservedSampleCollection
ClimaCalibrate.SampleBuilder.choose_obs
ClimaCalibrate.SampleBuilder.reconstruct_obs
ClimaCalibrate.SampleBuilder.get_obs
ClimaCalibrate.SampleBuilder.get_obs_metadata
ClimaCalibrate.SampleBuilder.get_samples
ClimaCalibrate.SampleBuilder.get_metadata
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
ClimaCalibrate.ObservationRecipe.reconstruct_g
ClimaCalibrate.ObservationRecipe.reconstruct_g_mean
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
ClimaCalibrate.Visualization.plot_g
ClimaCalibrate.Visualization.plot_g!
ClimaCalibrate.Visualization.plot_g_mean
ClimaCalibrate.Visualization.plot_g_mean!
ClimaCalibrate.Visualization.plot_obs
ClimaCalibrate.Visualization.plot_obs!
```

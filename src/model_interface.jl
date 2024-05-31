import EnsembleKalmanProcesses as EKP
import YAML


"""
    set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    set_up_forward_model(member, iteration, experiment_config::ExperimentConfig)

Set up and configure a single member's forward model. Used in conjunction with `run_forward_model`.

This function must be overriden by a component's model interface and 
should set things like the parameter path and other member-specific settings.
"""
set_up_forward_model(member, iteration, experiment_dir::AbstractString) =
    set_up_forward_model(member, iteration, ExperimentConfig(experiment_dir))

set_up_forward_model(member, iteration, experiment_config::ExperimentConfig) =
    error("set_up_forward_model not implemented")

"""
    run_forward_model(model_config)

Execute the forward model simulation with the given configuration.

This function should be overridden with model-specific implementation details.
`config` should be obtained from `set_up_forward_model`:
`run_forward_model(set_up_forward_model(member, iter, experiment_dir))`
"""
run_forward_model(model_config) = error("run_forward_model not implemented")

"""
    observation_map(iteration)

Runs the observation map for the specified iteration.
This function must be implemented for each calibration experiment.
"""
function observation_map(iteration)
    error("observation_map not implemented")
end

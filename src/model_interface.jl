import EnsembleKalmanProcesses as EKP
import YAML

export forward_model, observation_map

"""
    forward_model(iteration, member)

Execute the forward model simulation with the given configuration.

This function must be overridden by a component's model interface and 
should set things like the parameter path and other member-specific settings.
"""
function forward_model(iteration, member)
    error("forward_model not implemented")
end

"""
    observation_map(iteration)

Runs the observation map for the specified iteration.
This function must be implemented for each calibration experiment.
"""
function observation_map(iteration)
    error("observation_map not implemented")
end

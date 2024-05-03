import EnsembleKalmanProcesses as EKP
import YAML

"""
    AbstractPhysicalModel

Abstract type to define the interface for physical models.
"""
abstract type AbstractPhysicalModel end

"""
    get_config(physical_model::AbstractPhysicalModel, member, iteration, experiment_path::AbstractString)
    get_config(physical_model::AbstractPhysicalModel, member, iteration, experiment_config)

Fetch the model information for a specific ensemble member and iteration based on a provided path.
This function must be overriden by a component's model interface and 
should set things like the parameter path and other member-specific settings.
"""
function get_config(
    physical_model::AbstractPhysicalModel,
    member,
    iteration,
    experiment_path::AbstractString,
)
    experiment_config = ExperimentConfig(experiment_path)
    return get_config(physical_model, member, iteration, experiment_config)
end

get_config(
    physical_model::AbstractPhysicalModel,
    member,
    iteration,
    experiment_config,
) = error("get_config not implemented for $physical_model")

"""
    run_forward_model(physical_model::AbstractPhysicalModel, config)

Executes the forward model simulation with the given configuration.
The `config` should be obtained from `get_config`. 
This function should be overridden with model-specific implementation details.
"""
run_forward_model(physical_model::AbstractPhysicalModel, model_config) =
    error("run_forward_model not implemented for $physical_model")

"""
    get_forward_model(experiment_id::Val)

Retrieves a custom physical model struct for the specified experiment ID.
Throws an error if the experiment ID is unrecognized.
"""
function get_forward_model(experiment_id::Val)
    error("get_forward_model not implemented for $experiment_id")
end

"""
    observation_map(val:Val, iteration)

Runs the observation map for the specified iteration.
This function must be implemented for each calibration experiment.
"""
function observation_map(val::Val, iteration)
    error(
        "observation_map not implemented for experiment $val at iteration $iteration",
    )
end

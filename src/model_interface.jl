import EnsembleKalmanProcesses as EKP

import YAML

abstract type AbstractPhysicalModel end

function get_config(
    physical_model::AbstractPhysicalModel,
    member,
    iteration,
    experiment_id::AbstractString;
    kwargs...,
)
    experiment_config = ExperimentConfig(experiment_id; kwargs...)
    return get_config(physical_model, member, iteration, experiment_config)
end

"""
    get_config(member, iteration, experiment_id::AbstractString)
    get_config(member, iteration, experiment_config::AbstractDict)

Returns an AtmosConfig object for the given member and iteration.
If given an experiment id string, it will load the config from the corresponding YAML file.
Turns off default diagnostics and sets the TOML parameter file to the member's path.
This assumes that the config dictionary has `output_dir` and `restart_file` keys.
"""
get_config(
    physical_model::AbstractPhysicalModel,
    member,
    iteration,
    _,
) = error("get_config not implemented for $physical_model")

"""
    run_forward_model(physical_model, config)

Runs the forward model with the given configuration, returned by `get_config`.
"""
run_forward_model(physical_model::AbstractPhysicalModel, config) =
    error("run_forward_model not implemented for $physical_model")

"""
    get_forward_model(experiment_id::Val)

Returns the custom physical model objet for the given experiment id. An error is thrown if the experiment id is not recognized.
"""
function get_forward_model(experiment_id::Val)
    error("get_forward_model not implemented for $experiment_id")
end

"""
    observation_map(physical_model::AbstractPhysicalModel, iteration)

Returns the observation for the given case id Value and iteration.

NB: ensure that the model output is sufficiently sensitive to the input parameters.
"""
function observation_map(val::Val, iteration)
    error(
        "observation_map not implemented for experiment $val at iteration $iteration",
    )
end

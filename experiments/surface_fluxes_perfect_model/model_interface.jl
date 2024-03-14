import EnsembleKalmanProcesses as EKP
using CalibrateAtmos
import CalibrateAtmos:
    AbstractPhysicalModel, get_config, run_forward_model, get_forward_model
import YAML

"""
    SurfaceFluxModel

A type representing the surface fluxes perfect model.

We are using the inverse of the following problem
y(x) = G(θ, x) + ε
to obtain the posterior distribution of θ given y, x, and G.

where
y is the observed data, namely the profile-averaged frictional velocity, ustar)
G is the physical model, includes the model preliminaries, such as stationary parameters that are not being calibrated. In essence it wraps the surface_conditions function from the SurfaceFluxes package that calculates the MOST turbulent fluxes and the related characteristics.
θ is the calibatable parameter vector, namely [coefficient_a_h_businger, coefficient_a_m_businger, coefficient_b_h_businger, coefficient_b_m_businger]
ε is the observation error (for the perfect model case, we set it to 0)
x (optional) is the input data (e.g., the initial/boundary conditions and other non-stationary data inputs that y depends on - e.g. scenarios)

We need to follow the following steps for the calibration:
1. define model G, and the parameter vector θ that we want to calibrate
2. define the input data x, which is the initial/boundary conditions and other non-stationary predictors for the physical model (in this case we are generating large scale vertical profiles of atmospheric conditions)
    - we let the profiles to be the input data x, while the roughness length are stationary model preliminaries (uncalibrated stationary parameters)
3. obtain the observed data y (in this case of a perfect model, we are generating it using model G. We add some noise so we can see slightly slower convergence as we calibrate the model. In a real world scenario, we would obtain this from observations where each y vector observation would have an x input associated with it.)
4. define the prior distributions for θ (this is subjective and can be based on expert knowledge or previous studies)

"""
struct SurfaceFluxModel <: AbstractPhysicalModel end

include("sf_model.jl")
include("observation_map.jl")

function get_forward_model(::Val{:surface_fluxes_perfect_model})
    return SurfaceFluxModel()
end

function get_config(
    model::SurfaceFluxModel,
    member,
    iteration,
    experiment_id::AbstractString,
)
    config_dict = YAML.load_file("experiments/$experiment_id/model_config.yml")
    return get_config(model, member, iteration, config_dict)
end

"""
    get_config(member, iteration, experiment_id::AbstractString)
    get_config(member, iteration, config_dict::AbstractDict)

Returns an config dictionary object for the given member and iteration.
If given an experiment id string, it will load the config from the corresponding YAML file.
This assumes that the config dictionary has the `output_dir` key.
"""

function get_config(
    ::SurfaceFluxModel,
    member,
    iteration,
    config_dict::AbstractDict,
)
    # Specify member path for output_dir
    output_dir = config_dict["output_dir"]
    # Set TOML to use EKP parameter(s)
    member_path =
        EKP.TOMLInterface.path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end

    return config_dict
end

"""
    run_forward_model(config::AbstractDict)

Runs the model with the given an AbstractDict object.
"""

function run_forward_model(
    ::SurfaceFluxModel,
    config::AbstractDict;
    lk = nothing,
)
    x_inputs = if isnothing(lk)
        load_profiles(config["x_data_file"])
    else
        lock(lk) do
            load_profiles(config["x_data_file"])
        end
    end
    FT = typeof(x_inputs.profiles_int[1].T)
    obtain_ustar(FT, x_inputs, config)
end

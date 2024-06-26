import EnsembleKalmanProcesses as EKP
using ClimaCalibrate
import ClimaCalibrate: set_up_forward_model, run_forward_model, ExperimentConfig
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

experiment_dir = joinpath(
    pkgdir(ClimaCalibrate),
    "experiments",
    "surface_fluxes_perfect_model",
)
include(joinpath(experiment_dir, "sf_model.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))

function set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    return set_up_forward_model(
        member,
        iteration,
        ExperimentConfig(experiment_dir),
    )
end

"""
    set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    set_up_forward_model(member, iteration, experiment_config::ExperimentConfig)

Returns an config dictionary object for the given member and iteration.
Given an experiment dir, it will load the ExperimentConfig
This assumes that the config dictionary has the `output_dir` key.
"""
function set_up_forward_model(
    member,
    iteration,
    experiment_config::ExperimentConfig,
)
    # Specify member path for output_dir
    model_config = YAML.load_file(
        joinpath(
            "experiments",
            "surface_fluxes_perfect_model",
            "model_config.yml",
        ),
    )
    output_dir = (experiment_config.output_dir)
    # Set TOML to use EKP parameter(s)
    member_path =
        EKP.TOMLInterface.path_to_ensemble_member(output_dir, iteration, member)
    model_config["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(model_config, "toml")
        push!(model_config["toml"], parameter_path)
    else
        model_config["toml"] = [parameter_path]
    end

    return model_config
end

"""
    run_forward_model(config::AbstractDict)

Runs the model with the given an AbstractDict object.
"""

function run_forward_model(config::AbstractDict)
    x_inputs = load_profiles(config["x_data_file"])
    FT = typeof(x_inputs.profiles_int[1].T)
    obtain_ustar(FT, x_inputs, config)
end

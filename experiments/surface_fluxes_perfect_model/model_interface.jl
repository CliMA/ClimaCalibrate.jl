import EnsembleKalmanProcesses as EKP
using ClimaCalibrate
import ClimaCalibrate: forward_model

pkgdir_CC = pkgdir(ClimaCalibrate)

"""
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

experiment_dir =
    joinpath(pkgdir_CC, "experiments", "surface_fluxes_perfect_model")
include(joinpath(experiment_dir, "sf_model.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))

function forward_model(iteration, member)
    # Specify member path for output_dir
    model_config = Dict()
    output_dir = joinpath(pkgdir_CC, "output", "surface_fluxes_perfect_model")
    # Set TOML to use EKP parameter(s)
    member_path =
        EKP.TOMLInterface.path_to_ensemble_member(output_dir, iteration, member)
    model_config["output_dir"] = member_path
    model_config["toml"] = [joinpath(member_path, "parameters.toml")]
    x_data_file = joinpath(
        pkgdir_CC,
        "experiments",
        "surface_fluxes_perfect_model",
        "data",
        "synthetic_profile_data.jld2",
    )
    x_inputs = load_profiles(x_data_file)
    FT = typeof(x_inputs.profiles_int[1].T)
    obtain_ustar(FT, x_inputs, model_config)
end

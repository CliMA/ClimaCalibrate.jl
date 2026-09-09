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
θ is the calibratable parameter vector, here the single Businger coefficient
coefficient_a_m_businger
ε is the observation error, drawn from the distribution whose variance the
calibration is given as its noise covariance
x (optional) is the input data (e.g., the initial/boundary conditions and other non-stationary data inputs that y depends on - e.g. scenarios)

We need to follow the following steps for the calibration:
1. define model G, and the parameter vector θ that we want to calibrate
2. define the input data x, which is the initial/boundary conditions and other non-stationary predictors for the physical model (in this case we are generating large scale vertical profiles of atmospheric conditions)
    - we let the profiles to be the input data x, while the roughness length are stationary model preliminaries (uncalibrated stationary parameters)
3. obtain the observed data y (in this case of a perfect model, we generate it with model G and add observation noise, so that the calibration has something to recover. In a real world scenario, we would obtain this from observations where each y vector observation would have an x input associated with it.)
4. define the prior distributions for θ (this is subjective and can be based on expert knowledge or previous studies)
"""

"""
    SurfaceFluxModelInterface(output_dir, ensemble_size)

# Fields
- `output_dir`: Where the calibration writes each member's parameters and reads
  its output.
- `ensemble_size`: How many members the observation map has to collect.

The two are fields rather than globals because the forward model runs in a
worker or a scheduler job, which loads this file and the interface with it.
"""
struct SurfaceFluxModelInterface <: ClimaCalibrate.AbstractModelInterface
    output_dir::String
    ensemble_size::Int
end

experiment_dir =
    joinpath(pkgdir_CC, "experiments", "surface_fluxes_perfect_model")

ClimaCalibrate.model_interface_filepath(::SurfaceFluxModelInterface) =
    joinpath(experiment_dir, "model_interface.jl")
include(joinpath(experiment_dir, "sf_model.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))

function ClimaCalibrate.forward_model(
    interface::SurfaceFluxModelInterface,
    iteration,
    member,
)
    # Specify member path for output_dir
    model_config = Dict()
    (; output_dir) = interface
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

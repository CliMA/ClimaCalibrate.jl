
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFPP
import ClimaParams as CP
import JLD2
import Random

"""
    generate_G_preliminaries(FT)

Return the settings the surface fluxes model needs beyond the calibrated
parameters: the roughness lengths, the gustiness, and the discretization.

The roughness lengths are held fixed at 1e-2 m, since this experiment
calibrates the Businger coefficients rather than the surface.
"""
function generate_G_preliminaries(FT)
    config = SF.SurfaceFluxConfig(
        SF.ConstantRoughnessParams{FT}(z0m = FT(1e-2), z0s = FT(1e-2)),
        SF.ConstantGustinessSpec(FT(1)),
    )
    return (; config, scheme = SF.LayerAverageScheme())
end

"""
    MODEL_ERROR_FRACTION

The standard deviation of the model error, as a fraction of `ustar`.

A climate model reports a statistic of a chaotic trajectory, so two runs of the
same configuration return different numbers, and the loss the calibration sees
is rough rather than smooth. This idealized model is deterministic, so it stands
in for that with an error drawn per set of parameters. Drawing it from the
parameters rather than from a fresh seed is what keeps a calibration
reproducible: two backends running the same member get the same number.
"""
const MODEL_ERROR_FRACTION = 0.02

"""
    obtain_ustar(FT, x_inputs, model_config; return_ustar = false, model_error = true)

Obtain the friction velocity, ustar, of each profile from the surface fluxes
model, and write it to `model_config["output_dir"]`.

`model_config["toml"]` holds the parameter files that set the Businger
coefficients, which is how the calibration passes an ensemble member its
parameters. `model_error` adds [`MODEL_ERROR_FRACTION`](@ref); the observation
is generated without it, so that the misfit at the true parameters is the
observation error and the model error together, which is what the calibration
is given as its noise covariance.
"""
function obtain_ustar(
    FT,
    x_inputs,
    model_config;
    return_ustar = false,
    model_error = true,
)
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = CP.merge_toml_files(model_config["toml"]),
    )
    param_set = SFPP.SurfaceFluxesParameters(toml_dict, UF.BusingerParams)
    (; config, scheme) = generate_G_preliminaries(FT)

    (; profiles_sfc, profiles_int) = x_inputs
    ustar_array = Array{FT}(undef, length(profiles_int))
    @inbounds for (ii, prof_int) in enumerate(profiles_int)
        prof_sfc = profiles_sfc[ii]
        # The surface temperature and humidity are prescribed, so they are
        # passed as the initial guess with no callback to update them
        conditions = SF.surface_fluxes(
            param_set,
            prof_int.T,
            prof_int.q,
            FT(0),
            FT(0),
            prof_int.ρ,
            prof_sfc.T,
            prof_sfc.q,
            FT(0),
            prof_int.z - prof_sfc.z,
            FT(0),
            (prof_int.u, prof_int.v),
            (prof_sfc.u, prof_sfc.v),
            nothing,
            config,
            scheme,
        )
        ustar_array[ii] = conditions.ustar
    end

    # One draw for the whole profile set, since it stands in for the error of a
    # single model run rather than for scatter between profiles
    if model_error
        rng = Random.MersenneTwister(hash(param_set.ufp.a_m))
        ustar_array .*= 1 + FT(MODEL_ERROR_FRACTION) * randn(rng, FT)
    end

    JLD2.save_object(
        joinpath(model_config["output_dir"], "model_ustar_array.jld2"),
        ustar_array,
    )
    return return_ustar ? ustar_array : nothing
end

"""
    load_profiles(full_x_data_file_path)

Load the generated profiles from file.
"""
function load_profiles(full_x_data_file_path)
    data = JLD2.load(full_x_data_file_path)
    return (;
        profiles_sfc = data["profiles_sfc"],
        profiles_int = data["profiles_int"],
    )
end

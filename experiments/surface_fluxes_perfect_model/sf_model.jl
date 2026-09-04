
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFPP
import ClimaParams as CP
import JLD2

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
    obtain_ustar(FT, x_inputs, model_config; return_ustar = false)

Obtain the friction velocity, ustar, of each profile from the surface fluxes
model, and write it to `model_config["output_dir"]`.

`model_config["toml"]` holds the parameter files that set the Businger
coefficients, which is how the calibration passes an ensemble member its
parameters.
"""
function obtain_ustar(FT, x_inputs, model_config; return_ustar = false)
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

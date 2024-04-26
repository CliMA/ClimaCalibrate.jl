
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFPP
import ClimaParams as CP
import Thermodynamics as TD
import RootSolvers as RS
import JLD2
using LinearAlgebra: I
using Distributions: MvNormal
using EnsembleKalmanProcesses: constrained_gaussian, combine_distributions

"""
    generate_G_preliminaries()

Generate the necessary (stationary) inputs, that are passed to the surface fluxes model.
These could be initial conditions, boundary conditions and stationary parameters for time-dependent model.

"""
function generate_G_preliminaries(FT)
    uf_params = UF.BusingerParams
    scheme = SF.LayerAverageScheme()
    # we are not calibrating the roughness lengths (z0) here, so we set them to a constant value
    z0_momentum = Array{FT}([1e-2])
    z0_thermal = Array{FT}([1e-2])
    maxiter = 10
    param_set = SFPP.SurfaceFluxesParameters(FT, UF.BusingerParams)
    tol_neutral = FT(SF.Parameters.cp_d(param_set) / 10)
    gryanik_noniterative = false
    return (
        uf_params = uf_params,
        scheme = scheme,
        z0_momentum = z0_momentum,
        z0_thermal = z0_thermal,
        maxiter = maxiter,
        tol_neutral = tol_neutral,
        gryanik_noniterative = gryanik_noniterative,
    )
end

"""
    assemble_surface_conditions(thermo_params, prof_int, prof_sfc, z0m, z0b)

Prepare the surface state structs for the surface fluxes model.
"""
function assemble_surface_conditions(
    thermo_params,
    prof_int,
    prof_sfc,
    z0m,
    z0b,
)
    ts_sfc =
        TD.PhaseEquil_ρTq(thermo_params, prof_sfc.ρ, prof_sfc.T, prof_sfc.q)
    ts_int =
        TD.PhaseEquil_ρTq(thermo_params, prof_int.ρ, prof_int.T, prof_int.q)

    state_in = SF.StateValues(prof_int.z, (prof_int.u, prof_int.v), ts_int)
    state_sfc = SF.StateValues(prof_sfc.z, (prof_sfc.u, prof_sfc.v), ts_sfc)
    return SF.ValuesOnly(state_in, state_sfc, z0m, z0b)
end

"""
    obtain_ustar(FT, x_inputs, model_config; return_ustar = false)

Obtain the frictional velocity, ustar, from the surface fluxes model.
"""
function obtain_ustar(FT, x_inputs, model_config; return_ustar = false)

    # dict containing theta
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = CP.merge_toml_files(model_config["toml"]),
    )

    # extract model inputs, x
    (; profiles_sfc, profiles_int) = x_inputs

    # generate stationary model parameters
    (;
        scheme,
        z0_momentum,
        z0_thermal,
        maxiter,
        tol_neutral,
        gryanik_noniterative,
    ) = generate_G_preliminaries(FT)

    sch = scheme
    param_set = SFPP.SurfaceFluxesParameters(toml_dict, UF.BusingerParams)
    ustar_array = Array{FT}(
        undef,
        length(profiles_int),
        length(z0_momentum),
        length(z0_thermal),
    )
    @inbounds for (ii, prof_int) in enumerate(profiles_int)
        prof_sfc = profiles_sfc[ii]
        @inbounds for (kk, z0m) in enumerate(z0_momentum)
            @inbounds for (ll, z0b) in enumerate(z0_thermal)
                sc_states = assemble_surface_conditions(
                    param_set.thermo_params,
                    prof_int,
                    prof_sfc,
                    z0m,
                    z0b,
                )
                sc = SF.surface_conditions(
                    param_set,
                    sc_states,
                    sch;
                    maxiter,
                    tol_neutral,
                    soltype = RS.VerboseSolution(),
                    noniterative_stable_sol = gryanik_noniterative,
                )

                ustar_array[ii, kk, ll] = sc.ustar
            end
        end
    end
    @info "Saving ustar" ustar_array
    # save ustar_array to file
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

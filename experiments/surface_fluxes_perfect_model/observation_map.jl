using Statistics
import JLD2
import ClimaCalibrate: observation_map, path_to_ensemble_member

experiment_dir = joinpath(
    pkgdir(ClimaCalibrate),
    "experiments",
    "surface_fluxes_perfect_model",
)

"""
    observation_map(::Val{:surface_fluxes_perfect_model}, iteration)

Returns the observation map (from the raw model output to the observable y),
as specified by process_member_data, for the given iteration.
"""
function ClimaCalibrate.observation_map(
    interface::SurfaceFluxModelInterface,
    iteration,
)
    model_output = "model_ustar_array.jld2"
    (; output_dir, ensemble_size) = interface

    dims = 1
    G_ensemble = Array{Float64}(undef, dims..., ensemble_size)
    for m in 1:ensemble_size
        member_path = path_to_ensemble_member(output_dir, iteration, m)

        try
            ustar = JLD2.load_object(joinpath(member_path, model_output))
            G_ensemble[:, m] = process_member_data(ustar)
        catch e
            @info "An error occured in the observation map for member $m"
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

"""
    process_member_data(ustar)

Reduce one ensemble member's `ustar` profiles to the observable, their mean.

The element type is `Float64`, which is what the `EnsembleKalmanProcess` holds.
"""
process_member_data(ustar) = Float64[nanmean(ustar)]

nanmean(x) = mean(filter(!isnan, x))
nanvar(x) = var(filter(!isnan, x))

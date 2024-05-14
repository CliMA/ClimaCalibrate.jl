using Statistics
import YAML
import JLD2
import ClimaCalibrate:
    observation_map, ExperimentConfig, path_to_ensemble_member

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
function observation_map(::Val{:surface_fluxes_perfect_model}, iteration)
    config = ExperimentConfig(experiment_dir)
    (; output_dir, ensemble_size) = config
    model_output = "model_ustar_array.jld2"

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
    process_member_data(ustar; output_variance = false)

Process the data from a single ensemble member to obtain the observation.
If `output_variance` is true, return the observation and its variance.

This is used to transform the model output to the observation space.
Note that the outputs need to have element type of Float64 for the EKP struct.
"""
function process_member_data(ustar; output_variance = false)

    profile_mean = nanmean(ustar)
    observation = Float64[profile_mean]
    if !(output_variance)
        return observation
    else
        variance = Matrix{Float64}(undef, 1, 1)
        variance[1] = nanvar(ustar)
        return (; observation, variance)
    end
end
nanmean(x) = mean(filter(!isnan, x))
nanvar(x) = var(filter(!isnan, x))

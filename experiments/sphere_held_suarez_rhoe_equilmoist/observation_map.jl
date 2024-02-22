using NetCDF
using Statistics
import YAML
import EnsembleKalmanProcesses: TOMLInterface
import JLD2
import CalibrateAtmos: observation_map

export observation_map

function longitudinal_avg(arr)
    dims = 2
    for (idx, dim_size) in enumerate(size(arr))
        if dim_size == 180
            dims = idx
        end
    end
    return dropdims(mean(arr; dims); dims)
end

function latitudinal_avg(arr)
    dims = 3
    for (idx, dim_size) in enumerate(size(arr))
        if dim_size == 80
            dims = idx
        end
    end
    return dropdims(mean(arr; dims); dims)
end

function observation_map(::Val{:sphere_held_suarez_rhoe_equilmoist}, iteration)
    experiment_id = "sphere_held_suarez_rhoe_equilmoist"
    config =
        YAML.load_file(joinpath("experiments", experiment_id, "ekp_config.yml"))
    output_dir = config["output_dir"]
    ensemble_size = config["ensemble_size"]
    model_output = "ta_60d_average.nc"

    dims = 1
    G_ensemble = Array{Float64}(undef, dims..., ensemble_size)
    for m in 1:ensemble_size
        member_path =
            TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
        ta = ncread(joinpath(member_path, model_output), "ta")
        G_ensemble[:, m] = process_member_data(ta)
    end
    return G_ensemble
end

function process_member_data(ta; output_variance = false)
    # Cut off first 120 days to get equilibrium, take second level slice
    level_slice = 2
    ta_second_height = ta[3:size(ta)[1], :, :, level_slice]
    # Average over long and latitude
    area_avg_ta_second_height =
        longitudinal_avg(latitudinal_avg(ta_second_height))
    observation = Float64[area_avg_ta_second_height[3]]
    if !(output_variance)
        return observation
    else
        variance = Matrix{Float64}(undef, 1, 1)
        variance[1] = var(area_avg_ta_second_height)
        return (; observation, variance)
    end
end

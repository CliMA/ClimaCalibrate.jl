import NetCDF as NC
using Statistics
import YAML
import EnsembleKalmanProcesses: TOMLInterface
import JLD2
import CalibrateAtmos: observation_map, get_ekp_config
using ClimaAnalysis
export observation_map

function observation_map(::Val{:sphere_held_suarez_rhoe_equilmoist}, iteration)
    experiment_id = "sphere_held_suarez_rhoe_equilmoist"
    config = get_ekp_config(experiment_id)
    output_dir = config["output_dir"]
    ensemble_size = config["ensemble_size"]

    dims = 1
    G_ensemble = Array{Float64}(undef, dims..., ensemble_size)
    for m in 1:ensemble_size
        member_path =
            TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
        try
            simdir = SimDir(member_path)
            G_ensemble[:, m] .= process_member_data(simdir)
        catch e
            # Catch error for missing files
            @assert e isa NC.NetCDFError
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function process_member_data(simdir; output_variance = false)
    # Cut off first 120 days to get equilibrium, take second level slice
    ta = get(simdir; short_name = "ta", reduction = "average", period = "60d")
    area_avg_ta_second_height = slice(average_lat(average_lon(ta)), z = 242)
    observation =
        Float64(slice(area_avg_ta_second_height, time = 2.0736e7).data[1])
    if !(output_variance)
        return observation
    else
        variance = Matrix{Float64}(undef, 1, 1)
        variance[1] = var(area_avg_ta_second_height.data)
        return (; observation, variance)
    end
end

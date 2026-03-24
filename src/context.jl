module Context

import EnsembleKalmanProcesses as EKP
import JLD2

export CalibrationContext

struct CalibrationContext
    iter::Int64
    member::Union{Int64, Nothing}
    output_dir::String
    ensemble_size::Int64
    prior::Union{EKP.ParameterDistributions.ParameterDistribution, Nothing}
    ekp::Union{EKP.EnsembleKalmanProcess, Nothing}
    user_config::Any
end

# TODO: Add constructor with checks

function save_object(filename, ctx::CalibrationContext)
    JLD2.save_object(filename, ctx)
end

function load_object(filename)
    return JLD2.load_object(filename)
end

end

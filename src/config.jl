module Config

import EnsembleKalmanProcesses as EKP

struct CalibrationConfig
    ensemble_size::Int64
    n_iterations::Int64
    observations::Any
    prior::Any
    output_dir::String
    user_config::Any
end

# TODO: Add constructor with checks

function save_object(filename, config::CalibrationConfig)
    JLD2.save_object(filename, config)
end

function load_object(filename)
    return JLD2.load_object(filename)
end

end

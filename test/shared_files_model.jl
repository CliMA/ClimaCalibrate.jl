# A model for the SharedFilesBackend tests. Loaded by the test process and by
# every worker process it starts, so both sides define the same interface.
import ClimaCalibrate as CAL
import ClimaParams as CP
import JLD2

const FILE_MODEL_OUTPUT = "model_output.jld2"

# Reads its parameter, waits `delay` seconds, and writes the parameter back
# out, so the ensemble update moves the parameter toward the observation.
struct FileModel <: CAL.AbstractModelInterface
    output_dir::String
    ensemble_size::Int
    delay::Float64
end

function CAL.forward_model(interface::FileModel, iteration, member)
    (; output_dir, delay) = interface
    sleep(delay)
    param_path = CAL.parameter_path(output_dir, iteration, member)
    toml_dict = CP.create_toml_dict(Float64; override_file = param_path)
    (; test_param) = CP.get_parameter_values(toml_dict, "test_param")
    member_path = CAL.path_to_ensemble_member(output_dir, iteration, member)
    JLD2.save_object(joinpath(member_path, FILE_MODEL_OUTPUT), test_param)
    return nothing
end

function CAL.observation_map(interface::FileModel, iteration)
    (; output_dir, ensemble_size) = interface
    G = Array{Float64}(undef, 1, ensemble_size)
    for m in 1:ensemble_size
        member_path = CAL.path_to_ensemble_member(output_dir, iteration, m)
        G[:, m] .= JLD2.load_object(joinpath(member_path, FILE_MODEL_OUTPUT))
    end
    return G
end

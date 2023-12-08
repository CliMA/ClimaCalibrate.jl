import EnsembleKalmanProcesses as EKP
import ClimaAtmos as CA
import YAML

function get_atmos_config(member, iteration, experiment_id::AbstractString)
    config_dict = YAML.load_file("experiments/$experiment_id/atmos_config.yml")
    return get_atmos_config(member, iteration, config_dict)
end

"""
    get_atmos_config(member, iteration, experiment_id::AbstractString)
    get_atmos_config(member, iteration, config_dict::AbstractDict)

Returns an AtmosConfig object for the given member and iteration. 
If given an experiment id string, it will load the config from the corresponding YAML file.
Turns off default diagnostics and sets the TOML parameter file to the member's path.
This assumes that the config dictionary has `output_dir` and `restart_file` keys.
"""
function get_atmos_config(member, iteration, config_dict::AbstractDict)
    # Specify member path for output_dir
    # Set TOML to use EKP parameter(s)
    output_dir = config_dict["output_dir"]
    member_path =
        EKP.TOMLInterface.path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end

    # Turn off default diagnostics
    config_dict["output_default_diagnostics"] = false

    # Set restart file for initial equilibrium state
    ENV["RESTART_FILE"] = config_dict["restart_file"]
    return CA.AtmosConfig(config_dict)
end

"""
    run_forward_model(atmos_config::CA.AtmosConfig)

Runs the atmosphere model with the given an AtmosConfig object.
Currently only has basic error handling.
"""
function run_forward_model(atmos_config::CA.AtmosConfig)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        !isnothing(sol_res.sol) && sol_res.sol .= eltype(sol_res.sol)(NaN)
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end

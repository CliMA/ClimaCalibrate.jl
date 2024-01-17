module CalibrateAtmos

include("ekp_interface.jl")
include("atmos_interface.jl")

using PrecompileTools
import SciMLBase
import ClimaAtmos as CA
import YAML

@setup_workload begin
    output = joinpath("precompilation")
    job_id = "sphere_held_suarez_rhoe_equilmoist"
    config_file = joinpath("experiments", job_id, "atmos_config.yml")
    config_dict = YAML.load_file(config_file)
    config_dict["output_dir"] = output
    @compile_workload begin
        initialize(job_id)
        config = CA.AtmosConfig(config_dict)
        simulation = CA.get_simulation(config)
        (; integrator) = simulation
        SciMLBase.step!(integrator)
        CA.call_all_callbacks!(integrator)
    end
end

end # module CalibrateAtmos

# Using PrecompileTools faster model runs

PrecompileTools.jl enables developers to force the Julia compiler to save more code to disk, preventing re-compilation in the future.

For CalibrateAtmos, this is useful under certain conditions:
1. The atmosphere model configuration is set and will not change often. 
2. The model runtime is short compared to the compile time.

For point 1 above, this is because the model configuration specifies things like the floating-point type and callbacks, which affect the MethodInstances that get precompiled. Generically precompiling ClimaAtmos would take much too long to be useful.

For point 2, if the model runtime is an order of magnitude or more than the compilation time, any benefit from reduced compilation time will be trivial.

# How do I precompile my configuration?
The easiest way is by copying and pasting the code snippet below into `src/CalibrateAtmos.jl` and replacing the `job_id` with your experiment ID.
This will precompile the model step and all callbacks for the given configuration.
```julia
using PrecompileTools
import SciMLBase
import ClimaAtmos as CA
import YAML

@setup_workload begin
    output = joinpath("precompilation")
    job_id = "your configuration"
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
```

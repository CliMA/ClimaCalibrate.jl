module ClimaCalibrate

project_dir() = dirname(Base.active_project())

include("ekp_interface.jl")
include("model_interface.jl")
include("slurm.jl")
include("pbs.jl")
include("backends.jl")
include("emulate_sample.jl")
include("slurm_workers.jl")

end # module ClimaCalibrate

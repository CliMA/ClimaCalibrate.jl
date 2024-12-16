module ClimaCalibrate

export project_dir

project_dir() = dirname(Base.active_project())

include("ekp_interface.jl")
include("model_interface.jl")
include("slurm.jl")
include("pbs.jl")
include("workers.jl")
include("backends.jl")
include("emulate_sample.jl")

end # module ClimaCalibrate

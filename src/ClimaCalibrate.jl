module ClimaCalibrate

export project_dir

project_dir() = dirname(Base.active_project())

include("ekp_interface.jl")
include("model_interface.jl")
include("slurm.jl")
include("pbs.jl")
include("workers.jl")
include("backends.jl")
include("observation_recipe.jl")
include("ensemble_builder.jl")
include("checkers.jl")
include("svd_analysis.jl")

end # module ClimaCalibrate

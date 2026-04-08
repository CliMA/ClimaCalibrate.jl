module ClimaCalibrate
import Reexport: @reexport

export project_dir

project_dir() = dirname(Base.active_project())

include("ekp_utils.jl")
@reexport using .EKPUtils

include("backend.jl")
@reexport using .Backend

include("calibration.jl")
@reexport using .Calibration

include("model_interface.jl")
include("observation_recipe.jl")
include("ensemble_builder.jl")
include("checkers.jl")
include("svd_analysis.jl")

end # module ClimaCalibrate

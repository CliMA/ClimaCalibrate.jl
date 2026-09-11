"""
    ClimaCalibrate

Calibrate a forward model against observations using
[EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl).

Implement [`forward_model`](@ref) and [`observation_map`](@ref) for a subtype of
[`AbstractModelInterface`](@ref), then pass a backend, an
`EnsembleKalmanProcess`, and that interface to [`calibrate`](@ref). The backend
decides where the ensemble runs: [`JuliaBackend`](@ref) in the current process,
[`WorkerBackend`](@ref) across Distributed.jl workers, or an
[`HPCBackend`](@ref) as one scheduler job per ensemble member.

See the documentation at <https://CliMA.github.io/ClimaCalibrate.jl/stable/>.
"""
module ClimaCalibrate
import Reexport: @reexport

export project_dir

"""
    project_dir()

Return the directory of the currently active Julia project.

This is the default [`experiment_dir`](@ref), i.e. what an `HPCBackend` job
script is given as `--project` unless the model interface overrides it.
"""
project_dir() = dirname(Base.active_project())

include("model_interface.jl")

include("ekp_utils.jl")
@reexport using .EKPUtils

include("backend.jl")
@reexport using .Backend

include("calibration.jl")
@reexport using .Calibration

include("sample_builder.jl")
include("observation_recipe.jl")
include("ensemble_builder.jl")
include("checkers.jl")
include("svd_analysis.jl")

include("visualization.jl")

export SampleBuilder, ObservationRecipe, EnsembleBuilder, Checker, Visualization

end # module ClimaCalibrate

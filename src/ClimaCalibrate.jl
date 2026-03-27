module ClimaCalibrate

export project_dir

project_dir() = dirname(Base.active_project())

# TODO: Ask Nat about using Reexport later
include("EKPUtils.jl")
using .EKPUtils:
    minibatcher_over_samples,
    observation_series_from_samples,
    g_ens_matrix,
    get_metadata_for_nth_iteration,
    get_observations_for_nth_iteration
export minibatcher_over_samples,
    observation_series_from_samples,
    g_ens_matrix,
    get_metadata_for_nth_iteration,
    get_observations_for_nth_iteration

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

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

See the documentation at <https://CliMA.github.io/ClimaCalibrate.jl/dev/>.
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

# Functions that only have methods once an extension is loaded. Without a hint,
# calling one of these gives a bare MethodError that names no package.
const _CLIMAANALYSIS_STUBS = (
    SampleBuilder.build_samples,
    SampleBuilder.build_samples_by_times,
    SampleBuilder.num_samples,
    SampleBuilder.reconstruct_col,
    SampleBuilder.get_samples,
    SampleBuilder.get_metadata,
    ObservationRecipe.covariance,
    ObservationRecipe.observation,
    ObservationRecipe.short_names,
    ObservationRecipe.seasonally_aligned_yearly_sample_date_ranges,
    ObservationRecipe.reconstruct_g,
    ObservationRecipe.reconstruct_g_mean,
    ObservationRecipe.reconstruct_g_mean_final,
    ObservationRecipe.reconstruct_diag_cov,
    ObservationRecipe.reconstruct_vars,
    EnsembleBuilder.GEnsembleBuilder,
    EnsembleBuilder.fill_g_ens_col!,
    EnsembleBuilder.is_complete,
    EnsembleBuilder.get_g_ensemble,
    EnsembleBuilder.ranges_by_short_name,
    EnsembleBuilder.metadata_by_short_name,
    EnsembleBuilder.missing_short_names,
)

const _MAKIE_STUBS = (
    Visualization.plot_g,
    Visualization.plot_g!,
    Visualization.plot_g_mean,
    Visualization.plot_g_mean!,
    Visualization.plot_obs,
    Visualization.plot_obs!,
)

function _register_extension_hints()
    for (stubs, extension, advice) in (
        (
            _CLIMAANALYSIS_STUBS,
            :ClimaCalibrateClimaAnalysisExt,
            "Load ClimaAnalysis and NaNStatistics first: \
            `import ClimaAnalysis, NaNStatistics`.",
        ),
        (
            _MAKIE_STUBS,
            :ClimaCalibrateMakieExt,
            "Load a Makie backend first: `import CairoMakie`.",
        ),
    )
        Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
            exc.f in stubs || return nothing
            # With the extension loaded, the call has a method and the error is
            # about the arguments. Advice to import what is already imported
            # sends the reader the wrong way
            isnothing(Base.get_extension(@__MODULE__, extension)) ||
                return nothing
            print(
                io,
                "\n\n`$(nameof(exc.f))` is provided by a package \
                extension. $advice",
            )
        end
    end
    return nothing
end

function __init__()
    _register_extension_hints()
    return nothing
end

end # module ClimaCalibrate

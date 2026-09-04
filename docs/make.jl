using Documenter
using ClimaCalibrate
# The ClimaAnalysis extension is triggered by both of these. ClimaAnalysis
# happens to depend on NaNStatistics, so importing it alone loads the extension
# too, but relying on that means `get_extension` below returns `nothing` the day
# that dependency changes, and the build fails without saying why.
import ClimaAnalysis
import NaNStatistics
import CairoMakie # needed to load the Makie extension
import Makie
import Literate

Literate.markdown(
    joinpath(@__DIR__, "literate_example.jl"),
    joinpath(@__DIR__, "src"),
)

ClimaCalibrateClimaAnalysisExt =
    Base.get_extension(ClimaCalibrate, :ClimaCalibrateClimaAnalysisExt)
makedocs(
    # The Makie extension is left out: `Makie.@recipe` exports a plot type whose
    # attribute documentation carries `@ref`s into Makie's own docs, which
    # Documenter resolves only if Makie is listed here too.
    #
    # The submodules are listed individually because `checkdocs` works per
    # module: `names(ClimaCalibrate)` does not reach what they export, so
    # without them a newly added `ObservationRecipe` function would go
    # undocumented without anything noticing
    modules = [
        ClimaCalibrate,
        ClimaCalibrate.EKPUtils,
        ClimaCalibrate.Backend,
        ClimaCalibrate.Calibration,
        ClimaCalibrate.SampleBuilder,
        ClimaCalibrate.ObservationRecipe,
        ClimaCalibrate.EnsembleBuilder,
        ClimaCalibrate.Checker,
        ClimaCalibrate.Visualization,
        ClimaCalibrateClimaAnalysisExt,
    ],
    sitename = "ClimaCalibrate.jl",
    authors = "Clima",
    checkdocs = :exports,
    # Checking the external links makes one HTTP request per link, so it runs
    # on CI and not on every local build. A link that answers with a redirect
    # loop, a rate limit, or a temporary outage is reported and does not fail
    # the build
    linkcheck = !isempty(get(ENV, "CI", "")),
    warnonly = [:linkcheck],
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "How a calibration works" => "concepts.md",
        "Getting Started" => "quickstart.md",
        "Calibration Tutorial" => "literate_example.md",
        "Backends" => "backends.md",
        "Writing submission scripts" => "submit_scripts.md",
        "Observations" => [
            "Overview" => "observations.md",
            "Building samples" => "sample_builder.md",
            "Building observations" => "observation_recipe.md",
            "Building the G ensemble matrix" => "ensemble_builder.md",
        ],
        "Visualization" => "visualization.md",
        "Troubleshooting" => "troubleshooting.md",
        "How do I?" => "howdoi.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/CliMA/ClimaCalibrate.jl.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true,
)

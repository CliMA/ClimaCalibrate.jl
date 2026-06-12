using Documenter
using Documenter: doctest
using ClimaCalibrate
import ClimaAnalysis # needed to load ClimaAnalysis extension
using Base.CoreLogging
using DocumenterCitations
import Literate

disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

doctest(ClimaCalibrate; plugins = [bib])
disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging

Literate.markdown(
    joinpath(@__DIR__, "literate_example.jl"),
    joinpath(@__DIR__, "src"),
)

ClimaCalibrateClimaAnalysisExt =
    Base.get_extension(ClimaCalibrate, :ClimaCalibrateClimaAnalysisExt)
makedocs(
    plugins = [bib],
    modules = [ClimaCalibrate, ClimaCalibrateClimaAnalysisExt],
    sitename = "ClimaCalibrate.jl",
    authors = "Clima",
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "quickstart.md",
        "Distributed Calibration Tutorial" => "literate_example.md",
        "Backends" => "backends.md",
        "Submission Scripts" => "submit_scripts.md",
        "Observations" => "observations.md",
        "Observation Recipes" => "observation_recipe.md",
        "G Ensemble Builder" => "ensemble_builder.md",
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

using Documenter
using Documenter: doctest
using CalibrateAtmos
using Base.CoreLogging
using DocumenterCitations

disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

doctest(CalibrateAtmos; plugins = [bib])
disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging

makedocs(
    plugins = [bib],
    modules = [CalibrateAtmos],
    sitename = "CalibrateAtmos.jl",
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
        "Experiment Setup Guide" => "experiment_setup_guide.md",
        "Emulate and Sample" => "emulate_sample.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/CliMA/CalibrateAtmos.jl.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true,
)

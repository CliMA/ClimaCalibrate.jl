using Documenter
using Documenter: doctest
using ClimaCalibrate
using Base.CoreLogging
using DocumenterCitations

disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

doctest(ClimaCalibrate; plugins = [bib])
disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging

makedocs(
    plugins = [bib],
    modules = [ClimaCalibrate],
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
        "Backends" => "backends.md",
        "Emulate and Sample" => "emulate_sample.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/CliMA/ClimaCalibrate.jl.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true,
)

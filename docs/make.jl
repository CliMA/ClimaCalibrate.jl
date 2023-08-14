using Documenter
using Documenter: doctest
using Base.CoreLogging
using DocumenterCitations

disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging

makedocs(
    CitationBibliography(joinpath(@__DIR__, "bibliography.bib")),
    modules = Vector{Module}(),
    sitename = "CalibrateAtmos.jl",
    authors = "Clima",
    strict = true,
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(
    repo = "github.com/CliMA/CalibrateAtmos.jl.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true,
)

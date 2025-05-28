using Test
using ClimaCalibrate
using Aqua

@testset "Aqua tests (performance)" begin
    ua = Aqua.detect_unbound_args_recursively(ClimaCalibrate)
    @test length(ua) == 0

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(ClimaCalibrate; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("ClimaCalibrate", pkgdir(last(x).module)), ambs)

    # Uncomment for debugging:
    # for method_ambiguity in ambs
    #     @show method_ambiguity
    # end
    @test length(ambs) == 0
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(ClimaCalibrate)
    Aqua.test_stale_deps(ClimaCalibrate)
    Aqua.test_deps_compat(ClimaCalibrate, check_extras = false)
    Aqua.test_project_extras(ClimaCalibrate)
    Aqua.test_piracies(ClimaCalibrate)
end

nothing

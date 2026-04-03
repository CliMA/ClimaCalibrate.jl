using Test
using SafeTestsets

#! format: off
@safetestset "Aqua" begin include("aqua.jl") end
@safetestset "EKP utils" begin include("ekp_utils.jl") end
@safetestset "EKP interface" begin include("ekp_interface.jl") end
@safetestset "Model interface" begin include("model_interface.jl") end
@safetestset "Julia backend" begin include("julia_backend.jl") end
@safetestset "Observation recipe" begin include("observation_recipe.jl") end
@safetestset "Ensemble builder" begin include("ensemble_builder.jl") end
@safetestset "SVD analysis" begin include("svd_analysis.jl") end
#! format: on

nothing

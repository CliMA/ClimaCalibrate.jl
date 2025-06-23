using Test
using SafeTestsets

#! format: off
@safetestset "EKP interface" begin include("ekp_interface.jl") end
@safetestset "Model interface" begin include("model_interface.jl") end
@safetestset "Emulate sample" begin include("emulate_sample.jl") end
@safetestset "Julia backend" begin include("julia_backend.jl") end
@safetestset "Aqua" begin include("aqua.jl") end
@safetestset "Observation recipe" begin include("observation_recipe.jl") end
#! format: on

nothing

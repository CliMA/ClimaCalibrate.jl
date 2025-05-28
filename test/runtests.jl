using Test
using SafeTestsets

#! format: off
@safetestset "EKP interface" begin include("ekp_interface.jl") end
@safetestset "Model interface" begin include("model_interface.jl") end
# Disabled since we use EKP 2.0 in testing, CES is still incompatible with EKP 2.0
# @safetestset "Emulate sample" begin include("emulate_sample.jl") end
@safetestset "Julia backend" begin include("julia_backend.jl") end
@safetestset "Aqua" begin include("aqua.jl") end
#! format: on

nothing

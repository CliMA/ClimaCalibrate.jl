using Test

import Random
Random.seed!(1234)

include("test_init.jl")
include("test_atmos_config.jl")
include("test_emulate_sample.jl")

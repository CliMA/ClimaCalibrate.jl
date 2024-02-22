using Test

import Random
Random.seed!(1234)

include("test_init.jl")
# include("test_atmos_config.jl") # TODO: to be moved to ClimaAtmos
include("test_model_interface.jl")
include("test_emulate_sample.jl")

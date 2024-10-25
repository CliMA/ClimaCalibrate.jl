using Test

include("ekp_interface.jl")
include("model_interface.jl")
# Disabled since we use EKP 2.0 in testing, CES is still incompatible with EKP 2.0
# include("emulate_sample.jl")
include("pure_julia_e2e.jl")
include("aqua.jl")

# experiment-specific Project.toml instantiation, function extensions and config load

using Pkg
experiment_id = ARGS[1]
Pkg.activate("experiments/$experiment_id")
Pkg.instantiate()
using CalibrateAtmos
pkg_dir = pkgdir(CalibrateAtmos)
experiment_path = "$pkg_dir/experiments/$experiment_id"
include("$experiment_path/model_interface.jl")
include("$experiment_path/generate_truth.jl")

CalibrateAtmos.calibrate(experiment_id)

include("$pkg_dir/plot/convergence_$experiment_id.jl")

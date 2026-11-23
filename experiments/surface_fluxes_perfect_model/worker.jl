# Worker for a SharedFilesBackend calibration of this experiment on a machine
# that reaches the driver only through a bucket, such as a Vast.ai instance
# from `Backend.Vast.launch_workers`. Reads CLIMA_BUCKET_ENDPOINT,
# CLIMA_BUCKET, CLIMA_BUCKET_PREFIX, and the AWS_* credentials.
using Pkg
Pkg.develop(path = joinpath(@__DIR__, "..", ".."))
Pkg.add("AWSS3")

import ClimaCalibrate
using AWSS3
# The forward model reads the generated profile data, so it is generated here
include(joinpath(@__DIR__, "generate_data.jl"))

files = ClimaCalibrate.SharedFiles.BucketFiles(
    ENV["CLIMA_BUCKET_ENDPOINT"],
    ENV["CLIMA_BUCKET"],
    ENV["CLIMA_BUCKET_PREFIX"],
)
ClimaCalibrate.run_worker(
    SurfaceFluxModelInterface("/output", 8),
    "/output";
    files,
    outputs = ["model_ustar_array.jld2"],
)

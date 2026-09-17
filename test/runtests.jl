using ClimaCalibrate
using ParallelTestRunner


# Buildkite runs the scheduler-dependent files, which need a real cluster:
#   hpc_backend.jl, worker_backend.jl        - all three pipelines
#   slurm_unit_tests.jl, slurm_manager_unit_tests.jl
#       - pipeline.yml, clima_gpu_pipeline.yml
#   pbs_unit_tests.jl, pbs_manager_unit_tests.jl
#       - derecho_pipeline.yml
# The tests that run without a scheduler are below.

# Simplify adding tests to the test suite
macro include(x)
    return Expr(:quote, Expr(:call, :include, x))
end

#! format: off
testsuite = Dict(
    "EKP utils" => @include("ekp_utils.jl"),
    "EKP interface" => @include("ekp_interface.jl"),
    "Model interface" => @include("model_interface.jl"),
    "Julia backend" => @include("julia_backend.jl"),
    "Job status" => @include("job_status.jl"),
    "HPC config" => @include("backend_config.jl"),
    "HPC job scripts" => @include("hpc_job_scripts.jl"),
    "Workers per node" => @include("workers_per_node.jl"),
    "Worker pool" => @include("worker_pool.jl"),
    "Sampler" => @include("sample_builder.jl"),
    "Observation recipe" => @include("observation_recipe.jl"),
    "Ensemble builder" => @include("ensemble_builder.jl"),
    "SVD analysis" => @include("svd_analysis.jl"),
    "Visualization" => @include("visualization.jl"),
    "Aqua" => @include("aqua.jl"),
)
#! format: on

runtests(ClimaCalibrate, ARGS; testsuite)

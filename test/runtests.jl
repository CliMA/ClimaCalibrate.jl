using Test
using SafeTestsets

# Buildkite runs the scheduler-dependent files, which need a real cluster:
#   hpc_backend.jl, worker_backend.jl        - all three pipelines
#   slurm_unit_tests.jl, slurm_manager_unit_tests.jl
#       - pipeline.yml, clima_gpu_pipeline.yml
#   pbs_unit_tests.jl, pbs_manager_unit_tests.jl
#       - derecho_pipeline.yml
# The tests that run without a scheduler are below.

#! format: off
@safetestset "EKP utils" begin include("ekp_utils.jl") end
@safetestset "EKP interface" begin include("ekp_interface.jl") end
@safetestset "Model interface" begin include("model_interface.jl") end
@safetestset "Julia backend" begin include("julia_backend.jl") end
@safetestset "Job status" begin include("job_status.jl") end
@safetestset "HPC config" begin include("backend_config.jl") end
@safetestset "HPC job scripts" begin include("hpc_job_scripts.jl") end
@safetestset "Workers per node" begin include("workers_per_node.jl") end
@safetestset "Sampler" begin include("sample_builder.jl") end
@safetestset "Observation recipe" begin include("observation_recipe.jl") end
@safetestset "Ensemble builder" begin include("ensemble_builder.jl") end
@safetestset "SVD analysis" begin include("svd_analysis.jl") end
@safetestset "Visualization" begin include("visualization.jl") end
# Aqua runs last so that the extensions are loaded by the tests above. With
# them unloaded, its ambiguity, piracy, and unbound-argument checks skip all of
# ext/.
@safetestset "Aqua" begin include("aqua.jl") end
#! format: on

nothing

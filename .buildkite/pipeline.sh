#! /bin/bash
set -euo pipefail

experiment_id="surface_fluxes_perfect_model"
exp_dir="experiments/$experiment_id"
# Need way of reading this info from EKP config
ensemble_size=10
n_iterations=3
ntasks=1
cpus_per_task=1
time_limit=5

plot="true"
generate_data="true"

# Overall pipeline configuration
cat << 'EOM'
agents:
  queue: new-central
  modules: climacommon/2024_03_18
env:
  JULIA_DEPOT_PATH: "${BUILDKITE_BUILD_PATH}/${BUILDKITE_PIPELINE_SLUG}/depot/default"
EOM

# Initialization
cat << EOM
steps:
  - label: Initialize
    key: init
    command: |
      julia --project -e 'import Pkg; Pkg.instantiate(;verbose=true)'
      julia --project=$exp_dir -e 'import Pkg; Pkg.instantiate(;verbose=true)'
EOM

if [ "$generate_data" == "true" ] ; then
    echo "      julia --project=$exp_dir $exp_dir/generate_data.jl"
fi

cat << EOM
      julia --project=$exp_dir -e '
        import CalibrateAtmos
        CalibrateAtmos.initialize("surface_fluxes_perfect_model")'
    agents:
      slurm_cpus_per_task: 8
    env:
      JULIA_NUM_PRECOMPILE_TASKS: 8
      JULIA_MAX_NUM_PRECOMPILE_FILES: 50
EOM

# Run each iteration
for i in $(seq 0 $((n_iterations - 1)))
do
cat << EOM

  - wait

  - label: ":abacus: iter $i"
    key: iter_$i
    parallelism: $ensemble_size
    command: |
      srun julia --project=$exp_dir -e '
        import CalibrateAtmos as CAL
        experiment_id = "$experiment_id"
        i = $i; member = parse(Int, ENV["BUILDKITE_PARALLEL_JOB"]) + 1
        include("$exp_dir/model_interface.jl")
        physical_model = CAL.get_forward_model(Val(Symbol(experiment_id)))
        config = CAL.get_config(physical_model, member, i, experiment_id)
        CAL.run_forward_model(physical_model, config)
      '
    artifact_paths: output/$experiment_id
    agents:
      slurm_cpus_per_task: $cpus_per_task
      slurm_ntasks: $ntasks
      slurm_time: $time_limit

  - wait
  
  - label: ":recycle: update"
    key: iter_${i}_update
    depends_on: iter_$i
    command: |
      julia --project=$exp_dir -e '
        import CalibrateAtmos as CAL

        experiment_id = "$experiment_id"
        i = $i
        include("$exp_dir/model_interface.jl")
        G_ensemble = CAL.observation_map(Val(Symbol(experiment_id)), i)
        CAL.save_G_ensemble(experiment_id, i, G_ensemble)
        CAL.update_ensemble(experiment_id, i)
      '
    artifact_paths: output/$experiment_id
EOM
done

if [ "$plot" == "true" ] ; then

cat << EOM

  - wait

  - label: ":artist_palette: plot"
    command: julia --project=$exp_dir $exp_dir/plot.jl
    artifact_paths: output/$experiment_id
EOM

fi

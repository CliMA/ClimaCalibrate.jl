#! /bin/bash
set -euo pipefail

experiment_id="surface_fluxes_perfect_model"
# Need way of reading this info from EKP config
ensemble_size=10
n_iterations=3
ntasks=1
cpus_per_task=1
time_limit=5

# Initialize pipeline, project and calibration
cat << EOM
agents:
  queue: new-central
  modules: climacommon/2024_03_18

steps:
  - label: Initialize
    key: init
    command: |
      julia --project=experiments/$experiment_id -e '
        import Pkg; Pkg.build("CalibrateAtmos")
        Pkg.instantiate(;verbose=true)'
      julia --project=experiments/$experiment_id experiments/$experiment_id/generate_truth.jl
      julia --project=experiments/$experiment_id -e '
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
      srun julia --project=experiments/$experiment_id -e '
        import Pkg; Pkg.status();
        @show Base.active_project()
        import CalibrateAtmos as CAL
        experiment_id = "$experiment_id"
        i = $i; member = parse(Int, ENV["BUILDKITE_PARALLEL_JOB"]) + 1
        include("experiments/$experiment_id/model_interface.jl")
        physical_model = CAL.get_forward_model(Val(Symbol(experiment_id)))
        config = CAL.get_config(physical_model, member, i, experiment_id)
        CAL.run_forward_model(physical_model, config)
      '
    agents:
      slurm_cpus_per_task: $cpus_per_task
      slurm_ntasks: $ntasks
      slurm_time: $time_limit

  - wait
  
  - label: ":recycle: update"
    key: iter_${i}_update
    depends_on: iter_$i
    command: |
      julia --project=experiments/$experiment_id -e '
        import CalibrateAtmos as CAL

        experiment_id = "$experiment_id"
        i = $i
        include("experiments/$experiment_id/model_interface.jl")
        G_ensemble = CAL.observation_map(Val(Symbol(experiment_id)), i)
        CAL.save_G_ensemble(experiment_id, i, G_ensemble)
        CAL.update_ensemble(experiment_id, i)
      '
EOM
done

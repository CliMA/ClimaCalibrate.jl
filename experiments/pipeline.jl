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
ekp_config =
    YAML.load_file(joinpath("experiments", experiment_id, "ekp_config.yml"))

# initialize the CalibrateAtmos
CalibrateAtmos.initialize(experiment_id)

# run experiment with CalibrateAtmos for N_iter iterations
N_iter = ekp_config["n_iterations"]
N_mem = ekp_config["ensemble_size"]
for i in 0:(N_iter - 1)
    # run G model to produce output from N_mem ensemble members
    physical_model =
        CalibrateAtmos.get_forward_model(Val(Symbol(experiment_id)))

    for m in 1:N_mem # TODO: parallelize with threads!
        # model run for each ensemble member
        model_config =
            CalibrateAtmos.get_config(physical_model, m, i, experiment_id)
        CalibrateAtmos.run_forward_model(physical_model, model_config)
        @info "Finished model run for member $m at iteration $i"
    end

    # update EKP with the ensemble output and update calibrated parameters
    G_ensemble = CalibrateAtmos.observation_map(Val(Symbol(experiment_id)), i)
    output_dir = ekp_config["output_dir"]
    iter_path = CalibrateAtmos.path_to_iteration(output_dir, i)
    JLD2.save_object(joinpath(iter_path, "observation_map.jld2"), G_ensemble)
    CalibrateAtmos.update_ensemble(experiment_id, i)

end

include("$pkg_dir/plot/convergence_$experiment_id.jl")

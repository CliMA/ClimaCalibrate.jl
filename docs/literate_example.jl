# # Distributed Calibration Tutorial Using Julia Workers
# This example will teach you how to use ClimaCalibrate to parallelize your calibration with workers.
# Workers are additional processes spun up to run code in a distributed fashion. 
# In this tutorial, we will run ensemble members' forward models on different workers.

# The example calibration uses CliMA's atmosphere model, [`ClimaAtmos.jl`](https://github.com/CliMA/ClimaAtmos.jl/), 
# in a column spatial configuration for 30 days to simulate outgoing radiative fluxes.
# Radiative fluxes are used in the observation map to calibrate the astronomical unit.

# First, we load in some necessary packages.
using Distributed
import ClimaCalibrate as CAL
import ClimaAnalysis: SimDir, get, slice, average_xy
using ClimaUtilities.ClimaArtifacts
import EnsembleKalmanProcesses: I, ParameterDistributions.constrained_gaussian

# Next, we add workers. These are primarily added by 
# [`Distributed.addprocs`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.addprocs)
# or by starting Julia with multiple processes: `julia -p <nprocs>`.

# `addprocs` itself initializes the workers and registers them with the main Julia process, but there are multiple ways to call it.
# The simplest is just `addprocs(nprocs)`, which will create new local processes on your machine.
# The other is to use [`SlurmManager`](@ref), which will acquire and start workers on Slurm resources.
# You can use keyword arguments to specify the Slurm resources:

# `addprocs(ClimaCalibrate.SlurmManager(nprocs), gpus_per_task = 1, time = "01:00:00")`

# For this example, we would add one worker if it was compatible with Documenter.jl:
# ```julia
# addprocs(1)
# ```

# We can see the number of workers and their ID numbers:
nworkers()
#-
workers()

# We can call functions on the worker using [`remotecall`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.remotecall_fetch-Tuple{Any,%20Integer,%20Vararg{Any}}). 
# We pass in the function name and the worker ID followed by the function arguments.
remotecall_fetch(*, 1, 4, 4)
# ClimaCalibrate uses this functionality to run the forward model on workers.

# Since the workers start in their own Julia sessions, we need to import packages and declare variables.
# `Distributed.@everywhere` executes code on all workers, allowing us to load the code that they need.
@everywhere begin
    output_dir = joinpath("output", "climaatmos_calibration")
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    import ClimaComms
end
output_dir = joinpath("output", "climaatmos_calibration")
mkpath(output_dir)

# First, we need to set up the forward model, which take in the sampled parameters,
# runs, and saves diagnostic output that can be processed and compared to observations. The 
# forward model must override `ClimaCalibrate.forward_model(iteration, member)`,
# since the workers will run this function in parallel.

# Since `forward_model(iteration, member)` only takes in the iteration and member numbers, 
# so we need to use these as hooks to set the model parameters and output directory.
# Two useful functions:
# - [`path_to_ensemble_member`](@ref): Returns the ensemble member's output directory
# - [`parameter_path`](@ref): Returns the ensemble member's parameter file as specified by [`EKP.TOMLInterface`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/TOMLInterface/#EnsembleKalmanProcesses.TOMLInterface.save_parameter_ensemble)

# The forward model below is running `ClimaAtmos.jl` in a minimal `column` spatial configuration.
@everywhere function CAL.forward_model(iteration, member)
    config_dict = Dict(
        "dt" => "2000secs",
        "t_end" => "30days",
        "config" => "column",
        "h_elem" => 1,
        "insolation" => "timevarying",
        "output_dir" => output_dir,
        "output_default_diagnostics" => false,
        "dt_rad" => "6hours",
        "rad" => "clearsky",
        "co2_model" => "fixed",
        "log_progress" => false,
        "diagnostics" => [
            Dict(
                "reduction_time" => "average",
                "short_name" => "rsut",
                "period" => "30days",
                "writer" => "nc",
            ),
        ],
    )
    #md # Set the output path for the current member
    member_path = CAL.path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path

    #md # Set the parameters for the current member
    parameter_path = CAL.parameter_path(output_dir, iteration, member)
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end

    #md # Turn off default diagnostics
    config_dict["output_default_diagnostics"] = false

    comms_ctx = ClimaComms.SingletonCommsContext()
    atmos_config = CA.AtmosConfig(config_dict; comms_ctx)
    simulation = CA.get_simulation(atmos_config)
    CA.solve_atmos!(simulation)
    return simulation
end

# Next, the observation map is required to process a full ensemble of model output
# for the ensemble update step. The observation map just takes in the iteration number,
# and always outputs an array. 
# For observation map output `G_ensemble`, `G_ensemble[:, m]` must the output of ensemble member `m`.
# This is needed for compatibility with EnsembleKalmanProcesses.jl.
const days = 86_400
function CAL.observation_map(iteration)
    single_member_dims = (1,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

    for m in 1:ensemble_size
        member_path = CAL.path_to_ensemble_member(output_dir, iteration, m)
        simdir_path = joinpath(member_path, "output_active")
        if isdir(simdir_path)
            simdir = SimDir(simdir_path)
            G_ensemble[:, m] .= process_member_data(simdir)
        else
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

# Separating out the individual ensemble member output processing often
# results in more readable code.
function process_member_data(simdir::SimDir)
    isempty(simdir.vars) && return NaN
    rsut =
        get(simdir; short_name = "rsut", reduction = "average", period = "30d")
    return slice(average_xy(rsut); time = 30days).data
end

# Now, we can set up the remaining experiment details:
# - ensemble size, number of iterations
# - the prior distribution
# - the observational data
ensemble_size = 30
n_iterations = 7
noise = 0.1 * I
prior = constrained_gaussian("astronomical_unit", 6e10, 1e11, 2e5, Inf)

# For a perfect model, we generate observations from the forward model itself.
# This is most easily done by creating an empty parameter file and 
# running the 0th ensemble member:
@info "Generating observations"
parameter_file = CAL.parameter_path(output_dir, 0, 0)
mkpath(dirname(parameter_file))
touch(parameter_file)
simulation = CAL.forward_model(0, 0)
# Lastly, we use the observation map itself to generate the observations.
observations = Vector{Float64}(undef, 1)
observations .= process_member_data(SimDir(simulation.output_dir))

# Now we are ready to run our calibration, putting it all together using the 
# `calibrate` function. The `WorkerBackend` will automatically use all workers
# available to the main Julia process.
# Other backends are available for forward models that can't use workers or need to be parallelized internally.
# The simplest backend is the `JuliaBackend`, which runs all ensemble members sequentially and does not require `Distributed.jl`.
# For more information, see the [`Backends`](https://clima.github.io/ClimaCalibrate.jl/dev/backends/) page.
eki = CAL.calibrate(
    CAL.WorkerBackend(),
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    output_dir,
)

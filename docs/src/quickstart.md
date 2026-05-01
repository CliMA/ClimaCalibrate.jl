# Getting Started

!!! note "Preliminaries"
    You may find it helpful to read the [documentation](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/)
    of EnsembleKalmanProcesses.jl before reading this section.

Every calibration requires
- observational data, which can be a Vector or an
  [`EnsembleKalmanProcess.Observation`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/Observations/#EnsembleKalmanProcesses.Observation)
- a prior parameter distribution. The easiest way to construct a distribution is
  with the [`EnsembleKalmanProcess.constrained_gaussian`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/ParameterDistributions/#EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian)
  function,
- a forward model, which uses input parameters to return diagnostic output
- an observation map, which maps the forward model's diagnostic output to a
  vector comparable to the observations

## Implementing your experiment

All [`calibrate`](@ref) functions require a backend, an
`EnsembleKalmanProcesses.EnsembleKalmanProcess` object, and a model interface.
This tutorial will not go into details on how to construct the
`EnsembleKalmanProcess` object. Please refer to the
[docs](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/) instead.

### Backend system

!!! note "Backends"
    For more information about the backend system, refer to the documentation
    [here](@ref Backends).

There are three different kind of backends which are [`JuliaBackend`](@ref),
[`WorkerBackend`](@ref), and the HPC cluster backends.

The [`JuliaBackend`](@ref) is the simplest backend. The work done by each
ensemble member is done sequentially.

```@example backend
import ClimaCalibrate

backend = ClimaCalibrate.JuliaBackend()
nothing # hide
```

Next, the [`WorkerBackend`](@ref) is a backend compatible with Distributed.jl.
The work done by each ensemble member is done in parallel on different
processes. This backend is compatible with the Slurm and PBS job schedulers. It
requires starting a job with the resources necessary to start the worker
processes. In the example below, worker processes are being launched by
`addprocs` on a HPC cluster that supports Slurm. You would pass `backend` to
the [`calibrate`](@ref) function.

```julia
import ClimaCalibrate
import Distributed

Distributed.addprocs(ClimaCalibrate.SlurmManager())
backend = ClimaCalibrate.WorkerBackend()
```

Finally, the [`HPCBackend`](@ref) is a backend specfic to each HPC cluster. The
work done by each ensemble member is done in parallel on different jobs. In the
example, each job would start with the `directives`, `modules`, and `env_vars`
listed. The job would last for 720 minutes with single task of 12 CPUs and 1
GPU with regular job priority. The `climacommon` module will be loaded when the
job starts and the environment variables for ClimaComms will be set.

```@example backend
import ClimaCalibrate

backend = ClimaCalibrate.DerechoBackend(;
    directives = [
        :job_priority => "regular",
        :time => 720,
        :ntasks => 1,
        :cpus_per_task => 12,
        :gpus_per_task => 1,
    ],
    modules = ["climacommon"],
    env_vars = ["CLIMACOMMS_CONTEXT" => "SINGLETON", "CLIMACOMMS_DEVICE" => "CUDA"],
)
nothing # hide
```

### Model interface

ClimaCalibrate provides the abstract type [`AbstractModelInterface`](@ref). For
calibration, you will create a struct that will subtype this type and implements
the required interface for this function to work.

The necessary functions are
- `forward_model(interface, iteration, member)` which runs the forward model for
  a single ensemble member.
- `observation_map(interface, iteration)` which processes model output and
  returns a matrix of outputs where each column is the forward model output.
  This matrix is called the `G_ensemble` matrix.

If you want to calibrate using one of the `HPCBackend`s, you also need to
implement
- `model_interface_filepath(interface)` which returns the path to the file that
  defines the model interface.

#### Forward Model

Your forward model must implement the
[`forward_model(interface, iteration, member)`](@ref) function stub.

Since this function only takes in the iteration and member numbers, there are
some hooks to obtain parameters and the output directory:

- [`ClimaCalibrate.Calibration.path_to_ensemble_member`](@ref) returns the
  ensemble member's output directory,

which can be used to set the forward model's output directory.

- [`ClimaCalibrate.Calibration.parameter_path`](@ref) returns the ensemble
member's parameter file, which can be loaded in via TOML or passed to
ClimaParams.

#### Observation map

!!! note "Observational data"
    Observational data generally consists of a vector of observations with
    length `d` and the covariance matrix of the observational noise with size
    `d × d`.

    If you need to stack or sample from observations,
    EnsembleKalmanProcesses.jl's
    [Observation](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/Observations/#Observation) or
    [ObservationSeries](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/Observations/#ObservationSeries) are fully-featured.

    For preprocessing observational data, you want to preprocess for `NaN`s
    and regrid and convert units to match the simulation data and vice versa.

    If you are using `ClimaAnalysis` to preprocess the observational data, then
    you may want to use [`ObservationRecipe`](@ref) to create
    observations from `OutputVar`s.

An **observation map** to process model output and return the full ensemble's
observations is also required.

This is provided by implementing the function stub
[`observation_map(interface, iteration)`](@ref). This function needs to return
an `Matrix` where the `i`th column is the `i`th ensemble member's observational
output. This matrix is called the G ensemble matrix.

Here is a simple readable template for the `observation_map`

```julia
function ClimaCalibrate.observation_map(interface, iteration)
    # This assumes the output_dir is a field of interface
    (; output_dir) = interface
    ekp = JLD2.load_object(ClimaCalibrate.ekp_path(output_dir, iteration))
    ensemble_size = EKP.get_N_ens(ekp)
    G_ensemble = ClimaCalibrate.g_ens_matrix(ekp)
    for member in 1:ensemble_size
        G_ensemble[:, member] = process_member_data(iteration, member)
    end
    return G_ensemble
end
```

Note that each column of the G ensemble matrix should match with the
observations. A common source of error is that the ordering of the variables in
the observations is not the same as the ordering of the variables for the
columns of the G ensemble matrix.

!!! note "GEnsembleBuilder"
    If you are using `ObservationRecipe` to construct your observations and are
    using ClimaAnalysis to postprocess your simulation output, then you might
    want to use [`GEnsembleBuilder`](@ref) which simplifies the construction of the
    G ensemble matrix.

#### Optional postprocessing

It may be the case that `observation_map` is insufficient as you need more
information, such as information from the `ekp` object to compute `G_ensemble`.
Further postprocessing of the `G_ensemble` object can be done by implementing
the `postprocess_g_ensemble` as shown below.

```julia
function ClimaCalibrate.postprocess_g_ensemble(
    interface,
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration
)
    return g_ensemble
end
```

After each evaluation of the observation map and before updating the ensemble,
it may be helpful to print the errors from the `ekp` object or plot
`G_ensemble`. This can be done by implementing the `analyze_iteration` as shown
below.

```julia
function ClimaCalibrate.analyze_iteration(
    interface,
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration,
)
    @info "Analyzing iteration"
    @info "Iteration $iteration"
    @info "Current mean parameter: $(EnsembleKalmanProcesses.get_ϕ_mean_final(prior, ekp))"
    @info "g_ensemble: $g_ensemble"
    @info "output_dir: $output_dir"
    return nothing
end
```

### Parameters

Every parameter that is being calibrated requires a prior distribution to sample from.

EnsembleKalmanProcesses.jl's
[constrained_gaussian](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/API/ParameterDistributions/#EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian)
provides a user-friendly way to constructor Gaussian distributions.

Multiple distributions can be combined using
`combine_distributions(vec_of_distributions)`.

For more information, see the EKP documentation for
[prior distributions](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/).

### Experiment Configuration

A calibration consisting of `m` ensemble members that will run for `n`
iterations. The recommended ensemble size is a function of the chosen method and
the number of parameters being calibrated. See the
[EnsembleKalmanProcesses.jl documentation](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/defaults/#ens-size)
for more information for choosing the appropriate ensemble size.

### Calibrate

Now all of the pieces should be in place:
- forward map
- observation map
- observations
- covariance matrix of the observations (noise)
- prior distribution
- ensemble size
- number of iterations

Lastly, you need to set the output directory and the number of
iterations to run for.

```julia
n_iterations = 7
output_dir = "output/my_experiment"
```
Once all of this has been set up, you can call put it all together using the
[`calibrate`](@ref) function:

```julia
# Construct the EnsembleKalmanProcess object as ekp
ClimaCalibrate.calibrate(
    backend,
    ekp,
    interface,
    n_iterations,
    prior,
    output_dir,
)
```

For more information on parallelizing your calibration, see the
[Backends](@ref Backends) page.

### File structure

For a calibration that ran for a single iteration, the calibration output directory
might look like this.

```
.
├── iteration_001
│   ├── eki_file.jld2
│   ├── G_ensemble.jld2
│   ├── member_001
│   │   ├── checkpoint.txt
│   │   └── parameters.toml
│   ├── member_002
│   │   ├── checkpoint.txt
│   │   └── parameters.toml
│   ├── member_003
│   │   ├── checkpoint.txt
│   │   └── parameters.toml
│   └── prior.jld2
└── iteration_002
    ├── eki_file.jld2
    ├── member_001
    │   └── parameters.toml
    ├── member_002
    │   └── parameters.toml
    └── member_003
        └── parameters.toml
```

Each file in the output directory serves a specific purpose:

- `eki_file.jld2`: The serialized `EnsembleKalmanProcess` state saved **before**
  the iteration runs. For example, `iteration_001/eki_file.jld2` holds the state
  used to generate the parameters for iteration 1.
- `parameters.toml`: Each member's sampled parameter values, written before the
  forward model runs. Load this via TOML or pass it to ClimaParams in your
  `forward_model`.
- `G_ensemble.jld2`: The G ensemble matrix produced by the observation map
  **after** all forward models in the iteration complete.
- `checkpoint.txt`: A flag file written when a member's forward model completes
  successfully, used to skip completed members on restart.
- `prior.jld2`: The prior distribution, saved once in `iteration_001`.

The JLD2 files can be loaded using
[`JLD2`](https://juliaio.github.io/JLD2.jl/stable/).

To access these paths programmatically:
- [`ekp_path(output_dir, iteration)`](@ref): Path to `eki_file.jld2` for the
  given iteration.
- [`parameter_path(output_dir, iteration, member)`](@ref): Path to an ensemble
  member's `parameters.toml`.
- [`path_to_ensemble_member(output_dir, iteration, member)`](@ref): Path to an
  ensemble member's output directory.
- [`path_to_iteration(output_dir, iteration)`](@ref): Path to an iteration's
  directory.

# Checkpointing

ClimaCalibrate checkpoints each forward model and iteration so that an
interrupted calibration can seamlessly pick up where it left off without wasting
resources.

If a calibration (run via `calibrate`) exits after completing an iteration, when
it is restarted it will automatically run the next iteration. This is done by
checking if the ensemble forward map results file (`G_ensemble.jld2`) and the
EKI file (`eki_file.jld2`) have been saved.

If a calibration is interrupted during forward model execution, causing a
partial iteration, incomplete forward models will be rerun when the calibration
is restarted. Completed forward models will not be rerun. This is done by
checking each model's checkpoint file and the flag it contains.

!!! note "Forward model restarts"
    Although the model is checkpointed, this does not mean the forward model
    will automatically restarts. This functionality is delegated to forward
    model.

# Example Calibrations

The [example tutorial](https://clima.github.io/ClimaCalibrate.jl/dev/literate_example/)
provides a clear calibration example that can be run locally using the
[`WorkerBackend`](@ref).

Another example experiment can be found in the package repo under
`experiments/surface_fluxes_perfect_model`.
This experiment uses the
[SurfaceFluxes.jl](https://github.com/CliMA/SurfaceFluxes.jl) package to
generate a physical model that calculates the Monin Obukhov turbulent surface
fluxes based on idealized atmospheric and surface conditions. Since this is a
"perfect model" example, the same model is used to generate synthetic
observations using its default parameters and a small amount of noise. These
synthetic observations are considered to be the ground truth, which is used to
assess the model ensembles' performance when parameters are drawn from the prior
parameter distributions.

It is a perfect-model calibration, using its own output as observational data.
By default, it runs 20 ensemble members for 8 iterations. This example can be
run on the most common backend, the [`JuliaBackend`](@ref), with the following
script:

```julia
import ClimaCalibrate
import EnsembleKalmanProcesses as EKP

include(joinpath(pkgdir(ClimaCalibrate), "experiments", "surface_fluxes_perfect_model", "utils.jl"))
@show ensemble_size n_iterations observation variance prior

# Construct the initial ensemble and EKP object
initial_ensemble = EKP.construct_initial_ensemble(prior, ensemble_size)
ekp = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    observation,
    variance,
    EKP.Inversion(),
    EKP.default_options_dict(EKP.Inversion()),
)

output_dir = "my_experiment"
mkpath(output_dir)
eki = ClimaCalibrate.calibrate(
    JuliaBackend(),
    ekp,
    SurfaceFluxModelInterface(),
    n_iterations,
    prior,
    output_dir,
)

theta_star_vec =
    (; coefficient_a_m_businger = 4.7, coefficient_a_h_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger", "coefficient_a_h_businger"],
    output_dir,
)

g_vs_iter_plot(eki, output_dir)
```

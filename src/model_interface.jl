import EnsembleKalmanProcesses as EKP
import JLD2
import YAML

export AbstractModelInterface,
    forward_model,
    observation_map,
    analyze_iteration,
    postprocess_g_ensemble,
    model_interface_filepath,
    experiment_dir,
    exeflags

"""
    AbstractModelInterface

Abstract supertype for user-defined calibration experiments.

Users subtype this to define their experiment-specific configuration and
dispatch the calibration interface functions.

# Required interface

Subtypes must implement:
- `forward_model(interface, iteration, member)` which runs the forward model for a
  single ensemble member.
- `observation_map(interface, iteration)` which processes model output and
  returns a `G_ensemble` matrix.

To use the `HPCBackend`, the subtypes must also implement:
- `model_interface_filepath(interface)` which returns the path to the file that
  defines the model interface. The `HPCBackend` job script includes this file so
  that all interface functions defined on the subtype are available in the worker
  process.

# Optional interface

- `analyze_iteration(interface, ekp, g_ensemble, prior, output_dir, iteration)`
  which inspects results after each ensemble update. The default implementation
  logs the mean constrained parameters and covariance-weighted error.
- `postprocess_g_ensemble(interface, ekp, g_ensemble, prior, output_dir,
  iteration)` which transforms `g_ensemble` before the ensemble update. The
  default implementation returns `g_ensemble`.

For `HPCBackend`, the subtypes can also implement:
- `experiment_dir(interface)` which returns the Julia project directory passed
  as `--project` to the job script. The default implementation is to return
  `project_dir()`.
- `exeflags(interface)` which returns additional flags (e.g. `--threads 4`)
  passed to the Julia executable in the job script. The default implementation
  is to return the empty string.

# Example

```julia
struct MyModelInterface <: ClimaCalibrate.AbstractModelInterface
    config::String
end

function ClimaCalibrate.forward_model(interface::MyModelInterface, iteration, member)
    # Run the model using interface.config
end

function ClimaCalibrate.observation_map(interface::MyModelInterface, iteration)
    # Read model outputs and return G_ensemble matrix
end
```
"""
abstract type AbstractModelInterface end

"""
    forward_model(interface::AbstractModelInterface, iteration, member)

Execute the forward model simulation with the given configuration.

This function must be overridden by the user's model interface,
dispatching on their subtype of `AbstractModelInterface`.
"""
function forward_model(interface::AbstractModelInterface, iteration, member)
    error("forward_model not implemented for $(nameof(typeof(interface)))")
end

"""
    observation_map(interface::AbstractModelInterface, iteration)

Run the observation map for the specified iteration.
This function must be implemented for each calibration experiment,
dispatching on the user's subtype of `AbstractModelInterface`.
"""
function observation_map(interface::AbstractModelInterface, iteration)
    error("observation_map not implemented for $(nameof(typeof(interface)))")
end

"""
    analyze_iteration(interface::AbstractModelInterface, ekp, g_ensemble, prior, output_dir, iteration)

Analyze results after updating the ensemble and before starting the next iteration.

This function is optional to implement.

For example, one may want to print information from the `ekp` object or plot
`g_ensemble`.
"""
function analyze_iteration(
    ::AbstractModelInterface,
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration,
)
    @info "Mean constrained parameter(s): $(EKP.get_ϕ_mean_final(prior, ekp))"
    @info "Covariance-weighted error: $(last(EKP.get_error(ekp)))"
    return nothing
end

"""
    postprocess_g_ensemble(
        interface::AbstractModelInterface,
        ekp,
        g_ensemble,
        prior,
        output_dir,
        iteration
    )

Postprocess `g_ensemble` after evaluating the observation map and before
updating the ensemble.
"""
function postprocess_g_ensemble(
    ::AbstractModelInterface,
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration,
)
    return g_ensemble
end

"""
    model_interface_filepath(interface::AbstractModelInterface)

Return the path to the file that defines the model interface.

The [`HPCBackend`](@ref) job script includes this file so that all interface
functions defined on the `AbstractModelInterface` subtype (e.g.
[`forward_model`](@ref), [`observation_map`](@ref), and any optional overrides)
are available in the worker process, along with their required packages.
"""
function model_interface_filepath(interface::AbstractModelInterface)
    error(
        "model_interface_filepath not implemented for $(nameof(typeof(interface)))",
    )
end

"""
    experiment_dir(interface::AbstractModelInterface)

Return the path to the experiment's Julia project directory.

The [`HPCBackend`](@ref) uses this to construct the job script command:
```
julia --project=\$experiment_dir \$exeflags -e '...'
```
so that each ensemble member's forward model job runs with the correct project
environment. By default, returns `project_dir()` (the currently active project).

You should override this in your `AbstractModelInterface` subtype if your
experiment lives in a separate project directory.
"""
function experiment_dir(::AbstractModelInterface)
    return project_dir()
end

"""
    exeflags(::AbstractModelInterface)

Return additional flags passed to the Julia executable in the `HPCBackend` job
script.

The [`HPCBackend`](@ref) constructs each ensemble member's job command as:
```
julia --project=\$experiment_dir \$exeflags -e '...'
```
Override this in your `AbstractModelInterface` subtype to pass extra flags such
as `--threads` or `-O0`. By default, returns `""` (no extra flags).
"""
function exeflags(::AbstractModelInterface)
    return ""
end

"""
    _load(filename)

Load the object with the name `filename`.

This is used by calibration with the `HPCBackend` to support loading the
model interface object.
"""
_load(filename) = JLD2.load_object(filename)

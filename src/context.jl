"""
    AbstractCalibrationContext

Abstract supertype for user-defined calibration experiments.

Users subtype this to define their experiment-specific configuration and
dispatch the calibration interface functions.

# Required interface

Subtypes must implement:
- `forward_model(ctx, iteration, member)` which runs the forward model for a
  single ensemble member.
- `observation_map(ctx, iteration)` which processes model output and,
  returns a `G_ensemble` matrix.

# Optional interface (have defaults)

- `analyze_iteration(ctx, ekp, g_ensemble, prior, iteration)` — inspect
  results after each ensemble update. The default implementation logs the mean
  constrained parameters and covariance-weighted error.
- `postprocess_g_ensemble(ctx, ekp, g_ensemble, prior, iteration)` — transform
  `g_ensemble` before the ensemble update. The default implementation returns
  `g_ensemble`.

# Example

```julia
struct MyExperiment <: ClimaCalibrate.AbstractCalibrationContext
    data_file::String
end

function ClimaCalibrate.forward_model(ctx::MyExperiment, iteration, member)
    # Run the model using ctx.data_file
end

function ClimaCalibrate.observation_map(ctx::MyExperiment, iteration)
    # Read model outputs and return G_ensemble matrix
end
```
"""
abstract type AbstractCalibrationContext end

# Notes
# ctx would be static across all iterations and members
# A better name for this struct could be AbstractModelInterface since that is
# what it is

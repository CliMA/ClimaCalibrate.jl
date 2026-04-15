"""
    AbstractCalibrationContext

Abstract supertype for user-defined calibration experiments.

Users subtype this to define their experiment-specific configuration and
dispatch the calibration interface functions. This solves two problems:

1. **No type piracy**: dispatching on a user-owned subtype avoids overriding
   unowned method signatures like `forward_model(::Int, ::Int)`.
2. **Clean separation of concerns**: the context holds experiment configuration
   (things fixed for the entire calibration), while changing state like
   `iteration`, `member`, and `ekp` are passed as function arguments.

# Required interface

Subtypes must implement:
- `forward_model(ctx, iteration, member)` — run the forward model for one
  ensemble member.
- `observation_map(ctx, iteration)` — map model output to observation space,
  returning a `G_ensemble` matrix.

# Optional interface (have defaults)

- `analyze_iteration(ctx, ekp, g_ensemble, prior, iteration)` — inspect
  results after each ensemble update. Default: logs mean parameters and error.
- `postprocess_g_ensemble(ctx, ekp, g_ensemble, prior, iteration)` — transform
  `g_ensemble` before the ensemble update. Default: identity (returns
  `g_ensemble` unchanged).

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

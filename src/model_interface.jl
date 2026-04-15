import EnsembleKalmanProcesses as EKP
import YAML

export forward_model, observation_map, analyze_iteration

"""
    forward_model(ctx::AbstractCalibrationContext, iteration, member)

Execute the forward model simulation with the given configuration.

This function must be overridden by the user's model interface,
dispatching on their subtype of `AbstractCalibrationContext`.
"""
function forward_model(ctx::AbstractCalibrationContext, iteration, member)
    error("forward_model not implemented for $(nameof(typeof(ctx)))")
end

"""
    observation_map(ctx::AbstractCalibrationContext, iteration)

Runs the observation map for the specified iteration.
This function must be implemented for each calibration experiment,
dispatching on the user's subtype of `AbstractCalibrationContext`.
"""
function observation_map(ctx::AbstractCalibrationContext, iteration)
    error("observation_map not implemented for $(nameof(typeof(ctx)))")
end

"""
    analyze_iteration(ctx::AbstractCalibrationContext, ekp, g_ensemble, prior, iteration)

After updating the ensemble and before starting the next iteration,
`analyze_iteration` is evaluated.

This function is optional to implement.

For example, one may want to print information from the `eki` object or plot
`g_ensemble`.
"""
function analyze_iteration(
    ctx::AbstractCalibrationContext,
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
    postprocess_g_ensemble(ctx::AbstractCalibrationContext, ekp, g_ensemble, prior, iteration)

Postprocess `g_ensemble` after evaluating the observation map and before
updating the ensemble.
"""
function postprocess_g_ensemble(
    ctx::AbstractCalibrationContext,
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration,
)
    return g_ensemble
end

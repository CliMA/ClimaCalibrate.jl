import EnsembleKalmanProcesses as EKP
import YAML

export forward_model, observation_map, analyze_iteration

"""
    forward_model(iteration, member)

Execute the forward model simulation with the given configuration.

This function must be overridden by a component's model interface and 
should set things like the parameter path and other member-specific settings.
"""
function forward_model(ctx)
    error("forward_model not implemented")
end

"""
    observation_map(iteration)

Runs the observation map for the specified iteration.
This function must be implemented for each calibration experiment.
"""
function observation_map(ctx)
    error("observation_map not implemented")
end

"""
    analyze_iteration(ekp, g_ensemble, prior, output_dir, iteration)

After updating the ensemble and before starting the next iteration,
`analyze_iteration` is evaluated.

This function is optional to implement.

For example, one may want to print information from the `eki` object or plot
`g_ensemble`.
"""
function analyze_iteration(ctx, g_ensemble)
    (; prior, ekp) = ctx
    @info "Mean constrained parameter(s): $(EKP.get_ϕ_mean_final(prior, ekp))"
    @info "Covariance-weighted error: $(last(EKP.get_error(ekp)))"
    return nothing
end

"""
    postprocess_g_ensemble(ekp, g_ensemble, prior, output_dir, iteration)

Postprocess `g_ensemble` after evaluating the observation map and before
updating the ensemble.
"""
function postprocess_g_ensemble(ctx, g_ensemble)
    return g_ensemble
end

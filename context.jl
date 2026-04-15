# ClimaCalibrate.parameter_path should be overwritten to take in a CalibrationContext

# Questions
# Why do we want the user to be able to access ekp, prior, and ensemble_size?
# This is used in analyze_iteration
# Passing the EKP object is annoying since it is not updated in place and it is
# saved and reloaded in different locations

# How do we avoid type piracy when the user are told to override the functions?
# One solution is to provide an abstract type, tell the user to subtype it and
# implement x, y, and z

abstract type AbstractCalibrationContext end

"""
    CalibrationContext

A struct passed to user defined functions for calibration.
"""
struct CalibrationContext{CONFIG}
    iteration::Int
    # TODO: I don't like member since member is only relevant for the forward model
    # and nothing else
    member::Int
    output_dir::String
    # TODO: This is a poor man inheritance...
    user_config::CONFIG
end

# Users would do something like

struct OwnCalibrationContext
    # stuff
end

# args is something that is supplied to the calibration
OwnCalibrationContext(iteration, member, output_dir, args)

forward_model(context::AbstractCalibrationContext)
observation_map(context)
analyze_iteration(ekp, g_ensemble, context::AbstractCalibrationContext)
postprocess_g_ensemble(
    ekp,
    g_ensemble,
    prior,
    context::AbstractCalibrationContext,
)

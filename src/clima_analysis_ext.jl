# Function stubs for ext/ClimaAnalysisExt.jl

function seasonal_average end


"""
    construct_seasonal_noise_covariance(var::OutputVar; replace_nans = true, model_error_scale = nothing, regularization = nothing)

Construct an `EKP.SVDplusD` noise covariance from the seasonal average values of `var`
"""
function construct_seasonal_noise_covariance end


"""
    construct_noise_covariance(var::OutputVar, model_error_scale = 0.05; replace_nans = true, regularization = nothing)

Construct an `EKP.SVDplusD` noise covariance from `var`.

# Arguments
- `model_error_scale`: Add a model error term to the diagonal term: `(model_error_scale * average_time(var))^2`
- `replace_nans`: Replace NaNs in the `var`.
- `regulation`: Add a regularization term to the diagonal: `regularization * EKP.I`
"""
function construct_noise_covariance end

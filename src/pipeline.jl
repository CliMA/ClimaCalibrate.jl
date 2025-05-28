module Pipeline

abstract type AbstractSample end

struct SeasonalSample <: AbstractSample
    ignore_nan_in_average::Bool
    ignore_nan_in_sample::Bool
end

abstract type AbstractCovariance end

struct NoiseCovariance{FT1 <: AbstractFloat, FT2 <: AbstractFloat} <:
       AbstractCovariance
    model_error_scale::Union{Nothing, FT1}
    regularization::Union{Nothing, FT2}

    """All NaNs are ignored when computing the covariance"""
    ignore_nan::Bool
end

# TODO: Not sure about this name because EKP uses SVDplusD
struct SVDPlusDCovariance{FT1 <: AbstractFloat, FT2 <: AbstractFloat} <:
       AbstractCovariance
    model_error_scale::Union{Nothing, FT1}
    replace_nans::Bool
    regulaization::Union{Nothing, FT2}
end

function sample end

function covariance end

function observation end

function reconstruct_var_from_obs end

end

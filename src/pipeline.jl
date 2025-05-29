module Pipeline

abstract type AbstractSample end

Base.@kwdef struct SeasonalSample <: AbstractSample
    ignore_nan_in_average::Bool = true
    ignore_nan_in_sample::Bool = true
end

abstract type AbstractCovariance end

Base.@kwdef struct NoiseCovariance{
    FT1 <: AbstractFloat,
    FT2 <: AbstractFloat,
} <: AbstractCovariance
    model_error_scale::FT1 = 0.0
    regularization::FT2 = 0.0

    """All NaNs are ignored when computing the covariance"""
    ignore_nan::Bool = true
end

# TODO: Not sure about this name because EKP uses SVDplusD
Base.@kwdef struct SVDPlusDCovariance{
    FT1 <: AbstractFloat,
    FT2 <: AbstractFloat,
} <: AbstractCovariance
    model_error_scale::FT1 = 0.0
    replace_nans::Bool = true
    regularization::FT2 = 0.0
end

function sample end

function covariance end

function observation end

function reconstruct_var_from_obs end

end



# TODO: Once the functions for reconstruct is finalized, update this

# Note that we do not include NaNStatistics and Statistics because those
# packages are automatically loaded by ClimaAnalysis and it does not matter how
# deep in the stack the packages are
extension_fns = [
    :ClimaAnalysis =>
        [:sample, :covariance, :observation, :reconstruct_var_from_obs],
    :LinearAlgebra =>
        [:sample, :covariance, :observation, :reconstruct_var_from_obs],
]

"""
    is_pkg_loaded(pkg::Symbol)

Check if `pkg` is loaded or not.
"""
function is_pkg_loaded(pkg::Symbol)
    return any(k -> Symbol(k.name) == pkg, keys(Base.loaded_modules))
end

function __init__()
    # Register error hint if a package is not loaded
    if isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(
            MethodError,
        ) do io, exc, _argtypes, _kwargs
            for (pkg, fns) in extension_fns
                if Symbol(exc.f) in fns && !is_pkg_loaded(pkg)
                    if pkg == :ClimaAnalysis
                        print(
                            io,
                            "\nImport ClimaAnalysis to enable `$(exc.f)`.";
                        )
                    elseif pkg == :LinearAlgebra
                        print(
                            io,
                            "\nImport LinearAlgebra to enable `$(exc.f)`.";
                        )
                    end
                end
            end
        end
    end
end

# Function stubs for ext/ClimaAnalysisExt.jl
import Statistics: mean
import LinearAlgebra: Diagonal

import ClimaCalibrate
using ClimaAnalysis
import EnsembleKalmanProcesses as EKP

"""
    limit_pressure_dim_to_era5_range(output_var)

Limit the pressure dimension of `output_var` to the ERA5 range (100.0 - 100_000.0).
"""
function limit_pressure_dim_to_era5_range end

"""
    limit_pressure_dim(output_var, min_pressure, max_pressure)

Limit the pressure dimension of `output_var` between `min_pressure` and `max_pressure`.
"""
function limit_pressure_dim end

"""
    get_monthly_averages(simdir, var_name)

Extract monthly averages of the given `var_name` from `simdir`.
"""
function get_monthly_averages end

"""
    seasonally_aligned_yearly_average(var, yr)

Return the yearly average of `var` for the specified year `yr`, using a seasonal 
alignment from December of the previous year to November of the given year.
"""
function seasonally_aligned_yearly_average end

"""
    seasonal_covariance(output_var; model_error_scale = nothing, regularization = nothing)

Computes the diagonal covariance matrix of seasonal averages of `output_var`.

# Arguments
- `output_var`: Climate variable data (OutputVar or similar)
- `model_error_scale`: Optional scaling factor for model error, applied as a fraction of the mean
- `regularization`: Optional regularization term added to variance values
"""
function seasonal_covariance end

"""
    tsvd_covariance(var,
        model_error_scale = 0.05;
        replace_nans = true,
        regularization = nothing
    )

Return an `EKP.SVDplusD` (truncated SVD plus Diagonal) covariance structure for a given `var`, 
adding a model error term scaled by `model_error_scale` to the diagonal.

# Arguments
- `var`: An OutputVar with a `time` dimension
- `model_error_scale`: The coefficient for the model error term in the diagonal.
- `replace_nans`: Toggle replacing NaNs in the `var`
- `regularization`: If not nothing, add a flat regularization term to the diagonal.
"""
function tsvd_covariance end

"""
    year_of_seasonal_averages(output_var, yr)

Compute seasonal averages for a specific year `yr` from the `output_var`.
"""
function year_of_seasonal_averages end

"""
    year_of_seasonal_observations(output_var, yr)

Create an `EKP.Observation` for a specific year `yr` from the `output_var`.
"""
function year_of_seasonal_observations end

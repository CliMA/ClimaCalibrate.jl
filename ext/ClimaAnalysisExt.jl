module ClimaAnalysisExt

import Statistics: mean
import LinearAlgebra: Diagonal

import ClimaCalibrate
using ClimaAnalysis
import EnsembleKalmanProcesses as EKP

"""
    limit_pressure_dim_to_era5_range(output_var)

Limit the pressure dimension of `output_var` to the ERA5 range (100.0 - 100_000.0).
"""
ClimaCalibrate.limit_pressure_dim_to_era5_range(v) =
    limit_pressure_dim(v, 100.0, 100_000.0)

"""
    limit_pressure_dim(output_var, min_pressure, max_pressure)

Limit the pressure dimension of `output_var` between `min_pressure` and `max_pressure`.
"""
function ClimaCalibrate.limit_pressure_dim(
    output_var,
    min_pressure,
    max_pressure,
)
    @assert has_pressure(output_var)
    pressure_dims = output_var.dims[pressure_name(output_var)]
    valid_pressure_levels = filter(pressure_dims) do pressure
        min_pressure <= pressure <= max_pressure
    end
    lowest_valid_level = minimum(valid_pressure_levels)
    highest_valid_level = maximum(valid_pressure_levels)

    return window(
        output_var,
        "pfull";
        left = lowest_valid_level,
        right = highest_valid_level,
    )
end

"""
    get_monthly_averages(simdir, var_name)

Extract monthly averages of the given `var_name` from `simdir`.
"""
get_monthly_averages(simdir, var_name) =
    get(simdir; short_name = var_name, reduction = "average", period = "1M")

"""
    seasonally_aligned_yearly_average(var, yr)

Return the yearly average of `var` for the specified year `yr`, using a seasonal 
alignment from December of the previous year to November of the given year.
"""
function ClimaCalibrate.seasonally_aligned_yearly_average(var, yr)
    year_window = window(
        var,
        "time";
        left = DateTime(yr - 1, 12, 1),
        right = DateTime(yr, 11, 30),
    )
    return average_time(year_window)
end

"""
    seasonal_covariance(output_var; model_error_scale = nothing, regularization = nothing)

Computes the diagonal covariance matrix of seasonal averages of `output_var`.

# Arguments
- `output_var`: Climate variable data (OutputVar or similar)
- `model_error_scale`: Optional scaling factor for model error, applied as a fraction of the mean
- `regularization`: Optional regularization term added to variance values
"""
function ClimaCalibrate.seasonal_covariance(
    output_var;
    model_error_scale = nothing,
    regularization = nothing,
)
    seasonal_averages = average_season_across_time(output_var)
    variance_per_season = map(split_by_season(seasonal_averages)) do season
        variance = flatten(variance_time(season)).data
        if !isnothing(model_error_scale)
            variance .+=
                (model_error_scale .* flatten(average_time(season)).data) .^ 2
        end
        return variance
    end
    diag_cov = vcat(variance_per_season...)
    !isnothing(regularization) && (diag_cov .+= regularization)
    return Diagonal(diag_cov)
end

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
function ClimaCalibrate.tsvd_covariance(
    var::OutputVar,
    model_error_scale = 0.05;
    replace_nans = true,
    regularization = nothing,
)
    ClimaAnalysis.has_time(var) || error("OutputVar missing time dimension.")
    replace_nans && (
        var = ClimaAnalysis.replace(
            var,
            (NaN => mean(filter(!isnan, var.data))),
        )
    )
    # Reshape `var` into a matrix `var_mat` where each column is a time slice
    var_vec = map(times(var)) do t
        flatten(slice(var, time = t)).data
    end
    var_mat = hcat(var_vec...)

    gamma_low_rank = EKP.tsvd_cov_from_samples(var_mat)

    gamma_diag = (model_error_scale * flatten(average_time(var)).data) .^ 2
    if !isnothing(regularization)
        gamma_diag += regularization * EKP.I
    end

    return EKP.SVDplusD(gamma_low_rank, Diagonal(gamma_diag))
end

"""
    year_of_seasonal_averages(output_var, yr)

Compute seasonal averages for a specific year `yr` from the `output_var`.
"""
function ClimaCalibrate.year_of_seasonal_averages(output_var, yr)
    seasonal_averages = average_season_across_time(output_var)
    season_and_years =
        map(ClimaAnalysis.Utils.find_season_and_year, dates(seasonal_averages))
    indices = findall(s -> s[2] == yr, season_and_years)
    isempty(indices) && error(
        "No data found in $(long_name(output_var)) for the given year: $yr",
    )
    min_idx, max_idx = extrema(indices)
    left = dates(seasonal_averages)[min_idx]
    right = dates(seasonal_averages)[max_idx]
    return window(seasonal_averages, "time"; left, right)
end

"""
    year_of_seasonal_observations(output_var, yr)

Create an `EKP.Observation` for a specific year `yr` from the `output_var`.
"""
function ClimaCalibrate.year_of_seasonal_observations(output_var, yr)
    seasonal_averages = year_of_seasonal_averages(output_var, yr)
    # Split into four OutputVars to get the same format as the covariance matrix
    obs_vec = flatten_seasonal_averages(seasonal_averages)
    obs_cov = seasonal_covariance(
        output_var;
        model_error_scale = 0.05,
        regularization = 1e-3,
    )
    name = get(
        output_var.attributes,
        "CF_name",
        get(output_var.attributes, "long_name", ""),
    )
    return EKP.Observation(obs_vec, obs_cov, "$(yr)_$name")
end

flatten_seasonal_averages(seasonal_averages) =
    vcat(map(x -> flatten(x).data, split_by_season(seasonal_averages))...)

end

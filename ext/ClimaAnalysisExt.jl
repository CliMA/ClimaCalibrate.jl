module ClimaAnalysisExt

import Statistics: mean
import LinearAlgebra: Diagonal

import ClimaCalibrate
using ClimaAnalysis
import EnsembleKalmanProcesses as EKP

function ClimaCalibrate.construct_seasonal_noise_covariance(
    var::OutputVar;
    replace_nans = true,
    model_error_scale = nothing,
    regularization = nothing,
)
    seasonal_averages = seasonal_average(var)
    return construct_noise_covariance(seasonal_averages)
end

function ClimaCalibrate.construct_noise_covariance(
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

end

__precompile__(false)
module CESExt

import CalibrateEmulateSample as CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import JLD2

import ClimaCalibrate as CAL


function CAL.get_input_output_pairs(ekp; N_iterations = nothing)
    N_iterations = isnothing(N_iterations) ? length(ekp.g) : N_iterations
    input_output_pairs = CES.Utilities.get_training_points(ekp, N_iterations)
    return input_output_pairs
end

function CAL.gp_emulator(input_output_pairs, obs_noise_cov)
    gppackage = GPJL()
    gauss_proc = GaussianProcess(gppackage, noise_learn = false)
    emulator = Emulator(gauss_proc, input_output_pairs; obs_noise_cov)
    optimize_hyperparameters!(emulator)
    return emulator
end

function CAL.sample(
    emulator,
    y_obs,
    prior,
    init_params;
    n_samples = 100_000,
    init_stepsize = 0.1,
    discard_initial = 0,
)
    mcmc = MCMCWrapper(RWMHSampling(), y_obs, prior, emulator; init_params)
    new_step = optimize_stepsize(mcmc; init_stepsize, N = 2000, discard_initial)
    chain = MarkovChainMonteCarlo.sample(
        mcmc,
        n_samples;
        stepsize = new_step,
        discard_initial = 0,
    )
    return (; mcmc, chain)
end

function CAL.save_posterior(mcmc, chain; filename = "samples.jld2")
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
    JLD2.save_object(filename, posterior)
    return posterior
end

end

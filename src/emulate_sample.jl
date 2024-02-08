import CalibrateEmulateSample as CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface

import JLD2

"""
    get_input_output_pairs(ekp)

Helper function for getting the input/output pairs from an EKP object.
"""
function get_input_output_pairs(ekp)
    N_iterations = length(ekp.g)
    input_output_pairs = CES.Utilities.get_training_points(ekp, N_iterations)
    return input_output_pairs
end

"""
    gp_emulator(input_output_pairs, obs_noise_cov)

Constructs a gaussian process emulator from the given input/output pairs and noise.
"""
function gp_emulator(input_output_pairs, obs_noise_cov)
    gppackage = Emulators.GPJL()
    gauss_proc = Emulators.GaussianProcess(gppackage, noise_learn = false)
    emulator = Emulator(gauss_proc, input_output_pairs; obs_noise_cov)
    optimize_hyperparameters!(emulator)
    return emulator
end

"""
    sample(emulator, y_obs, prior, init_params; n_samples = 100_000)

Constructs a MarkovChainMonteCarlo object, optimizes its stepsize, and takes
`n_samples` number of samples. 
Returns both the MCMC object and the samples in a NamedTuple.
"""
function sample(emulator, y_obs, prior, init_params; n_samples = 100_000)
    mcmc = MCMCWrapper(RWMHSampling(), y_obs, prior, emulator; init_params)
    new_step = optimize_stepsize(
        mcmc;
        init_stepsize = 0.1,
        N = 2000,
        discard_initial = 0,
    )
    chain = MarkovChainMonteCarlo.sample(
        mcmc,
        n_samples;
        stepsize = new_step,
        discard_initial = 0,
    )
    return (; mcmc, chain)
end

"""
    save_samples(mcmc, chain, prior; filename = "samples.jld2")

Given an MCMC object, a list of samples, and a prior distribution, transforms the samples
into constrained space and saves them to a JLD2 file.
"""
function save_samples(mcmc, chain, prior; filename = "samples.jld2")
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
    constrained_posterior = Emulators.transform_unconstrained_to_constrained(
        prior,
        MarkovChainMonteCarlo.get_distribution(posterior),
    )
    JLD2.save_object(filename, constrained_posterior)
    return constrained_posterior
end

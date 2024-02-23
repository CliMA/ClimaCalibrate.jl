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
function get_input_output_pairs(ekp; N_iterations = nothing)
    N_iterations = isnothing(N_iterations) ? length(ekp.g) : N_iterations
    input_output_pairs = CES.Utilities.get_training_points(ekp, N_iterations)
    return input_output_pairs
end

"""
    gp_emulator(input_output_pairs, obs_noise_cov)

Constructs a gaussian process emulator from the given input/output pairs and noise.
"""
function gp_emulator(input_output_pairs, obs_noise_cov)
    gppackage = GPJL()
    gauss_proc = GaussianProcess(gppackage, noise_learn = false)
    emulator = Emulator(gauss_proc, input_output_pairs; obs_noise_cov)
    optimize_hyperparameters!(emulator)
    return emulator
end

"""
    sample(
        emulator, 
        y_obs,
        prior, 
        init_params; 
        n_samples = 100_000, 
        init_stepsize = 0.1, 
        discard_initial = 0
    )

Constructs a MarkovChainMonteCarlo object, optimizes its stepsize, and takes
`n_samples` number of samples. 
The initial stepsize can be specified by `init_stepsize`, 
and the number of initial samples to discard can be set by `discard_initial`.
Returns both the MCMC object and the samples in a NamedTuple.
"""
function sample(
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

"""
    save_posterior(mcmc, chain; filename = "samples.jld2")

Given an MCMC object, a Markov chain of samples, and a prior distribution, 
constructs the posterior distribution and saves it to `filename`. 
Returns the samples in constrained (physical) parameter space.
"""
function save_posterior(mcmc, chain; filename = "samples.jld2")
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
    constrained_posterior = transform_unconstrained_to_constrained(
        posterior,
        MarkovChainMonteCarlo.get_distribution(posterior),
    )
    JLD2.save_object(filename, posterior)
    return constrained_posterior
end

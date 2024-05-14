# Function stubs for the CalibrateEmulateSample extension

extension_not_loaded_err() = error(
    "CES extension not loaded. Import CalibrateEmulateSample or run `Base.retry_load_extensions()`",
)

"""
    get_input_output_pairs(ekp)

Helper function for getting the input/output pairs from an EKP object.
"""
get_input_output_pairs(ekp; N_iterations) = extension_not_loaded_err()

"""
    gp_emulator(input_output_pairs, obs_noise_cov)

Constructs a gaussian process emulator from the given input/output pairs and noise.
"""
gp_emulator(input_output_pairs, obs_noise_cov) = extension_not_loaded_err()

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
sample(
    emulator,
    y_obs,
    prior,
    init_params;
    n_samples,
    init_stepsize,
    discard_initial,
) = extension_not_loaded_err()

"""
    save_posterior(mcmc, chain; filename = "samples.jld2")

Given an MCMC object, a Markov chain of samples, and a prior distribution, 
constructs the posterior distribution and saves it to `filename`. 
Returns the samples in constrained (physical) parameter space.
"""
save_posterior(mcmc, chain; filename) = extension_not_loaded_err()

# Emulate and Sample
Once you have run a successful calibration, we can fit an emulator to the resulting input/output pairs.

First, import the necessary packages:
```julia
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface

import JLD2
import CalibrateAtmos as CAL
```

Next, load in the data, EKP object, and prior distribution. These values are taken
from the perfect model experiment with experiment ID `sphere_held_suarez_rhoe_equilmoist`.
```julia
y_obs = [261.5493]
y_noise_cov = [0.02619;;]
ekp = JLD2.load_object(
    joinpath(
        pkgdir(CAL),
        "docs",
        "src",
        "assets",
        "eki_file_for_emulate_example.jld2",
    ),
)
init_params = [EKP.get_u_final(ekp)[1]]

prior_path = joinpath(
    pkgdir(CAL),
    "experiments",
    "sphere_held_suarez_rhoe_equilmoist",
    "prior.toml",
)

prior = CAL.get_prior(prior_path)
```
Get the input-output pairs which will be used to train the emulator. 
The inputs are the parameter values, and the outputs are the result of the observation map. 
In thise case, the outputs are the average air temperature at roughly 500 meters.
```julia
input_output_pairs = CAL.get_input_output_pairs(ekp)
```
Next, create the Gaussian Process-based emulator and Markov chain. 
The samples from the chain can be used in future predictive model runs with the same configuration.
The posterior distribution can be saved to a JLD2 file using `save_posterior`. Samples can be extracted from the posterior using ClimaParams.
```julia
emulator = CAL.gp_emulator(input_output_pairs, y_noise_cov)
(; mcmc, chain) = CAL.sample(emulator, y_obs, prior, init_params)
constrained_posterior = CAL.save_posterior(mcmc, chain; filename = "samples.jld2")
```

Finally, you can plot the prior and posterior distributions to see results:
```julia
using Plots
plot(prior)
posterior = get_posterior(mcmc, chain)
plot!(posterior)
vline!([65.0])
```

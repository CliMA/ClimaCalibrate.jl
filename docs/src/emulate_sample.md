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

Next, load in the data, EKP object, and prior distribution.
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
Get the input-output pairs which will be used to train the emulator:
```julia
input_output_pairs = CAL.get_input_output_pairs(ekp)
```
Create the Gaussian Process-based emulator, obtain samples, and save the samples to a JLD2 file.
```julia
emulator = CAL.gp_emulator(input_output_pairs, y_noise_cov)
(; mcmc, chain) = CAL.sample(emulator, y_obs, prior, init_params)
constrained_posterior = CAL.save_samples(mcmc, chain, prior; filename = "samples.jld2")
```

Finally, you can plot the prior and posterior distributions to see results:
```julia
using Plots
plot(prior)
posterior = get_posterior(mcmc, chain)
plot!(posterior)
vline!([65.0])
```

```@example
include("emulate_sample_example.jl")
```
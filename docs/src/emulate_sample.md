# Emulate and Sample
Once you have run a successful calibration, we can fit an emulator to the resulting input/output pairs.

First, import the necessary packages:
```julia
import JLD2

using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface

import ClimaCalibrate as CAL
```

Next, load in the data, EKP object, and prior distribution. These values are taken
from the Held-Suarez perfect model experiment in ClimaAtmos.

```julia
asset_path = joinpath(
    pkgdir(CAL),
    "docs",
    "src",
    "assets")

ekp = JLD2.load_object(joinpath(asset_path, "emulate_example_ekiobj.jld2"))
y_obs = ekp.obs_mean
y_noise_cov = ekp.obs_noise_cov
initial_params = [EKP.get_u_final(ekp)[1]]

prior_path = joinpath(asset_path, "emulate_example_prior.toml")
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
(; mcmc, chain) = CAL.sample(emulator, y_obs, prior, initial_params)
constrained_posterior = CAL.save_posterior(mcmc, chain; filename = "samples.jld2")
display(chain)
```

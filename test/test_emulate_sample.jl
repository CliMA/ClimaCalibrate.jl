import JLD2
import Statistics: mean

using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface

import CalibrateAtmos as CAL

y_obs = [261.5493]
y_noise_cov = [0.02619;;]
ekp = JLD2.load_object(joinpath("test_case_inputs", "eki_test.jld2"))
init_params = [EKP.get_u_final(ekp)[1]]

prior_path = joinpath("test_case_inputs", "sphere_hs_rhoe.toml")

prior = CAL.get_prior(prior_path)

input_output_pairs = CAL.get_input_output_pairs(ekp)

@test input_output_pairs.inputs.stored_data ==
      hcat([ekp.u[i].stored_data for i in 1:(length(ekp.u) - 1)]...)
@test input_output_pairs.outputs.stored_data ==
      hcat([ekp.g[i].stored_data for i in 1:length(ekp.g)]...)

emulator = CAL.gp_emulator(input_output_pairs, y_noise_cov)


(; mcmc, chain) = CAL.sample(emulator, y_obs, prior, init_params)
@test mean(chain.value[1:100000]) ≈ 4.19035299 rtol = 0.0001

constrained_posterior = CAL.save_posterior(mcmc, chain)

# generate_data: generate true y, noise and x_inputs

import Random
using ClimaCalibrate

pkg_dir = pkgdir(ClimaCalibrate)
# The path comes from the package, not from `Base.active_project()`: this file
# is included from whichever environment the caller is working in
experiment_path =
    joinpath(pkg_dir, "experiments", "surface_fluxes_perfect_model")
data_path = joinpath(experiment_path, "data")
include(joinpath(experiment_path, "model_interface.jl"))
FT = Float32

"""
    generate_profiles(FT)

Generate a set of test profiles for the surface fluxes model. Here we want the profiles to be
statically stable to be sensitive to the input parameters, `a_m` and `a_b`.
"""
function generate_profiles(FT)
    profiles_sfc = []
    profiles_int = []
    struct_to_nt(s) =
        NamedTuple{propertynames(s)}(map(x -> getfield(s, x), propertynames(s)))
    for i in 1:20
        push!(
            profiles_sfc,
            struct_to_nt(TestAtmosProfile{FT}(T = FT(280 + i / 10), z = FT(0))),
        )
        push!(
            profiles_int,
            struct_to_nt(
                TestAtmosProfile{FT}(T = FT(280.2 + i / 10), z = FT(10)),
            ),
        )
    end

    return profiles_sfc, profiles_int
end
Base.@kwdef mutable struct TestAtmosProfile{FT}
    u::FT = FT(2)
    v::FT = FT(0)
    ρ::FT = FT(1)
    q::FT = FT(0.001)
    T::FT = FT(300)
    z::FT = FT(0)
end

"""
    save_profiles(FT; data_path = "data", x_data_file = "data/surface_fluxes_test_data.jld2")

Save the generated profiles to file.
"""
function save_profiles(
    FT;
    data_path = "data",
    x_data_file = "data/surface_fluxes_test_data.jld2",
)

    mkpath(data_path)

    profiles_sfc, profiles_int = generate_profiles(FT)

    data = Dict(
        "profiles_sfc" => profiles_sfc[1:end],
        "profiles_int" => profiles_int[1:end],
    )
    JLD2.save(joinpath(data_path, x_data_file), data)
end

"""
    NOISE_FRACTION

The standard deviation of the observation error, as a fraction of the observed
`ustar`.

An instrument reading of a turbulent flux carries an error of a few percent, and
the calibration needs to be told what that error is: `EnsembleKalmanProcesses`
weights the model-data misfit by it, and takes a step whose size it sets. What
it is told is this error together with the model error, since the misfit
contains both.
"""
const NOISE_FRACTION = 0.02

"""
    synthetic_observed_y(x_inputs; data_path = "data", rng = Random.default_rng())

Run the model with its default parameters, and return the `ustar` of each
profile along with a noisy observation of their mean and the variance of that
noise.

The observation is one number drawn from `N(mean(ustar), (0.02 mean(ustar))^2)`,
which is the perfect-model setup: the truth is known, and the calibration sees
it through an instrument.
"""
function synthetic_observed_y(
    x_inputs;
    data_path = "data",
    rng = Random.MersenneTwister(1234),
)
    FT = typeof(x_inputs.profiles_int[1].T)
    config = Dict()
    config["toml"] = []
    config["output_dir"] = data_path
    ustar = obtain_ustar(
        FT,
        x_inputs,
        config,
        return_ustar = true,
        model_error = false,
    )

    truth = nanmean(ustar)
    noise_sd = NOISE_FRACTION * truth
    observation = Float64[truth + noise_sd * randn(rng)]
    variance = Matrix{Float64}(undef, 1, 1)
    variance[1] = noise_sd^2 + (MODEL_ERROR_FRACTION * truth)^2

    JLD2.save_object(joinpath(data_path, "synthetic_ustar_array.jld2"), ustar)
    return (; ustar, observation, variance)
end

data_files = [
    joinpath(data_path, "obs_mean.jld2")
    joinpath(data_path, "obs_noise_cov.jld2")
]
if any(x -> !isfile(x), data_files)

    profile_file = "synthetic_profile_data.jld2"
    save_profiles(FT, data_path = data_path, x_data_file = profile_file)

    x_inputs = load_profiles(joinpath(data_path, profile_file))
    (; observation, variance) =
        synthetic_observed_y(x_inputs, data_path = data_path)

    JLD2.save_object(joinpath(data_path, "obs_mean.jld2"), observation)
    JLD2.save_object(joinpath(data_path, "obs_noise_cov.jld2"), variance)
end

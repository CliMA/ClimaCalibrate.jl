# generate_truth: generate true y, noise and x_inputs
experiment_id = "surface_fluxes_perfect_model"

import SurfaceFluxes as SF
import SurfaceFluxes.Parameters as SFPP
import SurfaceFluxes.UniversalFunctions as UF
import Thermodynamics as TD
using YAML
import SurfaceFluxes.Parameters: SurfaceFluxesParameters
using CalibrateAtmos

pkg_dir = pkgdir(CalibrateAtmos)
experiment_path = "$pkg_dir/experiments/$experiment_id"
data_path = "$experiment_path/data"
include("$experiment_path/model_interface.jl")
include("$experiment_path/observation_map.jl")

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
    synthetic_observed_y(x_inputs; data_path = "data")

Generate synthetic observed y from the model truth.
"""
function synthetic_observed_y(x_inputs; data_path = "data", apply_noise = false)
    FT = typeof(x_inputs.profiles_int[1].T)
    config = YAML.load_file("$experiment_path/model_config.yml")
    config["output_dir"] = data_path
    y = obtain_ustar(FT, x_inputs, config, return_ustar = true)
    if apply_noise
        # add noise to model truth to generate synthetic observations with some observation error
        Γ = FT(0.003)^2 * I * (maximum(y) - minimum(y))
        noise_dist = MvNormal(zeros(1), Γ)
        apply_noise!(y, noise_dist) = y + rand(noise_dist)[1]
        # broadcast the noise to each element of y
        y_noisy = apply_noise!.(y, Ref(noise_dist))
    else
        y_noisy = deepcopy(y)
    end
    # save y to file
    ustar = y_noisy
    JLD2.save_object(
        joinpath(data_path, "synthetic_ustar_array_noisy.jld2"),
        ustar,
    )
    return y, y_noisy
end

# generate x inputs
profile_file = "synthetic_profile_data.jld2"
save_profiles(FT, data_path = data_path, x_data_file = profile_file)

# read x inputs
x_inputs = load_profiles(joinpath(data_path, profile_file))

# generate synthetic observed y
y, y_noisy = synthetic_observed_y(x_inputs, data_path = data_path)

# save the mean of y_noisy to file
nanmean(x) = mean(filter(!isnan, x))
ustar = y_noisy
JLD2.save_object(
    joinpath(data_path, "synthetic_ustar_array_noisy_mean.jld2"),
    ustar,
)

# save the mean and variance of y_noisy to file
ustar =
    JLD2.load_object(joinpath(data_path, "synthetic_ustar_array_noisy.jld2"))
(; observation, variance) = process_member_data(ustar; output_variance = true)
JLD2.save_object(joinpath(data_path, "obs_mean.jld2"), observation)
JLD2.save_object(joinpath(data_path, "obs_noise_cov.jld2"), variance)

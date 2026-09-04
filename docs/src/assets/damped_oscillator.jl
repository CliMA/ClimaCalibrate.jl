# Model code for the Calibration Tutorial.
#
# The forward model and the model interface live in their own file, which is
# what a worker or a scheduler job loads to run an ensemble member.

import ClimaCalibrate
import JLD2
import TOML

"""
    DampedOscillator(output_dir, ensemble_size, t)

Model interface for a damped harmonic oscillator.

The forward model and the observation map read their configuration from these
fields, so the same object can be used on a worker or serialized into an HPC job
script.
"""
struct DampedOscillator <: ClimaCalibrate.AbstractModelInterface
    """Where the calibration writes its output."""
    output_dir::String

    """Number of ensemble members."""
    ensemble_size::Int

    """Times at which the oscillator's displacement is observed."""
    t::Vector{Float64}
end

"""
    solve_oscillator(damping, frequency, t)

Displacement of a unit-amplitude damped oscillator at times `t`.
"""
solve_oscillator(damping, frequency, t) =
    @. exp(-damping * t) * cos(frequency * t)

"""
    ClimaCalibrate.forward_model(interface::DampedOscillator, iteration, member)

Run the oscillator with this member's parameters and save its displacement.

`forward_model` only receives the iteration and member numbers, so it reads the
parameters EKP drew for this member from the file ClimaCalibrate wrote, and
writes its output under the member's own directory.
"""
function ClimaCalibrate.forward_model(
    interface::DampedOscillator,
    iteration,
    member,
)
    (; output_dir, t) = interface
    member_path =
        ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, member)
    parameters = TOML.parsefile(
        ClimaCalibrate.parameter_path(output_dir, iteration, member),
    )
    damping = parameters["damping"]["value"]
    frequency = parameters["frequency"]["value"]

    displacement = solve_oscillator(damping, frequency, t)
    JLD2.save_object(joinpath(member_path, "displacement.jld2"), displacement)
    return displacement
end

"""
    ClimaCalibrate.observation_map(interface::DampedOscillator, iteration)

Collect every member's displacement into the G ensemble matrix.

Column `m` holds member `m`'s output, in the same order as the observation. A
member whose forward model failed leaves a column of `NaN`s, which is how EKP is
told to ignore it.
"""
function ClimaCalibrate.observation_map(interface::DampedOscillator, iteration)
    (; output_dir, ensemble_size, t) = interface
    G_ensemble = fill(NaN, length(t), ensemble_size)
    for m in 1:ensemble_size
        member_path =
            ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)
        output = joinpath(member_path, "displacement.jld2")
        isfile(output) || continue
        G_ensemble[:, m] .= JLD2.load_object(output)
    end
    return G_ensemble
end

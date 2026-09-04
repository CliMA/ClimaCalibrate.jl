import ClimaCalibrate
import Random

include(
    joinpath(
        pkgdir(ClimaCalibrate),
        "experiments",
        "surface_fluxes_perfect_model",
        "utils.jl",
    ),
)
backend = ClimaCalibrate.backend_type()
@assert backend <: ClimaCalibrate.HPCBackend
directives = Dict{Symbol, Any}(:time => 5, :ntasks => 1, :cpus_per_task => 1)
if backend == ClimaCalibrate.DerechoBackend
    directives[:queue] = "preempt"
    directives[:gpus_per_task] = 1
    directives[:cpus_per_task] = 4
end

climacommon_dict = Dict(
    ClimaCalibrate.DerechoBackend => "climacommon/2026_04_08",
    ClimaCalibrate.ClimaGPUBackend => "climacommon/2026_02_18",
    ClimaCalibrate.CaltechHPCBackend => "climacommon/2025_03_18",
)

cc_module = climacommon_dict[backend]
modules = [cc_module]

if backend == ClimaCalibrate.DerechoBackend
    hpc_config = ClimaCalibrate.PBSConfig(; directives, modules)
else
    hpc_config = ClimaCalibrate.SlurmConfig(; directives, modules)
end

interruption_model_interface, io = mktemp(@__DIR__)

struct CancelModelInterface <: ClimaCalibrate.AbstractModelInterface
    model_interface_filepath::String
end
ClimaCalibrate.forward_model(::CancelModelInterface, i, m) = m == 1 && exit()
ClimaCalibrate.model_interface_filepath(interface::CancelModelInterface) =
    interface.model_interface_filepath
model_interface_str = """
import ClimaCalibrate
struct CancelModelInterface <: ClimaCalibrate.AbstractModelInterface
    model_interface_filepath::String
end
ClimaCalibrate.forward_model(::CancelModelInterface, i, m) =
    m == 1 && exit()
ClimaCalibrate.model_interface_filepath(interface::CancelModelInterface) =
    interface.model_interface_filepath
"""
write(io, model_interface_str)
close(io)

"""
    make_ekp(prior, observation, variance; ekp_kwargs...)

Build the `EnsembleKalmanProcess` this experiment calibrates with.

`Unscented` takes its members from a quadrature stencil around the prior mean,
so it needs no initial ensemble and no `rng`: two runs of the same calibration
give the same answer, which is what lets the backends be compared against each
other.
"""
function make_ekp(prior, observation, variance; ekp_kwargs...)
    return EKP.EnsembleKalmanProcess(
        observation,
        variance,
        EKP.Unscented(prior);
        ekp_kwargs...,
    )
end

eki = make_ekp(prior, observation, variance; verbose = true)

ClimaCalibrate.initialize(eki, prior, output_dir)

backend = backend(hpc_config)
cancel_interface = CancelModelInterface(interruption_model_interface)

# The job script each member runs loads the interface from here
JLD2.save_object(joinpath(output_dir, "interface.jld2"), cancel_interface)
ClimaCalibrate.Calibration.run_iteration(
    backend,
    cancel_interface,
    1,
    ensemble_size,
    output_dir,
)

@testset "Test model checkpoints with interruptions" begin
    for m in 1:ensemble_size
        @test m == 1 ? ClimaCalibrate.model_started(output_dir, 1, m) :
              ClimaCalibrate.model_completed(output_dir, 1, m)
        rm(ClimaCalibrate.checkpoint_path(output_dir, 1, m))
    end
end

# The interrupted iteration above wrote a first iteration of its own, and
# `initialize` keeps the stored one. This calibration is compared against a
# `JuliaBackend` run below, so it starts from a directory of its own
rm(output_dir, recursive = true)

ekp = make_ekp(
    prior,
    observation,
    variance;
    # `Unscented` defaults to a scheduler that stops once the misfit target is
    # reached, which would end the run before `n_iterations`
    scheduler = EKP.DefaultScheduler(),
)
backend = ClimaCalibrate.backend_type()
eki = ClimaCalibrate.Calibration.calibrate(
    backend(hpc_config),
    ekp,
    SurfaceFluxModelInterface(output_dir, ensemble_size),
    n_iterations,
    prior,
    output_dir,
)

@test ClimaCalibrate.last_completed_iteration(output_dir) == n_iterations

@testset "Test model checkpoints for completion" begin
    for m in 1:ensemble_size
        @test ClimaCalibrate.model_completed.(output_dir, 1, m)
    end
end

test_sf_calibration_output(eki, prior, observation, variance)

# Remove previous output - this is not necessary but safe for tests
rm(output_dir, recursive = true)

# Pure Julia calibration, this should run anywhere
ekp = make_ekp(
    prior,
    observation,
    variance;
    # `Unscented` defaults to a scheduler that stops once the misfit target is
    # reached, which would end the run before `n_iterations`
    scheduler = EKP.DefaultScheduler(),
)
julia_eki = ClimaCalibrate.Calibration.calibrate(
    JuliaBackend(),
    ekp,
    SurfaceFluxModelInterface(output_dir, ensemble_size),
    n_iterations,
    prior,
    output_dir,
)
test_sf_calibration_output(julia_eki, prior, observation, variance)

compare_g_ensemble(eki, julia_eki)

theta_star_vec = (; coefficient_a_m_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger"],
    output_dir,
)

g_vs_iter_plot(eki, output_dir)

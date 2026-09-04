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

# Each phase below writes to a directory of its own, created fresh. Reusing one
# directory meant deleting and recreating it under the jobs the previous phase
# still had open: on a cluster filesystem a member then read `interface.jld2` as
# missing and `eki_file.jld2` as truncated, and the `rm` itself hit ENOTEMPTY
# against the `.nfs*` files those jobs left behind. `cleanup = false` keeps the
# directories for the CI artifact upload
mkpath(output_dir)
interrupted_dir =
    mktempdir(output_dir; prefix = "interrupted_", cleanup = false)
hpc_dir = mktempdir(output_dir; prefix = "hpc_", cleanup = false)
julia_dir = mktempdir(output_dir; prefix = "julia_", cleanup = false)

eki = make_ekp(prior, observation, variance; verbose = true)

ClimaCalibrate.initialize(eki, prior, interrupted_dir)

backend = backend(hpc_config)
cancel_interface = CancelModelInterface(interruption_model_interface)

ClimaCalibrate.Calibration.run_iteration(
    backend,
    cancel_interface,
    1,
    ensemble_size,
    interrupted_dir,
)

@testset "Test model checkpoints with interruptions" begin
    for m in 1:ensemble_size
        @test m == 1 ? ClimaCalibrate.model_started(interrupted_dir, 1, m) :
              ClimaCalibrate.model_completed(interrupted_dir, 1, m)
        rm(ClimaCalibrate.checkpoint_path(interrupted_dir, 1, m))
    end
end

ekp = make_ekp(prior, observation, variance)
backend = ClimaCalibrate.backend_type()
eki = ClimaCalibrate.Calibration.calibrate(
    backend(hpc_config),
    ekp,
    SurfaceFluxModelInterface(hpc_dir, ensemble_size),
    n_iterations,
    prior,
    hpc_dir,
)

@test ClimaCalibrate.last_completed_iteration(hpc_dir) == n_iterations

@testset "Test model checkpoints for completion" begin
    for m in 1:ensemble_size
        @test ClimaCalibrate.model_completed.(hpc_dir, 1, m)
    end
end

test_sf_calibration_output(eki, prior, observation, variance)

# Pure Julia calibration, this should run anywhere
ekp = make_ekp(prior, observation, variance)
julia_eki = ClimaCalibrate.Calibration.calibrate(
    JuliaBackend(),
    ekp,
    SurfaceFluxModelInterface(julia_dir, ensemble_size),
    n_iterations,
    prior,
    julia_dir,
)
test_sf_calibration_output(julia_eki, prior, observation, variance)

compare_g_ensemble(eki, julia_eki)

theta_star_vec = (; coefficient_a_m_businger = 4.7)

convergence_plot(
    eki,
    prior,
    theta_star_vec,
    ["coefficient_a_m_businger"],
    hpc_dir,
)

g_vs_iter_plot(eki, hpc_dir)

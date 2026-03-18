"""
    AbstractClimaCommonOption

An abstract type for the different options of `climacommon` to use for the
`ModuleSelector`.
"""
abstract type AbstractClimaCommonOption end

"""
    ClimaCommonVersion

An object for specifying which version of `climacommon` to use.
"""
struct ClimaCommonVersion <: AbstractClimaCommonOption
    climacommon::String
    function ClimaCommonVersion(climacommon_str::String)
        # Validate that climacommon_str is "climacommon" or
        # "climacommon/YYYY_MM_DD"
        if isnothing(
            match(r"^climacommon(?:/\d{4}_\d{2}_\d{2})?$", climacommon_str),
        )
            error(
                "'$climacommon_str' does not match expected format 'climacommon' or 'climacommon/YYYY_MM_DD'",
            )
        end
        check_climacommon_version(climacommon_str)
        new(climacommon_str)
    end
end

"""
    ClimaCommonNotUsed

An object for specifying that `climacommon` should not be loaded as a module.
"""
struct ClimaCommonNotUsed <: AbstractClimaCommonOption end

"""
    check_climacommon_version(climacommon_str)

Check `climacommon_str` against the version of `climacommon` that is loaded by
the user.
"""
function check_climacommon_version(
    climacommon_str::String;
    loaded_modules = get(ENV, "LOADEDMODULES", ""),
)
    # Since it is not possible at this point to determine what version of
    # climacommon will be used, we skip the check
    climacommon_str == "climacommon" && return nothing

    # It can be unsafe to automatically load the version of climacommon that is
    # found from inspecting LOADEDMODULES, so we only read the version of
    # climacommon to check against the climacommon that was requested
    if !isempty(loaded_modules)
        matches =
            collect(eachmatch(r"climacommon/\d{4}_\d{2}_\d{2}", loaded_modules))
        if !isempty(matches)
            loaded_climacommon_str = matches[end].match
            if loaded_climacommon_str != climacommon_str
                @warn "Requested climacommon version ($climacommon_str) does not match the loaded version ($loaded_climacommon_str). This may cause errors during instantiation of the Project.toml"
            end
        end
    end
    return nothing
end

"""
    get_climacommon_str_option(backend::HPCBackend)

Construct a `ClimaCommonVersion` with a default `climacommon` version or
`ClimaCommonNotUsed` depending on the `backend`.
"""
get_climacommon_str_option(backend::HPCBackend) =
    ClimaCommonVersion(default_climacommon_version(backend))

get_climacommon_str_option(::GCPBackend) = ClimaCommonNotUsed()

default_climacommon_version(::CaltechHPCBackend) = "climacommon/2024_10_09"
default_climacommon_version(::ClimaGPUBackend) = "climacommon/2025_05_15"
default_climacommon_version(::DerechoBackend) = "climacommon/2025_02_25"
function default_climacommon_version(b::HPCBackend)
    error(
        "No default climacommon version was defined for $(nameof(typeof(b))). Please implement `default_climacommon_version` for this backend.",
    )
end

get_climacommon_str(opt::ClimaCommonVersion) = opt.climacommon
get_climacommon_str(::ClimaCommonNotUsed) =
    error("climacommon is not used for this backend")

"""
    ModuleSelector

An object to simplify selecting which modules to load when submitting a slurm
or PBS script when using the `HPCBackend`s for calibration.
"""
struct ModuleSelector
    climacommon::AbstractClimaCommonOption
end

# TODO: Not sure about making the backend separate from the modules since I am
# not sure where this will be passed in

"""
    ModuleSelector(
        backend::HPCBackend;
        climacommon::Union{String, AbstractClimaCommonOption}
            = get_climacommon_str_option(backend)
    )

Select the appropriate module to load for each backend.
"""
function ModuleSelector(
    backend::HPCBackend;
    climacommon::Union{String, AbstractClimaCommonOption} = get_climacommon_str_option(
        backend,
    ),
)
    # Check if climacommon is consistent with the chosen backend
    if !use_climacommon(backend)
        climacommon isa ClimaCommonNotUsed || @warn(
            "climacommon is not available for $(nameof(typeof(backend))); ignoring it"
        )
        climacommon = ClimaCommonNotUsed()
    end
    climacommon isa String && (climacommon = ClimaCommonVersion(climacommon))

    return ModuleSelector(climacommon)
end

"""
    use_climacommon(backend::HPCBackend)

Return whether the backend uses `climacommon`.
"""
use_climacommon(::Union{CaltechHPCBackend, ClimaGPUBackend, DerechoBackend}) =
    true
use_climacommon(::HPCBackend) = false

"""
    module_load_string(mod_selector::ModuleSelector, ::HPCBackend)

Return a string that loads the correct modules for a given backend when executed
via bash.
"""
function module_load_string(mod_selector::ModuleSelector, ::CaltechHPCBackend)
    climacommon_str = get_climacommon_str(mod_selector.climacommon)
    return """export MODULEPATH="/resnick/groups/esm/modules:\$MODULEPATH"
    module purge
    module load $climacommon_str"""
end

function module_load_string(mod_selector::ModuleSelector, ::ClimaGPUBackend)
    climacommon_str = get_climacommon_str(mod_selector.climacommon)
    return """module purge
    module load $climacommon_str"""
end

function module_load_string(mod_selector::ModuleSelector, ::DerechoBackend)
    climacommon_str = get_climacommon_str(mod_selector)
    return """export MODULEPATH="/glade/campaign/univ/ucit0011/ClimaModules-Derecho:\$MODULEPATH"
    module purge
    module load $climacommon_str"""
end

function module_load_string(::ModuleSelector, ::GCPBackend)
    return """
    unset CUDA_ROOT
    unset NVHPC_CUDA_HOME
    unset CUDA_INC_DIR
    unset CPATH
    unset NVHPC_ROOT

    # NVHPC and HPC-X paths
    export NVHPC="/sw/nvhpc/Linux_x86_64/24.5"
    export HPCX_PATH="\${NVHPC}/comm_libs/12.4/hpcx/hpcx-2.19"

    # CUDA environment
    export CUDA_HOME="\${NVHPC}/cuda/12.4"
    export CUDA_PATH="\${CUDA_HOME}"
    export CUDA_ROOT="\${CUDA_HOME}"

    # MPI via MPIwrapper
    export MPITRAMPOLINE_LIB="/sw/mpiwrapper/lib/libmpiwrapper.so"
    export OPAL_PREFIX="\${HPCX_PATH}/ompi"

    # Library paths - CUDA first, then HPC-X
    export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${HPCX_PATH}/ompi/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

    # Executable paths
    export PATH="/sw/mpiwrapper/bin:\${CUDA_HOME}/bin:\${PATH}"
    export PATH="\${NVHPC}/profilers/Nsight_Systems/target-linux-x64:\${PATH}"

    # Julia
    export PATH="/sw/julia/julia-1.11.5/bin:\${PATH}"
    export JULIA_MPI_HAS_CUDA="true"
    """
end

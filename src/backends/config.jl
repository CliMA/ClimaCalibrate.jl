abstract type AbstractHPCConfig end

struct SlurmConfig <: AbstractHPCConfig
    directives::Vector{Pair{Symbol, Any}}
    modules::Vector{String}
    env_vars::Vector{Pair{String, Any}}
end

struct PBSConfig <: AbstractHPCConfig
    directives::Vector{Pair{Symbol, Any}}
    modules::Vector{String}
    env_vars::Vector{Pair{String, Any}}
end

"""
    SlurmConfig(;
        directives = Pair{Symbol, Any}[],
        modules = String[],
        env_vars = Pair{String, Any}[],
    )

Create a `SlurmConfig` specifying the `directives`, `modules`, and `env_vars`
for `SlurmBackend`s.
"""
function SlurmConfig(;
    directives = Pair{Symbol, Any}[],
    modules = String[],
    env_vars = Pair{String, Any}[],
)
    directives, modules, env_vars =
        parse_config_args(directives, modules, env_vars)

    # Format time
    @assert any(p -> first(p) == :time, directives) "Slurm directives must include key :time"
    time_idx = findfirst(p -> first(p) == :time, directives)
    directives[time_idx] =
        :time => format_slurm_time(last(directives[time_idx]))

    # Add default directives
    _add_unique_entry!(directives, :ntasks, 1)
    _add_unique_entry!(directives, :gpus_per_task, 0)

    _check_climacommon(modules)

    # Add environment variables
    _add_climacomms_env_vars!(env_vars, directives)

    return SlurmConfig(directives, modules, env_vars)
end

"""
    PBSConfig(;
        directives = Pair{Symbol, Any}[],
        modules = String[],
        env_vars = Pair{String, Any}[],
    )

Create a `PBSConfig` specifying the `directives`, `modules`, and `env_vars`
for the `DerechoBackend`.
"""
function PBSConfig(;
    directives = Pair{Symbol, Any}[],
    modules = String[],
    env_vars = Pair{String, Any}[],
)
    directives, modules, env_vars =
        parse_config_args(directives, modules, env_vars)

    # Format time
    @assert any(p -> first(p) == :time, directives) "PBS directives must include key :time"
    time_idx = findfirst(p -> first(p) == :time, directives)
    directives[time_idx] = :time => format_pbs_time(last(directives[time_idx]))

    # Add default directives
    # queue and job_priority are specific to Derecho, but we only support
    # Derecho right now
    _add_unique_entry!(directives, :queue, "main")
    _add_unique_entry!(directives, :ntasks, 1)
    _add_unique_entry!(directives, :cpus_per_task, 1)
    _add_unique_entry!(directives, :gpus_per_task, 0)
    _add_unique_entry!(directives, :job_priority, "regular")

    _check_climacommon(modules)

    # Set environmental variables for ClimaComms
    _add_unique_entry!(env_vars, "JULIA_MPI_HAS_CUDA", true)
    _add_climacomms_env_vars!(env_vars, directives)

    return PBSConfig(directives, modules, env_vars)
end

"""
    parse_config_args(directives, modules, env_vars)

Parse `directives`, `modules`, and `env_vars` by taking ownership of them and
enforcing uniqueness.
"""
function parse_config_args(directives, modules, env_vars)
    owned_directives = Pair{Symbol, Any}[]
    owned_modules = String[]
    owned_env_vars = Pair{String, Any}[]

    for (key, val) in directives
        push!(owned_directives, Symbol(key) => val)
    end

    for m in modules
        push!(owned_modules, m)
    end

    for (key, val) in env_vars
        push!(owned_env_vars, Symbol(key) => val)
    end

    allunique(keys(owned_directives)) ||
        error("Not all the directives are unique")
    allunique(owned_modules) || error("Not all the modules are unique")
    allunique(keys(owned_env_vars)) ||
        error("Not all the environment variables are unique")

    return owned_directives, owned_modules, owned_env_vars
end

"""
    _check_climacommon(modules)

Check if `climacommon` module is in `modules` and if it is, compare it against
the loaded `climacommon` module if it exists.

Note that we do not automatically load `climacommon` because the user might
intentionally not want to load it and it could be unsafe to read from
environment variables to choose which modules to automatically load.
"""
function _check_climacommon(modules)
    # Get climacommon from modules and from LOADEDMODULES which is an
    # environment variable that is modified by Environment Modules

    # Get climacommon from modules
    climacommon_regex = r"^climacommon(?:/\d{4}_\d{2}_\d{2})?$"
    job_modules_idx = findlast(m -> occursin(climacommon_regex, m), modules)
    job_module_cc =
        isnothing(job_modules_idx) ? nothing : modules[job_modules_idx]

    # Get climacommon from LOADEDMODULES
    loaded_modules = split(get(ENV, "LOADEDMODULES", ""), ":")
    loaded_modules_idx =
        findlast(m -> occursin(climacommon_regex, m), loaded_modules)
    loaded_module_cc =
        isnothing(loaded_modules_idx) ? nothing :
        loaded_modules[loaded_modules_idx]

    # Case when climacommon is not loaded currently and is not in the job script
    isnothing(job_module_cc) && isnothing(loaded_module_cc) && return nothing

    # Case when there is no climacommon in the job script
    if isnothing(job_module_cc)
        warn_message = "You most likely want to load climacommon in the config by passing \"climacommon\" as an element to the modules keyword argument"
        isnothing(loaded_modules_idx) || (
            warn_message *= " The currently loaded climacommon version is $job_module_cc"
        )
        # Do not throw an error here because there are fringe cases where you
        # might not want to load climacommon
        @warn warn_message
    end

    # Case when there is a mismatch between climacommon versions
    if !isnothing(job_module_cc) &&
       !isnothing(loaded_module_cc) &&
       job_module_cc != loaded_module_cc
        # Like the case above, do not throw an error because the user might want
        # to load different versions of climacommon
        @warn "The climacommon module ($job_module_cc) specfied in the job script and the climacommon module ($loaded_module_cc) currently loaded are not the same. You may get issues with precompilation and instantiation of the Julia environment when the job starts"
    end

    return nothing
end

"""
    _add_climacomms_env_vars!(env_vars, directives)

Add the environment variables `CLIMACOMMS_DEVICE` and `CLIMACOMMS_CONTEXT` for
ClimaComms.
"""
function _add_climacomms_env_vars!(env_vars, directives)
    gpu_idx = findfirst(p -> first(p) == :gpus_per_task, directives)
    gpus_per_task = last(directives[gpu_idx])
    climacomms_device = gpus_per_task > 0 ? "CUDA" : "CPU"
    _add_unique_entry!(env_vars, "CLIMACOMMS_DEVICE", climacomms_device)
    _add_unique_entry!(env_vars, "CLIMACOMMS_CONTEXT", "MPI")
    return nothing
end

"""
    _add_unique_entry!(directives::Vector::Pair{K, V}, k, v) where {K, V}

Add the pair `k => v` to `vec` if `vec` does not contain a pair whose first
value is `k`.
"""
function _add_unique_entry!(vec::Vector{Pair{K, V}}, k, v) where {K, V}
    idx = findfirst(p -> first(p) == k, vec)
    isnothing(idx) && push!(vec, k => v)
    return nothing
end

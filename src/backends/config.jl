export SlurmConfig, PBSConfig, AbstractHPCConfig

import OrderedCollections: OrderedDict

"""
    abstract type AbstractHPCConfig end

An abstract type for high-performance computing job configuration objects used
by [`HPCBackend`](@ref)s when creating job scripts.

## Interface

All subtypes of `AbstractHPCConfig` must have the following fields:
- `directives::OrderedDict{Symbol, Any}`: Scheduler directives (e.g., resource
  requests, time limits, etc.).
- `modules::Vector{String}`: List of modules to load in the job environment.
- `env_vars::OrderedDict{String, Any}`: Environment variables to set for the
  job environment.

Subtypes must also provide the methods:
- `generate_directives(config)`: Return a string of scheduler directives for
  the job script.
- `generate_modules(config)`: Return a string of module load commands for the
  job script.
- `generate_env_vars(config)`: Return a string of environment variable export
  commands for the job script.
"""
abstract type AbstractHPCConfig end

"""
    SlurmConfig <: AbstractHPCConfig

A configuration holding Slurm directives, modules, and environment variables
that will be used when creating a job scripts by the `SlurmBackend`s.
"""
struct SlurmConfig <: AbstractHPCConfig
    directives::OrderedDict{Symbol, Any}
    modules::Vector{String}
    env_vars::OrderedDict{String, Any}
end

"""
    PBSConfig <: AbstractHPCConfig

A configuration holding PBS directives, modules, and environment variables
that will be used when creating a job scripts by the [`DerechoBackend`](@ref).
"""
struct PBSConfig <: AbstractHPCConfig
    directives::OrderedDict{Symbol, Any}
    modules::Vector{String}
    env_vars::OrderedDict{String, Any}
end

"""
    SlurmConfig(;
        directives = Pair{Symbol, Any}[],
        modules = String[],
        env_vars = Pair{String, Any}[],
    )

Create a `SlurmConfig` specifying the `directives`, `modules`, and `env_vars`
for `SlurmBackend`s.

## Defaults

The default directive is
- `:gpus_per_task`: 0.

The default environment variables are
- `CLIMACOMMS_DEVICE`: "CPU" or "GPU" depending on the job directives,
- `CLIMACOMMS_CONTEXT`: "MPI".

## Examples

This example creates a Slurm configuration for a job with a single task, using
12 CPUs and 1 GPU, and a runtime of 720 minutes. It loads the latest version of
climacommon and explicitly sets environment variables for ClimaComms.

```julia
ClimaCalibrate.SlurmConfig(;
    directives = [
        :ntasks => 1,
        :gpus_per_task => 1,
        :cpus_per_task => 12,
        :time => 720,
    ],
    modules = ["climacommon"],
    env_vars = [
        "CLIMACOMMS_CONTEXT" => "SINGLETON",
        "CLIMACOMMS_DEVICE" => "CUDA",
    ],
)
```
"""
function SlurmConfig(;
    directives = Pair{Symbol, Any}[],
    modules = String[],
    env_vars = Pair{String, Any}[],
)
    directives, modules, env_vars =
        _parse_config_args(directives, modules, env_vars)

    # Format time
    @assert haskey(directives, :time) "Slurm directives must include key :time"
    directives[:time] = format_slurm_time(directives[:time])

    # Add default directives
    get!(directives, :gpus_per_task, 0)

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

The supported directives are: `time`, `queue`, `ntasks`, `cpus_per_task`,
`gpus_per_task`, and `job_priority`. These directive names follow the Slurm
naming convention (e.g., `time` instead of `walltime`). Any other directives
provided will be ignored.

## Defaults

The default directives are
- `queue`: "main",
- `ntasks`: 1,
- `cpus_per_task`: 1,
- `gpus_per_task`: 0,
- `job_priority`: "regular".

The default environment variables are
- `CLIMACOMMS_DEVICE`: "CPU" or "GPU" depending on the job directives,
- `CLIMACOMMS_CONTEXT`: "MPI".

## Examples

This example creates a PBS configuration for a job with a single task, using
12 CPUs and 1 GPU, and a runtime of 720 minutes. It loads the latest version of
climacommon and explicitly sets environment variables for ClimaComms.

```julia
ClimaCalibrate.PBSConfig(;
    directives = [
        :ntasks => 1,
        :gpus_per_task => 1,
        :cpus_per_task => 12,
        :time => 720,
    ],
    modules = ["climacommon"],
    env_vars = [
        "CLIMACOMMS_CONTEXT" => "SINGLETON",
        "CLIMACOMMS_DEVICE" => "CUDA",
    ],
)
```
"""
function PBSConfig(;
    directives = Pair{Symbol, Any}[],
    modules = String[],
    env_vars = Pair{String, Any}[],
)
    directives, modules, env_vars =
        _parse_config_args(directives, modules, env_vars)

    # Format time
    @assert haskey(directives, :time) "PBS directives must include key :time"
    directives[:time] = format_pbs_time(directives[:time])

    # Add default directives
    # queue and job_priority are specific to Derecho, but we only support
    # Derecho right now
    get!(directives, :queue, "main")
    get!(directives, :ntasks, 1)
    get!(directives, :cpus_per_task, 1)
    get!(directives, :gpus_per_task, 0)
    get!(directives, :job_priority, "regular")

    # Check for any directives that are not supported
    supported_directives =
        [:queue, :time, :ntasks, :cpus_per_task, :gpus_per_task, :job_priority]
    not_supported_directives =
        setdiff(keys(directives), Set(supported_directives))
    isempty(not_supported_directives) ||
        @warn "The following directives are not supported and will be ignored: $(join(not_supported_directives, ", "))"

    _check_climacommon(modules)

    # Set environmental variables for ClimaComms
    get!(env_vars, "JULIA_MPI_HAS_CUDA", true)
    _add_climacomms_env_vars!(env_vars, directives)

    return PBSConfig(directives, modules, env_vars)
end

"""
    _parse_config_args(directives, modules, env_vars)

Parse `directives`, `modules`, and `env_vars` by taking ownership of them and
enforcing uniqueness.
"""
function _parse_config_args(directives, modules, env_vars)
    owned_directives = OrderedDict{Symbol, Any}()
    owned_modules = String[]
    owned_env_vars = OrderedDict{String, Any}()

    for (key, val) in directives
        key = Symbol(key)
        haskey(owned_directives, key) &&
            error("Not all directives ($key) are unique")
        owned_directives[key] = val
    end

    for m in modules
        push!(owned_modules, String(m))
    end

    for (key, val) in env_vars
        key = String(key)
        haskey(owned_env_vars, key) &&
            error("Not all environment variables ($key) are unique")
        owned_env_vars[key] = val
    end

    allunique(owned_modules) || error("Not all modules are unique")

    return owned_directives, owned_modules, owned_env_vars
end

"""
    _check_climacommon(modules; env = ENV)

Check if `climacommon` module is in `modules` and if it is, compare it against
the loaded `climacommon` module if it exists.

Note that we do not automatically load `climacommon` because the user might
intentionally not want to load it and it could be unsafe to read from the
`LOADEDMODULES` environment variable to choose which modules to automatically
load.
"""
function _check_climacommon(modules; env = ENV)
    # Get climacommon from modules and LOADEDMODULES
    # LOADEDMODULES is an environment variable that is modified by Environment
    # Modules
    # Note that we do not modify the modules list to avoid arbitrarily loading
    # any modules

    # Get climacommon from modules
    climacommon_regex = r"^climacommon(?:/\d{4}_\d{2}_\d{2})?$"
    job_modules_idx = findlast(m -> occursin(climacommon_regex, m), modules)
    job_module_cc =
        isnothing(job_modules_idx) ? nothing : modules[job_modules_idx]

    # Get climacommon from LOADEDMODULES
    loaded_modules = split(get(env, "LOADEDMODULES", ""), ":")
    loaded_modules_idx =
        findlast(m -> occursin(climacommon_regex, m), loaded_modules)
    loaded_module_cc =
        isnothing(loaded_modules_idx) ? nothing :
        loaded_modules[loaded_modules_idx]

    # Case when climacommon is not loaded currently and is not in the job script
    isnothing(job_module_cc) && isnothing(loaded_module_cc) && return nothing

    # Case when there is no climacommon in the job script
    if isnothing(job_module_cc)
        warn_message = "You most likely want to load climacommon in the config by passing \"climacommon\" to the modules keyword argument"
        isnothing(loaded_modules_idx) || (
            warn_message *= ". The currently loaded climacommon version is $loaded_module_cc "
        )
        # Do not throw an error here because there are fringe cases where you
        # might not want to load climacommon
        @warn warn_message
    end

    # Case when there is a mismatch between climacommon versions
    # If climacommon is specified with no specific version, then we can't check
    # without loading the climacommon module
    if !isnothing(job_module_cc) &&
       !isnothing(loaded_module_cc) &&
       job_module_cc != "climacommon" &&
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
    gpus_per_task = directives[:gpus_per_task]
    climacomms_device = gpus_per_task > 0 ? "CUDA" : "CPU"
    get!(env_vars, "CLIMACOMMS_DEVICE", climacomms_device)
    get!(env_vars, "CLIMACOMMS_CONTEXT", "MPI")
    return nothing
end

"""
    generate_directives(config::SlurmConfig)

Generate Slurm directives from `config` that can be used when generating job
scripts.
"""
function generate_directives(config::SlurmConfig)
    (; directives) = config
    slurm_directives = map(collect(directives)) do (k, v)
        k = string(k)
        flag = length(k) == 1 ? "-" : "--"
        "#SBATCH $flag$(replace(k, "_" => "-"))=$(replace(string(v), "_" => "-"))"
    end
    return join(slurm_directives, "\n")
end

"""
    generate_directives(config::PBSConfig)

Generate PBS directives from the directives specified in `config` for inclusion
in job scripts.

The supported directives are: `queue`, `time`, `ntasks`, `cpus_per_task`,
`gpus_per_task`, and `job_priority`. These directive names follow the Slurm
naming convention (e.g., `time` instead of `walltime`). Any other directives
provided will be ignored.
"""
function generate_directives(config::PBSConfig)
    (; directives) = config
    queue = directives[:queue]
    walltime = directives[:time]
    num_nodes = directives[:ntasks]
    cpus_per_node = directives[:cpus_per_task]
    gpus_per_node = directives[:gpus_per_task]
    job_priority = directives[:job_priority]
    ranks_per_node = gpus_per_node > 0 ? gpus_per_node : cpus_per_node
    return """
    #PBS -j oe
    #PBS -A UCIT0011
    #PBS -q $queue
    #PBS -l job_priority=$job_priority
    #PBS -l walltime=$walltime
    #PBS -l select=$num_nodes:ncpus=$cpus_per_node:ngpus=$gpus_per_node:mpiprocs=$ranks_per_node"""
end

"""
    generate_modules(config::AbstractHPCConfig)

Generate a string of module load commands from the modules specified in
`config` for inclusion in job scripts.
"""
function generate_modules(config::AbstractHPCConfig)
    return join(("module load $m" for m in config.modules), "\n")
end

"""
    generate_env_vars(config::AbstractHPCConfig)

Generate a string of environmental variables from the modules specified in
`config` for inclusion in job scripts.
"""
function generate_env_vars(config::AbstractHPCConfig)
    return join(("export $k=\"$v\"" for (k, v) in config.env_vars), "\n")
end

"""
    format_slurm_time(minutes::Int)

Format `minutes` into a string accepted by slurm.
"""
function format_slurm_time(minutes::Int)
    days, remaining_minutes = divrem(minutes, (60 * 24))
    hours, remaining_minutes = divrem(remaining_minutes, 60)
    if days > 0
        return string(
            days,
            "-",
            lpad(hours, 2, '0'),
            ":",
            lpad(remaining_minutes, 2, '0'),
            ":00",
        )
    else
        return string(
            lpad(hours, 2, '0'),
            ":",
            lpad(remaining_minutes, 2, '0'),
            ":00",
        )
    end
end

"""
    format_slurm_time(str::AbstractString)

Return `str`.

This function does not validate whether `str` is correct or not.
"""
format_slurm_time(str::AbstractString) = str


"""
    format_pbs_time(minutes::Int)

Format `minutes` into a string (HH:MM:SS) accepted by PBS.
"""
function format_pbs_time(minutes::Int)
    hours, remaining_minutes = divrem(minutes, 60)
    return string(
        lpad(hours, 2, '0'),
        ":",
        lpad(remaining_minutes, 2, '0'),
        ":00",
    )
end

"""
    format_pbs_time(str::AbstractString)

Return `str`.

This function does not validate whether `str` is correct or not.
"""
format_pbs_time(str::AbstractString) = str

"""
    ClimaCalibrate.Backend.Vast

Rent Vast.ai instances as the workers of a
[`SharedFilesBackend`](@ref ClimaCalibrate.SharedFilesBackend), through the
`vastai` command line tool. It must be installed, with an API key holding the
`instance_read`, `instance_write`, and `misc` permissions.
"""
module Vast

import ..launch
import ...SharedFiles: BucketFiles, url

"""
    Workers(n; gpu = "RTX_3090", disk = 80,
            image = "nefrathenrici/clima-base:latest",
            julia_version = string(VERSION), env = Dict())

`n` instances with one `gpu` each, cheapest first, for the `workers` of a
`SharedFilesBackend`. Each runs `image`, whose entrypoint clones the
repository holding the model interface file, at the branch checked out
here, instantiates the active project with Julia `julia_version`, one the
image holds (1.10.10, 1.12.6, 1.13.3), and runs
`julia --threads=2 -e 'using AWSS3, ClimaCalibrate; ClimaCalibrate.run_worker()'`.
The branch must be pushed, and the project, the interface file, and the
calibration's `output_dir` must be inside the repository. The instance
environment holds `env`, the `CLIMA_*` variables the entrypoint and
`run_worker()` read, and `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
from this environment. The instances are destroyed when this process exits.
"""
Base.@kwdef struct Workers
    n::Int
    gpu::String = "RTX_3090"
    disk::Int = 80
    image::String = "nefrathenrici/clima-base:latest"
    julia_version::String = string(VERSION)
    env::Dict{String, String} = Dict{String, String}()
end

Workers(n; kwargs...) = Workers(; n, kwargs...)

function launch(w::Workers; files, model_interface, output_dir)
    files isa BucketFiles ||
        error("Vast.ai workers need `files` to be a BucketFiles, got $files")
    isnothing(Sys.which("vastai")) && error("vastai is not installed")
    (; root, remote, branch) = checkout(dirname(model_interface))
    clone = "/clima/" * replace(basename(remote), r"\.git$" => "")
    inside(path) = begin
        rel = relpath(abspath(path), root)
        startswith(rel, "..") && error(
            "$path is outside $root, which is what the workers clone",
        )
        joinpath(clone, rel)
    end
    vars = merge(
        Dict(
            "CLIMA_REPOS" => "$remote@$branch",
            "CLIMA_PROJECT" =>
                relpath(inside(dirname(Base.active_project())), "/clima"),
            "CLIMA_MODEL_INTERFACE" => inside(model_interface),
            "CLIMA_OUTPUT_DIR" => inside(output_dir),
            "CLIMA_FILES" => url(files),
            "JULIA_VERSION" => w.julia_version,
            "AWS_ACCESS_KEY_ID" => ENV["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY" => ENV["AWS_SECRET_ACCESS_KEY"],
        ),
        w.env,
    )
    flags = join(("-e $k=$v" for (k, v) in vars), " ")
    code = "using AWSS3,ClimaCalibrate;ClimaCalibrate.run_worker()"
    cmd = `vastai launch instance --raw -g $(w.gpu) -n 1 -i $(w.image)
           -d $(w.disk) --env $flags --args julia --threads=2 -e $code`
    ids = Int[]
    atexit(() -> destroy_workers(ids))
    for _ in 1:(w.n)
        # The command holds the credentials, so a failure must not print it
        process = open(cmd)
        out = read(process, String)
        success(process) ||
            error("vastai launch instance exited with $(process.exitcode)")
        found = match(r"new_contract[\"']?:\s*(\d+)", out)
        isnothing(found) && error("vastai returned no instance id: $out")
        id = parse(Int, found[1])
        @info "Launched Vast.ai instance $id with one $(w.gpu)"
        push!(ids, id)
    end
    return ids
end

# The checkout holding `path`: its root, the https URL of `origin`, and the
# branch, which must be pushed
function checkout(path)
    git(args...) = readchomp(`git -C $path $(collect(args))`)
    root = git("rev-parse", "--show-toplevel")
    branch = git("branch", "--show-current")
    isempty(branch) && error("$root is not on a branch")
    ahead = git("rev-list", "--count", "@{upstream}..HEAD")
    ahead == "0" || error(
        "$branch is $ahead commits ahead of its upstream, push it first",
    )
    isempty(git("status", "--porcelain", "--untracked-files=no")) ||
        @warn "$root has uncommitted changes the workers will not see"
    remote = replace(
        git("remote", "get-url", "origin"),
        r"^git@([^:]+):" => s"https://\1/",
    )
    return (; root, remote, branch)
end

"""
    destroy_workers(ids)

End the instances. `launch` registers this for exit, so it is only needed
by hand, for instances that outlived their driver.
"""
function destroy_workers(ids)
    for id in ids
        run(`vastai destroy instance -y $id`)
    end
    return nothing
end

"The instance's console output so far."
logs(id) = read(`vastai logs $id`, String)

end # module Vast

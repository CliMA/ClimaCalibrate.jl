"""
    ClimaCalibrate.Backend.Vast

Rent Vast.ai instances that run [`run_worker`](@ref ClimaCalibrate.run_worker),
through the `vastai` command line tool. It must be installed, with an API key
holding the `instance_read`, `instance_write`, and `misc` permissions.
"""
module Vast

"""
    launch_workers(n; worker, repos, project, gpu = "RTX_3090", disk = 80,
                   image = "nefrathenrici/clima-base:latest",
                   julia_version = "1.10.10", env = Dict())

Rent `n` instances with one `gpu` each, cheapest first. Each runs `image`,
whose entrypoint clones `repos` into `/clima` and instantiates `project`,
then runs `julia --threads=2 worker`, a path inside the clone. The instance
environment holds `env`, the `CLIMA_*` variables the entrypoint reads, and
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from this environment.
Returns the instance ids.
"""
function launch_workers(
    n;
    worker,
    repos,
    project,
    gpu = "RTX_3090",
    disk = 80,
    image = "nefrathenrici/clima-base:latest",
    julia_version = "1.10.10",
    env = Dict{String, String}(),
)
    vars = merge(
        Dict(
            "CLIMA_REPOS" => repos,
            "CLIMA_PROJECT" => project,
            "JULIA_VERSION" => julia_version,
            "AWS_ACCESS_KEY_ID" => ENV["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY" => ENV["AWS_SECRET_ACCESS_KEY"],
        ),
        env,
    )
    flags = join(("-e $k=$v" for (k, v) in vars), " ")
    isnothing(Sys.which("vastai")) && error("vastai is not installed")
    cmd = `vastai launch instance --raw -g $gpu -n 1 -i $image -d $disk
           --env $flags --args julia --threads=2 $worker`
    return map(1:n) do _
        # The command holds the credentials, so a failure must not print it
        process = open(cmd)
        out = read(process, String)
        success(process) ||
            error("vastai launch instance exited with $(process.exitcode)")
        found = match(r"new_contract[\"']?:\s*(\d+)", out)
        isnothing(found) && error("vastai returned no instance id: $out")
        id = parse(Int, found[1])
        @info "Launched Vast.ai instance $id with one $gpu"
        return id
    end
end

"""
    destroy_workers(ids)

End the instances. A worker exits on its own after its idle timeout, but an
instance bills until destroyed.
"""
function destroy_workers(ids)
    for id in ids
        run(`vastai destroy instance $id`)
    end
    return nothing
end

"The instance's console output so far."
logs(id) = read(`vastai logs $id`, String)

end # module Vast

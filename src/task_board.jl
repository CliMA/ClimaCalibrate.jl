"""
    ClimaCalibrate.TaskBoard

How a driver hands ensemble members to workers through shared files, and how
each side reports back.

Each member's directory, the one `path_to_ensemble_member` names, holds small
TOML files:

    task.toml     posted by the driver: attempt
    claim.toml    made by the worker that won the member: worker_id. Its
                  modification time is the lease, renewed by the owner and
                  read by the driver
    done.toml     written by the worker once the forward model finished,
                  listing the outputs it uploaded
    failed.toml   written by the worker when the forward model raised, or by
                  the driver after [`MAX_ATTEMPTS`](@ref) claims lapsed

`state/current.toml` names the iteration workers should look at, its ensemble
size, the lease length, and the outputs a worker uploads when it does not run
in place.

Claiming relies on `create`: of any number of workers trying to create
`claim.toml` for one member, exactly one succeeds.
"""
module TaskBoard

import TOML
import ..ClimaCalibrate
import ..SharedFiles: AbstractSharedFiles, create

"""
    MAX_ATTEMPTS

How many times a member is handed out before the driver marks it failed. A
claim that lapses, because its worker died or hung, costs one attempt.
"""
const MAX_ATTEMPTS = 3

member_key(iteration, member, name) = joinpath(
    ClimaCalibrate.path_to_ensemble_member("", iteration, member),
    name,
)
const CURRENT_KEY = "state/current.toml"

read_toml(files, key) = TOML.parse(read(files, key))
write_toml(files, key, dict) = write(files, key, sprint(TOML.print, dict))

"One of `:done`, `:failed`, `:claimed`, `:open`, or `:unpublished`."
function status(files::AbstractSharedFiles, iteration, member)
    ispath(files, member_key(iteration, member, "done.toml")) && return :done
    ispath(files, member_key(iteration, member, "failed.toml")) &&
        return :failed
    ispath(files, member_key(iteration, member, "claim.toml")) &&
        return :claimed
    ispath(files, member_key(iteration, member, "task.toml")) && return :open
    return :unpublished
end

"""
Members grouped by [`status`](@ref):
`(; open, claimed, done, failed, unpublished)`.
"""
function summary(files::AbstractSharedFiles, iteration, ensemble_size)
    groups = (;
        open = Int[],
        claimed = Int[],
        done = Int[],
        failed = Int[],
        unpublished = Int[],
    )
    for member in 1:ensemble_size
        push!(getproperty(groups, status(files, iteration, member)), member)
    end
    return groups
end

"Post a member for workers to claim. Returns `false` if it is already posted."
function publish!(files::AbstractSharedFiles, iteration, member)
    status(files, iteration, member) == :unpublished || return false
    write_toml(
        files,
        member_key(iteration, member, "task.toml"),
        Dict("attempt" => 1),
    )
    return true
end

"""
Point workers at `iteration`. `lease` is the seconds a claim stays valid
without renewal, `outputs` the files or directories, relative to the member
directory, a worker uploads once its forward model has run.
"""
function set_current!(
    files::AbstractSharedFiles,
    iteration,
    ensemble_size;
    lease,
    outputs = String[],
)
    write_toml(
        files,
        CURRENT_KEY,
        Dict(
            "iteration" => iteration,
            "ensemble_size" => ensemble_size,
            "lease" => Float64(lease),
            "outputs" => outputs,
        ),
    )
    return nothing
end

"""
The iteration workers should look at, as `(; iteration, ensemble_size,
lease, outputs)`, or `nothing` before a driver has posted one.
"""
function current(files::AbstractSharedFiles)
    ispath(files, CURRENT_KEY) || return nothing
    c = read_toml(files, CURRENT_KEY)
    return (;
        iteration = c["iteration"],
        ensemble_size = c["ensemble_size"],
        lease = c["lease"],
        outputs = c["outputs"],
    )
end

"""
Try to take an open member. Exactly one of any number of concurrent callers
gets `true`.
"""
function claim!(files::AbstractSharedFiles, iteration, member, worker_id)
    status(files, iteration, member) == :open || return false
    return create(
        files,
        member_key(iteration, member, "claim.toml"),
        sprint(TOML.print, Dict("worker_id" => worker_id)),
    )
end

"""
Extend a claim held by `worker_id`. Returns `false` if the claim is gone or
belongs to another worker.
"""
function renew!(files::AbstractSharedFiles, iteration, member, worker_id)
    key = member_key(iteration, member, "claim.toml")
    ispath(files, key) || return false
    read_toml(files, key)["worker_id"] == worker_id || return false
    touch(files, key)
    return true
end

"""
Whether the claim has gone `lease` seconds without renewal, by this host's
clock.
"""
lease_expired(files::AbstractSharedFiles, iteration, member, lease) =
    time() - mtime(files, member_key(iteration, member, "claim.toml")) > lease

"""
Take a lapsed claim back: the member reopens with its attempt raised, or
fails once [`MAX_ATTEMPTS`](@ref) is used up. Returns the new status.
"""
function requeue!(files::AbstractSharedFiles, iteration, member)
    task = read_toml(files, member_key(iteration, member, "task.toml"))
    rm(files, member_key(iteration, member, "claim.toml"))
    attempt = task["attempt"] + 1
    if attempt > MAX_ATTEMPTS
        write_toml(
            files,
            member_key(iteration, member, "failed.toml"),
            Dict(
                "attempt" => task["attempt"],
                "error" => "no completion after $(task["attempt"]) attempts",
            ),
        )
        return :failed
    end
    task["attempt"] = attempt
    write_toml(files, member_key(iteration, member, "task.toml"), task)
    return :open
end

"Record that the member's forward model finished and which files it uploaded."
function complete!(
    files::AbstractSharedFiles,
    iteration,
    member;
    worker_id,
    outputs = String[],
)
    write_toml(
        files,
        member_key(iteration, member, "done.toml"),
        Dict(
            "attempt" => attempt(files, iteration, member),
            "worker_id" => worker_id,
            "outputs" => outputs,
        ),
    )
    return nothing
end

"Record that the member's forward model raised, with the `error` text."
function fail!(files::AbstractSharedFiles, iteration, member; worker_id, error)
    write_toml(
        files,
        member_key(iteration, member, "failed.toml"),
        Dict(
            "attempt" => attempt(files, iteration, member),
            "worker_id" => worker_id,
            "error" => string(error),
        ),
    )
    return nothing
end

"Withdraw a failed member so that [`publish!`](@ref) posts it afresh."
function reset!(files::AbstractSharedFiles, iteration, member)
    for name in ("failed.toml", "claim.toml", "task.toml")
        rm(files, member_key(iteration, member, name))
    end
    return nothing
end

"The member's current attempt number."
attempt(files::AbstractSharedFiles, iteration, member) =
    read_toml(files, member_key(iteration, member, "task.toml"))["attempt"]

"The files a done member uploaded, relative to its directory."
outputs(files::AbstractSharedFiles, iteration, member) =
    read_toml(files, member_key(iteration, member, "done.toml"))["outputs"]

"The worker holding the member's claim."
owner(files::AbstractSharedFiles, iteration, member) =
    read_toml(files, member_key(iteration, member, "claim.toml"))["worker_id"]

"Why the member failed."
failure(files::AbstractSharedFiles, iteration, member) =
    read_toml(files, member_key(iteration, member, "failed.toml"))["error"]

end # module TaskBoard

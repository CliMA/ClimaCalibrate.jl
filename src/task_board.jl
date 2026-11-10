"""
    ClimaCalibrate.TaskBoard

How a driver hands ensemble members to workers through shared files, and how
each side reports back.

Each member of an iteration has a directory, the one `path_to_ensemble_member`
names, holding small TOML files:

    task.toml         posted by the driver: iteration, member, attempt
    claim/owner.toml  made by the worker that won the member. Its modification
                      time is the lease, renewed by the owner and read by the
                      driver
    done.toml         written by the worker once the forward model finished
    failed.toml       written by the worker when the forward model raised, or
                      by the driver after [`MAX_ATTEMPTS`](@ref) claims lapsed

`state/current.toml` names the iteration workers should look at, its ensemble
size, and the lease length.

Claiming relies on [`create`](@ref): of any number of workers
trying to create `claim/` for one member, exactly one succeeds.
"""
module TaskBoard

import TOML
import ..SharedFiles: AbstractSharedFiles, create

"""
    MAX_ATTEMPTS

How many times a member is handed out before the driver marks it failed. A
claim that lapses, because its worker died or hung, costs one attempt.
"""
const MAX_ATTEMPTS = 3

"""
    member_key(iteration, member[, name])

Key of a member's directory, `iteration_XXX/member_XXX`, matching
`path_to_ensemble_member`, or of the file `name` inside it.
"""
member_key(
    iteration,
    member,
) = "iteration_$(lpad(iteration, 3, '0'))/member_$(lpad(member, 3, '0'))"
member_key(iteration, member, name) = member_key(iteration, member) * "/" * name
const CURRENT_KEY = "state/current.toml"

read_toml(files, key) = TOML.parse(read(files, key))
write_toml(files, key, dict) = write(files, key, sprint(TOML.print, dict))

"""
    status(files, iteration, member)

One of `:done`, `:failed`, `:claimed`, `:open`, or `:unpublished`.
"""
function status(files::AbstractSharedFiles, iteration, member)
    ispath(files, member_key(iteration, member, "done.toml")) && return :done
    ispath(files, member_key(iteration, member, "failed.toml")) &&
        return :failed
    ispath(files, member_key(iteration, member, "claim")) && return :claimed
    ispath(files, member_key(iteration, member, "task.toml")) && return :open
    return :unpublished
end

"""
    summary(files, iteration, ensemble_size)

Members of the iteration grouped by [`status`](@ref), as a named tuple of
vectors: `open`, `claimed`, `done`, `failed`, `unpublished`.
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

"""
    publish!(files, iteration, member)

Post a member for workers to claim. Returns `false` and changes nothing if
the member is already posted, claimed, done, or failed.
"""
function publish!(files::AbstractSharedFiles, iteration, member)
    status(files, iteration, member) == :unpublished || return false
    write_toml(
        files,
        member_key(iteration, member, "task.toml"),
        Dict(
            "iteration" => iteration,
            "member" => member,
            "attempt" => 1,
            "created" => time(),
        ),
    )
    return true
end

"""
    set_current!(files, iteration, ensemble_size; lease)

Point workers at `iteration`. `lease` is the number of seconds a claim stays
valid without renewal, which both sides read from here.
"""
function set_current!(
    files::AbstractSharedFiles,
    iteration,
    ensemble_size;
    lease,
)
    write_toml(
        files,
        CURRENT_KEY,
        Dict(
            "iteration" => iteration,
            "ensemble_size" => ensemble_size,
            "lease" => Float64(lease),
        ),
    )
    return nothing
end

"""
    current(files)

The iteration workers should look at, as `(; iteration, ensemble_size,
lease)`, or `nothing` before a driver has posted one.
"""
function current(files::AbstractSharedFiles)
    ispath(files, CURRENT_KEY) || return nothing
    c = read_toml(files, CURRENT_KEY)
    return (;
        iteration = c["iteration"],
        ensemble_size = c["ensemble_size"],
        lease = c["lease"],
    )
end

"""
    claim!(files, iteration, member, worker_id)

Try to take an open member. Returns `true` for exactly one of any number of
concurrent callers, and `false` for the rest and for a member that is not
open.
"""
function claim!(files::AbstractSharedFiles, iteration, member, worker_id)
    status(files, iteration, member) == :open || return false
    create(files, member_key(iteration, member, "claim")) || return false
    write_toml(
        files,
        member_key(iteration, member, "claim/owner.toml"),
        Dict("worker_id" => worker_id, "claimed" => time()),
    )
    return true
end

"""
    renew!(files, iteration, member, worker_id)

Extend a claim held by `worker_id`. Returns `false` if the claim is gone or
belongs to another worker, in which case the caller must abandon the member.
"""
function renew!(files::AbstractSharedFiles, iteration, member, worker_id)
    key = member_key(iteration, member, "claim/owner.toml")
    ispath(files, key) || return false
    read_toml(files, key)["worker_id"] == worker_id || return false
    touch(files, key)
    return true
end

"""
    lease_expired(files, iteration, member, lease)

Whether the member's claim has gone `lease` seconds without renewal. A claim
directory with no owner file yet ages from its own modification time. The
age is the driver's clock against a stamp from the worker's, so the two
hosts' clocks must agree to well within `lease`.
"""
function lease_expired(files::AbstractSharedFiles, iteration, member, lease)
    owner = member_key(iteration, member, "claim/owner.toml")
    key = ispath(files, owner) ? owner : member_key(iteration, member, "claim")
    return time() - mtime(files, key) > lease
end

"""
    requeue!(files, iteration, member)

Take a lapsed claim back. The member returns to `:open` with its attempt
count raised, or becomes `:failed` once [`MAX_ATTEMPTS`](@ref) is used up.
Returns the new status.
"""
function requeue!(files::AbstractSharedFiles, iteration, member)
    task = read_toml(files, member_key(iteration, member, "task.toml"))
    rm(files, member_key(iteration, member, "claim"))
    attempt = task["attempt"] + 1
    if attempt > MAX_ATTEMPTS
        write_toml(
            files,
            member_key(iteration, member, "failed.toml"),
            Dict(
                "attempt" => task["attempt"],
                "worker_id" => "",
                "runtime_s" => 0.0,
                "error" => "no completion after $(task["attempt"]) attempts",
            ),
        )
        return :failed
    end
    task["attempt"] = attempt
    write_toml(files, member_key(iteration, member, "task.toml"), task)
    return :open
end

"""
    complete!(files, iteration, member; worker_id, runtime_s)

Record that the member's forward model finished. Written after the member's
outputs.
"""
function complete!(
    files::AbstractSharedFiles,
    iteration,
    member;
    worker_id,
    runtime_s,
)
    write_toml(
        files,
        member_key(iteration, member, "done.toml"),
        Dict(
            "attempt" => attempt(files, iteration, member),
            "worker_id" => worker_id,
            "runtime_s" => Float64(runtime_s),
            "completed" => time(),
        ),
    )
    return nothing
end

"""
    fail!(files, iteration, member; worker_id, runtime_s, error)

Record that the member's forward model raised. `error` is the message and
stack trace as text. Not retried within the iteration.
"""
function fail!(
    files::AbstractSharedFiles,
    iteration,
    member;
    worker_id,
    runtime_s,
    error,
)
    write_toml(
        files,
        member_key(iteration, member, "failed.toml"),
        Dict(
            "attempt" => attempt(files, iteration, member),
            "worker_id" => worker_id,
            "runtime_s" => Float64(runtime_s),
            "error" => string(error),
        ),
    )
    return nothing
end

"""
    reset!(files, iteration, member)

Withdraw a failed member so that [`publish!`](@ref) posts it afresh, at
attempt 1.
"""
function reset!(files::AbstractSharedFiles, iteration, member)
    rm(files, member_key(iteration, member, "failed.toml"))
    rm(files, member_key(iteration, member, "claim"))
    rm(files, member_key(iteration, member, "task.toml"))
    return nothing
end

"""
    attempt(files, iteration, member)

The member's current attempt number.
"""
attempt(files::AbstractSharedFiles, iteration, member) =
    read_toml(files, member_key(iteration, member, "task.toml"))["attempt"]

end # module TaskBoard

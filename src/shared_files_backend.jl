# The driver and worker halves of `SharedFilesBackend`. Both sides speak
# through `TaskBoard` and never to each other.

const PREFIX_FILE = "shared_files_prefix"

"""
The files this calibration meets its workers in: `output_dir` itself, or
`backend.files` under a prefix made on the first call and kept in
`output_dir`, so a resumed calibration finds its own members and a fresh one
finds none.
"""
function shared_files(backend::SharedFilesBackend, output_dir)
    isnothing(backend.files) && return LocalFiles(output_dir)
    path = joinpath(output_dir, PREFIX_FILE)
    if !isfile(path)
        mkpath(output_dir)
        write(path, "$(basename(output_dir))-$(round(Int, time()))")
    end
    return SharedFiles.under(backend.files, read(path, String))
end

function initialize_backend!(
    backend::SharedFilesBackend,
    interface::AbstractModelInterface,
    output_dir,
)
    isnothing(backend.workers) && return nothing
    Backend.launch(
        backend.workers;
        files = shared_files(backend, output_dir),
        model_interface = ClimaCalibrate.model_interface_filepath(interface),
        output_dir,
    )
    return nothing
end

"""
    run_iteration(
        backend::SharedFilesBackend,
        interface::AbstractModelInterface,
        iter,
        ensemble_size,
        output_dir,
    )

Post every member of the `iter`th iteration for workers to claim, then wait
until each one is done or failed, taking back claims whose lease lapsed.
"""
function run_iteration(
    backend::SharedFilesBackend,
    interface::AbstractModelInterface,
    iter,
    ensemble_size,
    output_dir,
)
    staged = !isnothing(backend.files)
    files = shared_files(backend, output_dir)
    for member in 1:ensemble_size
        status = TaskBoard.status(files, iter, member)
        # A rerun of the iteration retries its failed members
        status == :failed && TaskBoard.reset!(files, iter, member)
        status in (:failed, :unpublished) || continue
        staged && upload_inputs(files, output_dir, iter, member)
        TaskBoard.publish!(files, iter, member)
    end
    TaskBoard.set_current!(
        files,
        iter,
        ensemble_size;
        lease = backend.lease,
        outputs = backend.outputs,
    )
    @info "Posted iteration $iter, waiting for workers" files

    unclaimed_since = time()
    groups = TaskBoard.summary(files, iter, 0)
    while length(groups.done) + length(groups.failed) < ensemble_size
        for member in groups.claimed
            TaskBoard.lease_expired(files, iter, member, backend.lease) ||
                continue
            outcome = TaskBoard.requeue!(files, iter, member)
            @warn "Member $member's claim lapsed, it is now $outcome"
        end
        # The clock runs only while members are open and none is claimed
        if isempty(groups.open) || !isempty(groups.claimed)
            unclaimed_since = time()
        elseif time() - unclaimed_since > backend.wait_timeout
            error("no worker claimed a member of iteration $iter within \
                   $(backend.wait_timeout) s")
        end
        sleep(backend.poll_interval)
        before, groups = groups, TaskBoard.summary(files, iter, ensemble_size)
        report(files, iter, before, groups)
    end

    for member in groups.done
        model_completed(output_dir, iter, member) && continue
        staged && download_outputs(files, output_dir, iter, member)
        write_model_completed(output_dir, iter, member)
    end
    check_failure_rate(length(groups.failed), ensemble_size, backend, iter)
    return nothing
end

# Log what changed between two summaries of an iteration
function report(files, iteration, before, after)
    for member in setdiff(after.claimed, before.claimed)
        owner = TaskBoard.owner(files, iteration, member)
        @info "Member $member of iteration $iteration claimed by $owner"
    end
    for member in setdiff(after.done, before.done)
        owner = TaskBoard.owner(files, iteration, member)
        @info "Member $member of iteration $iteration done on $owner"
    end
    for member in setdiff(after.failed, before.failed)
        failure = TaskBoard.failure(files, iteration, member)
        @warn "Member $member of iteration $iteration failed\n$failure"
    end
    return nothing
end

"""
    run_worker(
        interface::AbstractModelInterface,
        output_dir;
        files = nothing,
        worker_id = "<hostname>-<pid>",
        idle_timeout = 600,
        poll_interval = 5,
    )
    run_worker()

Claim and run ensemble members posted by a [`SharedFilesBackend`](@ref)
driver until none have been available for `idle_timeout` seconds. Returns
the number of members this worker ran.

Without `files`, `interface` and `output_dir` must be the ones the driver
uses, since each member runs in place in `output_dir`. With `files`, the
driver's shared files under the calibration's prefix, as the driver logs
them, `output_dir` is this worker's own directory: each member's
`parameters.toml` is downloaded into it before the forward model runs, and
the backend's `outputs` are uploaded afterwards.

The lease length comes from the driver. Start workers with at least two
threads: the claim is renewed from a second task while the forward model
runs.

A worker is a Julia process that builds the interface the way the driver
script does, then calls this:
```julia
julia --project=experiment --threads=2 -e '
    using ClimaCalibrate
    run_worker(include("model_interface.jl"), "output")'
```

Without arguments, the worker is set up by its launcher's environment:
`CLIMA_MODEL_INTERFACE` is the interface file, which evaluates to the
interface, `CLIMA_OUTPUT_DIR` the worker's directory, and `CLIMA_FILES` a
[`BucketFiles`](@ref ClimaCalibrate.SharedFiles.BucketFiles) as
`<endpoint>/<bucket>/<prefix>`.
"""
function run_worker(
    interface::AbstractModelInterface,
    output_dir;
    files = nothing,
    worker_id = "$(gethostname())-$(getpid())",
    idle_timeout = 600,
    poll_interval = 5,
)
    staged = !isnothing(files)
    files = staged ? files : LocalFiles(output_dir)
    @info "Worker $worker_id looking for members" files

    ran = 0
    idle_since = time()
    while time() - idle_since < idle_timeout
        current = TaskBoard.current(files)
        if !isnothing(current)
            (; iteration, ensemble_size) = current
            groups = TaskBoard.summary(files, iteration, ensemble_size)
            member = claim_any!(files, iteration, groups.open, worker_id)
            if !isnothing(member)
                run_claimed_member(
                    files,
                    interface,
                    iteration,
                    member;
                    worker_id,
                    lease = current.lease,
                    output_dir,
                    outputs = staged ? current.outputs : nothing,
                )
                ran += 1
                idle_since = time()
                continue
            end
            # Peers are still running this iteration, so the worker is waiting
            # for the update rather than idle after the run
            isempty(groups.claimed) || (idle_since = time())
        end
        sleep(poll_interval)
    end
    @info "Worker $worker_id found nothing to run for $idle_timeout s, \
           exiting" ran
    return ran
end

run_worker() = run_worker(
    Base.include(Main, ENV["CLIMA_MODEL_INTERFACE"]),
    ENV["CLIMA_OUTPUT_DIR"];
    files = SharedFiles.BucketFiles(ENV["CLIMA_FILES"]),
)

"Claim one of the `open` members of `iteration`, or return `nothing`."
function claim_any!(files, iteration, open, worker_id)
    for member in open
        TaskBoard.claim!(files, iteration, member, worker_id) && return member
    end
    return nothing
end

"""
    run_claimed_member(files, interface, iteration, member;
                       worker_id, lease, output_dir, outputs)

Run one claimed member's forward model, renewing the claim meanwhile, and
record the outcome. `outputs` is `nothing` when the member runs in place.
If the claim was taken away during the run, nothing is recorded.
"""
function run_claimed_member(
    files,
    interface,
    iteration,
    member;
    worker_id,
    lease,
    output_dir,
    outputs,
)
    @info "Worker $worker_id running member $member of iteration $iteration"

    running = Threads.Atomic{Bool}(true)
    renewal = Threads.@spawn while running[]
        sleep(lease / 10)
        still_owned = try
            TaskBoard.renew!(files, iteration, member, worker_id)
        catch e
            @warn "Transient error renewing claim on member $member" exception = e
            true
        end
        still_owned || break
    end

    error_text = nothing
    uploaded = String[]
    try
        isnothing(outputs) ||
            download_inputs(files, output_dir, iteration, member)
        ClimaCalibrate.forward_model(interface, iteration, member)
        isnothing(outputs) || (
            uploaded =
                upload_outputs(files, output_dir, iteration, member, outputs)
        )
    catch e
        e isa InterruptException && rethrow()
        error_text = sprint(showerror, e, catch_backtrace())
    finally
        running[] = false
    end
    wait(renewal)

    # The claim may have lapsed between renewals, so ownership is checked
    # once more before anything is recorded
    if !TaskBoard.renew!(files, iteration, member, worker_id)
        @warn "Worker $worker_id lost its claim on member $member, leaving it \
               to the new owner"
    elseif isnothing(error_text)
        TaskBoard.complete!(
            files,
            iteration,
            member;
            worker_id,
            outputs = uploaded,
        )
        @info "Worker $worker_id completed member $member"
    else
        TaskBoard.fail!(files, iteration, member; worker_id, error = error_text)
        @error "Member $member failed on worker $worker_id\n$error_text"
    end
    return nothing
end

# With `files` elsewhere than `output_dir`, a member's `parameters.toml` and
# its outputs travel as files named relative to the member directory

function upload_inputs(files, output_dir, iteration, member)
    SharedFiles.upload(
        files,
        TaskBoard.member_key(iteration, member, DEFAULT_PARAMETER_FILE),
        parameter_path(output_dir, iteration, member),
    )
    return nothing
end

function download_inputs(files, output_dir, iteration, member)
    SharedFiles.download(
        files,
        TaskBoard.member_key(iteration, member, DEFAULT_PARAMETER_FILE),
        parameter_path(output_dir, iteration, member),
    )
    return nothing
end

function download_outputs(files, output_dir, iteration, member)
    dir = path_to_ensemble_member(output_dir, iteration, member)
    for name in TaskBoard.outputs(files, iteration, member)
        SharedFiles.download(
            files,
            TaskBoard.member_key(iteration, member, name),
            joinpath(dir, name),
        )
    end
    return nothing
end

# Returns the files uploaded, with a directory in `outputs` expanded
function upload_outputs(files, output_dir, iteration, member, outputs)
    dir = path_to_ensemble_member(output_dir, iteration, member)
    uploaded = String[]
    for output in outputs, path in files_under(joinpath(dir, output))
        name = relpath(path, dir)
        SharedFiles.upload(
            files,
            TaskBoard.member_key(iteration, member, name),
            path,
        )
        push!(uploaded, name)
    end
    return uploaded
end

files_under(path) =
    isdir(path) ?
    [joinpath(root, f) for (root, _, names) in walkdir(path) for f in names] :
    [path]

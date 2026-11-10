# The driver and worker halves of `SharedFilesBackend`. Both sides speak
# through `TaskBoard` and never to each other.

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
    output_dir = abspath(output_dir)
    files = LocalFiles(output_dir)
    for member in 1:ensemble_size
        # A rerun of the iteration retries its failed members
        TaskBoard.status(files, iter, member) == :failed &&
            TaskBoard.reset!(files, iter, member)
        TaskBoard.publish!(files, iter, member)
    end
    TaskBoard.set_current!(files, iter, ensemble_size; lease = backend.lease)
    @info "Posted iteration $iter, waiting for workers" files

    counts = nothing
    unclaimed_since = time()
    while true
        groups = TaskBoard.summary(files, iter, ensemble_size)
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
        new_counts = map(length, groups)
        if new_counts != counts
            counts = new_counts
            @info "Iteration $iter" counts...
        end
        length(groups.done) + length(groups.failed) == ensemble_size && break
        sleep(backend.poll_interval)
    end

    groups = TaskBoard.summary(files, iter, ensemble_size)
    foreach(m -> write_model_completed(output_dir, iter, m), groups.done)
    check_failure_rate(length(groups.failed), ensemble_size, backend, iter)
    return nothing
end

"""
    run_worker(
        interface::AbstractModelInterface,
        output_dir;
        worker_id = "<hostname>-<pid>",
        idle_timeout = 600,
        poll_interval = 5,
    )

Claim and run ensemble members posted by a [`SharedFilesBackend`](@ref)
driver until none have been available for `idle_timeout` seconds. Returns
the number of members this worker ran.

`interface` and `output_dir` must be the ones the driver uses, since each
member runs in place in `output_dir`. The lease length comes from the driver.
Start workers with at least two threads: the claim is renewed from a second
task while the forward model runs.

A worker is a Julia process that builds the interface the way the driver
script does, then calls this:
```julia
julia --project=experiment --threads=2 -e '
    include("model_interface.jl")
    using ClimaCalibrate
    run_worker(MyModel(config), "output")'
```
"""
function run_worker(
    interface::AbstractModelInterface,
    output_dir;
    worker_id = "$(gethostname())-$(getpid())",
    idle_timeout = 600,
    poll_interval = 5,
)
    output_dir = abspath(output_dir)
    files = LocalFiles(output_dir)
    Threads.nthreads() == 1 &&
        @warn "Worker $worker_id has one thread, so its claims are only \
               renewed while the forward model yields"
    @info "Worker $worker_id looking for members" files

    ran = 0
    idle_since = time()
    while time() - idle_since < idle_timeout
        current = TaskBoard.current(files)
        member = if isnothing(current)
            nothing
        else
            claim_any!(files, current.iteration, current.ensemble_size, worker_id)
        end
        if isnothing(member)
            sleep(poll_interval)
            continue
        end
        run_claimed_member(
            files,
            interface,
            current.iteration,
            member;
            worker_id,
            lease = current.lease,
        )
        ran += 1
        idle_since = time()
    end
    @info "Worker $worker_id found nothing to run for $idle_timeout s, \
           exiting" ran
    return ran
end

"""
    claim_any!(files, iteration, ensemble_size, worker_id)

Claim one open member of the iteration, or return `nothing`. The members
are tried starting from one chosen by `worker_id`.
"""
function claim_any!(files, iteration, ensemble_size, worker_id)
    open = TaskBoard.summary(files, iteration, ensemble_size).open
    isempty(open) && return nothing
    for member in circshift(open, hash(worker_id) % length(open))
        TaskBoard.claim!(files, iteration, member, worker_id) && return member
    end
    return nothing
end

"""
    run_claimed_member(files, interface, iteration, member; worker_id, lease)

Run one claimed member's forward model, renewing the claim meanwhile, and
record the outcome. If the claim was taken away during the run, nothing is
recorded.
"""
function run_claimed_member(
    files,
    interface,
    iteration,
    member;
    worker_id,
    lease,
)
    @info "Worker $worker_id running member $member of iteration $iteration"

    running = Threads.Atomic{Bool}(true)
    renewal = Threads.@spawn while running[]
        sleep(lease / 10)
        TaskBoard.renew!(files, iteration, member, worker_id) || break
    end

    started = time()
    error_text = nothing
    try
        ClimaCalibrate.forward_model(interface, iteration, member)
    catch e
        e isa InterruptException && rethrow()
        error_text = sprint(showerror, e, catch_backtrace())
    finally
        running[] = false
    end
    wait(renewal)
    runtime_s = time() - started

    # The claim may have lapsed between renewals, so ownership is checked
    # once more before anything is recorded
    if !TaskBoard.renew!(files, iteration, member, worker_id)
        @warn "Worker $worker_id lost its claim on member $member, leaving it \
               to the new owner"
    elseif isnothing(error_text)
        TaskBoard.complete!(files, iteration, member; worker_id, runtime_s)
        @info "Worker $worker_id completed member $member in \
               $(round(runtime_s; digits = 1)) s"
    else
        TaskBoard.fail!(
            files,
            iteration,
            member;
            worker_id,
            runtime_s,
            error = error_text,
        )
        @error "Member $member failed on worker $worker_id\n$error_text"
    end
    return nothing
end

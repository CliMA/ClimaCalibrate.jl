using Test
import ClimaCalibrate.SharedFiles: LocalFiles
import ClimaCalibrate.TaskBoard as TB

@testset "post, claim, finish" begin
    f = LocalFiles(mktempdir())
    @test TB.status(f, 1, 1) == :unpublished
    @test TB.publish!(f, 1, 1)
    @test !TB.publish!(f, 1, 1)
    @test TB.status(f, 1, 1) == :open
    @test TB.attempt(f, 1, 1) == 1

    @test !TB.claim!(f, 1, 2, "a")
    @test TB.claim!(f, 1, 1, "a")
    @test !TB.claim!(f, 1, 1, "b")
    @test TB.status(f, 1, 1) == :claimed
    @test TB.renew!(f, 1, 1, "a")
    @test !TB.renew!(f, 1, 1, "b")

    TB.complete!(f, 1, 1; worker_id = "a")
    @test TB.status(f, 1, 1) == :done
    @test !TB.claim!(f, 1, 1, "b")
    done = TB.read_toml(f, TB.member_key(1, 1, "done.toml"))
    @test done["worker_id"] == "a"
    @test done["attempt"] == 1

    TB.publish!(f, 1, 2)
    TB.claim!(f, 1, 2, "a")
    TB.fail!(f, 1, 2; worker_id = "a", error = "boom")
    @test TB.status(f, 1, 2) == :failed
    @test TB.read_toml(f, TB.member_key(1, 2, "failed.toml"))["error"] == "boom"

    TB.publish!(f, 1, 3)
    @test TB.summary(f, 1, 4) == (;
        open = [3],
        claimed = Int[],
        done = [1],
        failed = [2],
        unpublished = [4],
    )
end

@testset "claims are exclusive" begin
    f = LocalFiles(mktempdir())
    TB.publish!(f, 1, 1)
    winners = Threads.Atomic{Int}(0)
    @sync for i in 1:64
        Threads.@spawn TB.claim!(f, 1, 1, "w$i") &&
                       Threads.atomic_add!(winners, 1)
    end
    @test winners[] == 1
end

@testset "a lapsed claim is handed out again" begin
    f = LocalFiles(mktempdir())
    TB.publish!(f, 1, 1)
    TB.claim!(f, 1, 1, "a")
    @test !TB.lease_expired(f, 1, 1, 10.0)
    sleep(0.2)
    @test TB.lease_expired(f, 1, 1, 0.1)
    TB.renew!(f, 1, 1, "a")
    @test !TB.lease_expired(f, 1, 1, 5.0)

    @test TB.requeue!(f, 1, 1) == :open
    @test TB.attempt(f, 1, 1) == 2
    # The old owner must find out it was taken away
    @test !TB.renew!(f, 1, 1, "a")

    @test TB.claim!(f, 1, 1, "b")
    @test TB.requeue!(f, 1, 1) == :open
    @test TB.claim!(f, 1, 1, "c")
    @test TB.requeue!(f, 1, 1) == :failed
    @test TB.status(f, 1, 1) == :failed
    failed = TB.read_toml(f, TB.member_key(1, 1, "failed.toml"))
    @test failed["attempt"] == TB.MAX_ATTEMPTS
    @test occursin("$(TB.MAX_ATTEMPTS) attempts", failed["error"])
    @test !TB.claim!(f, 1, 1, "d")
end

@testset "current iteration" begin
    f = LocalFiles(mktempdir())
    @test isnothing(TB.current(f))
    TB.set_current!(f, 2, 5; lease = 30)
    @test TB.current(f) == (; iteration = 2, ensemble_size = 5, lease = 30.0)
end

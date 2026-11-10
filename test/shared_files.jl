using Test
import ClimaCalibrate
import ClimaCalibrate.SharedFiles: LocalFiles, create

@testset "LocalFiles" begin
    s = LocalFiles(mktempdir())

    @testset "read and write" begin
        write(s, "a/b/c.txt", "hello")
        @test read(s, "a/b/c.txt") == "hello"
        write(s, "a/b/c.txt", "again")
        @test read(s, "a/b/c.txt") == "again"
        @test_throws SystemError read(s, "missing")
    end

    @testset "ispath, rm, mtime, touch" begin
        @test ispath(s, "a/b/c.txt")
        @test !ispath(s, "a/b/d.txt")
        t0 = mtime(s, "a/b/c.txt")
        sleep(0.05)
        touch(s, "a/b/c.txt")
        @test mtime(s, "a/b/c.txt") > t0
        rm(s, "a/b/c.txt")
        @test !ispath(s, "a/b/c.txt")
        rm(s, "a")
        @test !ispath(s, "a")
        rm(s, "a")
    end

    @testset "create is exclusive" begin
        @test create(s, "iteration_000/member_001/claim")
        @test !create(s, "iteration_000/member_001/claim")
        write(s, "iteration_000/member_001/claim/owner.toml", "")
        @test ispath(s, "iteration_000/member_001/claim/owner.toml")

        winners = Threads.Atomic{Int}(0)
        @sync for _ in 1:64
            Threads.@spawn create(s, "race") && Threads.atomic_add!(winners, 1)
        end
        @test winners[] == 1
    end
end

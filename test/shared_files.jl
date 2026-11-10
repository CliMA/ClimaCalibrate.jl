using Test
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
        rm(s, "a/b/c.txt")
    end

    @testset "create is exclusive" begin
        @test create(s, "x/claim.toml", "first")
        @test !create(s, "x/claim.toml", "second")
        @test read(s, "x/claim.toml") == "first"

        winners = Threads.Atomic{Int}(0)
        @sync for i in 1:64
            Threads.@spawn create(s, "race", "$i") &&
                           Threads.atomic_add!(winners, 1)
        end
        @test winners[] == 1
    end
end

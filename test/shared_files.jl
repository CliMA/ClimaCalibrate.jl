using Test
import ClimaCalibrate
import ClimaCalibrate.SharedFiles:
    LocalFiles, BucketFiles, create, upload, download

stores = Any[LocalFiles(mktempdir())]

# A bucket is tested when one is reachable. For MinIO in Docker:
#   docker run -p 9000:9000 -e MINIO_ROOT_USER=minio \
#       -e MINIO_ROOT_PASSWORD=minio123 minio/minio server /data
#   export CLIMA_TEST_BUCKET_ENDPOINT=http://localhost:9000 \
#       CLIMA_TEST_BUCKET=test AWS_ACCESS_KEY_ID=minio \
#       AWS_SECRET_ACCESS_KEY=minio123
if haskey(ENV, "CLIMA_TEST_BUCKET_ENDPOINT")
    import AWSS3
    bucket = BucketFiles(
        ENV["CLIMA_TEST_BUCKET_ENDPOINT"],
        ENV["CLIMA_TEST_BUCKET"],
        "test-$(getpid())";
        region = "us-east-1",
    )
    ext = Base.get_extension(ClimaCalibrate, :ClimaCalibrateAWSS3Ext)
    try
        AWSS3.s3_create_bucket(ext.config(bucket), bucket.bucket)
    catch e
        ext.status(e) == 409 || rethrow()
    end
    push!(stores, bucket)
end

@testset "$(nameof(typeof(s)))" for s in stores
    @testset "read and write" begin
        write(s, "a/b/c.txt", "hello")
        @test read(s, "a/b/c.txt") == "hello"
        write(s, "a/b/c.txt", "again")
        @test read(s, "a/b/c.txt") == "again"
        @test_throws Exception read(s, "missing")
    end

    @testset "ispath, rm, mtime, touch" begin
        @test ispath(s, "a/b/c.txt")
        @test !ispath(s, "a/b/d.txt")
        @test mtime(s, "a/b/d.txt") == 0.0
        t0 = mtime(s, "a/b/c.txt")
        # Bucket stamps have one-second resolution
        sleep(1.1)
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
        rm(s, "x/claim.toml")
        rm(s, "race")
    end

    @testset "upload and download" begin
        dir = mktempdir()
        bytes = rand(UInt8, 1000)
        write(joinpath(dir, "in.bin"), bytes)
        upload(s, "x/in.bin", joinpath(dir, "in.bin"))
        download(s, "x/in.bin", joinpath(dir, "out", "in.bin"))
        @test read(joinpath(dir, "out", "in.bin")) == bytes
        rm(s, "x/in.bin")
    end
end

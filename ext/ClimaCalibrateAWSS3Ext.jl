module ClimaCalibrateAWSS3Ext

import AWSS3
import AWSS3: AWS, S3, Dates
import ClimaCalibrate.SharedFiles: BucketFiles, create

function __init__()
    opts = if pkgversion(AWS.HTTP) >= v"2"
        (;
            connect_timeout = 30,
            read_idle_timeout = 60,
            write_idle_timeout = 60,
        )
    else
        (; connect_timeout = 30, readtimeout = 60)
    end
    AWS.DEFAULT_BACKEND[] = AWS.HTTPBackend(; opts...)
    return nothing
end

# An AWS.jl config whose service URL is the bucket's endpoint, which is how
# the S3 API of R2 or MinIO is reached
struct Endpoint <: AWS.AbstractAWSConfig
    url::String
    region::String
    credentials::AWS.AWSCredentials
end

AWS.credentials(c::Endpoint) = c.credentials
AWS.region(c::Endpoint) = c.region
AWS.generate_service_url(c::Endpoint, ::String, resource::String) =
    c.url * resource

config(f::BucketFiles) = Endpoint(
    f.endpoint,
    f.region,
    @something(f.credentials, AWS.AWSCredentials()),
)

key(f::BucketFiles, k) = isempty(f.prefix) ? String(k) : f.prefix * "/" * k

status(e) = e isa AWS.AWSException ? e.cause.status : 0

Base.read(f::BucketFiles, k::AbstractString) =
    String(AWSS3.s3_get(config(f), f.bucket, key(f, k); raw = true))

const MULTIPART_MB = 16

function Base.write(f::BucketFiles, k::AbstractString, data)
    if sizeof(data) > MULTIPART_MB * 1024 * 1024
        AWSS3.s3_multipart_upload(
            config(f),
            f.bucket,
            key(f, k),
            IOBuffer(data),
            MULTIPART_MB,
        )
    else
        AWSS3.s3_put(config(f), f.bucket, key(f, k), data)
    end
    return nothing
end

function Base.rm(f::BucketFiles, k::AbstractString)
    AWSS3.s3_delete(config(f), f.bucket, key(f, k))
    return nothing
end

# Response headers of a HEAD, or `nothing` for an absent object
function head(f::BucketFiles, k)
    try
        return AWSS3.s3_get_meta(config(f), f.bucket, key(f, k))
    catch e
        status(e) == 404 && return nothing
        rethrow()
    end
end

Base.ispath(f::BucketFiles, k::AbstractString) = !isnothing(head(f, k))

function Base.mtime(f::BucketFiles, k::AbstractString)
    headers = head(f, k)
    isnothing(headers) && return 0.0
    stamp =
        only(v for (name, v) in headers if lowercase(name) == "last-modified")
    return Dates.datetime2unix(
        Dates.DateTime(chopsuffix(stamp, " GMT"), Dates.RFC1123Format),
    )
end

# Copying the object onto itself with new metadata refreshes its
# modification time. An absent object stays absent
function Base.touch(f::BucketFiles, k::AbstractString)
    try
        AWSS3.s3_copy(
            config(f),
            f.bucket,
            key(f, k);
            metadata = Dict("touched" => string(time())),
        )
    catch e
        status(e) == 404 || rethrow()
    end
    return nothing
end

# 412 is an existing object and 409 a concurrent conditional write that may
# win, so both are a loss
function create(f::BucketFiles, k::AbstractString, data)
    try
        S3.put_object(
            f.bucket,
            key(f, k),
            Dict("body" => data, "headers" => Dict("If-None-Match" => "*"));
            aws_config = config(f),
        )
    catch e
        status(e) in (409, 412) && return false
        rethrow()
    end
    return true
end

end # module ClimaCalibrateAWSS3Ext

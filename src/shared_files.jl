"""
    ClimaCalibrate.SharedFiles

Files the driver and the workers of a decoupled calibration can both see:
tasks, claims, markers, and results.

Entries are addressed by keys, relative paths joined with `/`. The collection
behaves like a filesystem root: `read`, `write`, `rm`, `ispath`,
`mtime`, and `touch` are extended for it, and [`create`](@ref) adds the one
operation a filesystem lacks, a create-if-absent that exactly one of several
concurrent callers wins.

[`LocalFiles`](@ref) is a directory. A bucket implementation follows the same
interface.
"""
module SharedFiles

export AbstractSharedFiles, LocalFiles, create

"""
    AbstractSharedFiles

A keyed collection of files. Subtypes implement, for `f::AbstractSharedFiles`
and a key `String`: `read(f, key)::String`, `write(f, key, data)`,
`rm(f, key)`, `ispath(f, key)`, `mtime(f, key)`,
`touch(f, key)`, and [`create`](@ref).
"""
abstract type AbstractSharedFiles end

"""
    create(files, key)::Bool

Create an empty entry at `key`, returning `true` if this call created it and
`false` if it already existed. Of any number of concurrent callers exactly one
receives `true`.

Parents are created as needed.
"""
function create end

"""
    LocalFiles(root)

Shared files on a local or shared filesystem, rooted at `root`.
[`create`](@ref) is `mkdir`, so the entry it makes is a directory. Writes are
renamed into place.
"""
struct LocalFiles <: AbstractSharedFiles
    root::String
end

path(s::LocalFiles, key::AbstractString) = joinpath(s.root, key)

Base.read(s::LocalFiles, key::AbstractString) = read(path(s, key), String)

function Base.write(s::LocalFiles, key::AbstractString, data)
    dest = path(s, key)
    mkpath(dirname(dest))
    # Same directory as the destination, so the rename is atomic
    tmp = tempname(dirname(dest); cleanup = false)
    try
        write(tmp, data)
        Base.Filesystem.rename(tmp, dest)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    return nothing
end

Base.rm(s::LocalFiles, key::AbstractString) =
    rm(path(s, key); recursive = true, force = true)

Base.ispath(s::LocalFiles, key::AbstractString) = ispath(path(s, key))

Base.mtime(s::LocalFiles, key::AbstractString) = mtime(path(s, key))

Base.touch(s::LocalFiles, key::AbstractString) = (touch(path(s, key)); nothing)

function create(s::LocalFiles, key::AbstractString)
    dest = path(s, key)
    mkpath(dirname(dest))
    try
        mkdir(dest)
    catch e
        e isa Base.IOError && ispath(dest) && return false
        rethrow()
    end
    return true
end

end # module SharedFiles

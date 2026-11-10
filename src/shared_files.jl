"""
    ClimaCalibrate.SharedFiles

Files a driver and its workers can both see, addressed by keys, relative paths
joined with `/`. `read`, `write`, `rm`, `ispath`, `mtime`, and `touch` are
extended for a collection, and [`create`](@ref) is a create-if-absent that
exactly one of several concurrent callers wins. [`LocalFiles`](@ref) is a
directory.
"""
module SharedFiles

"A keyed collection of files. See [`SharedFiles`](@ref) for the operations."
abstract type AbstractSharedFiles end

"""
    create(files, key, data)::Bool

Write `data` at `key` if nothing is there yet. Returns whether this call
created the entry. Of any number of concurrent callers exactly one receives
`true`.
"""
function create end

"""
    LocalFiles(root)

Shared files in the directory `root`. Writes are renamed into place and
`create` opens with `O_EXCL`.
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

Base.rm(s::LocalFiles, key::AbstractString) = rm(path(s, key); force = true)

Base.ispath(s::LocalFiles, key::AbstractString) = ispath(path(s, key))

Base.mtime(s::LocalFiles, key::AbstractString) = mtime(path(s, key))

Base.touch(s::LocalFiles, key::AbstractString) = (touch(path(s, key)); nothing)

const EXCLUSIVE =
    Base.Filesystem.JL_O_WRONLY |
    Base.Filesystem.JL_O_CREAT |
    Base.Filesystem.JL_O_EXCL

function create(s::LocalFiles, key::AbstractString, data)
    dest = path(s, key)
    mkpath(dirname(dest))
    file = try
        Base.Filesystem.open(dest, EXCLUSIVE, 0o644)
    catch e
        e isa Base.IOError && e.code == Base.UV_EEXIST && return false
        rethrow()
    end
    try
        write(file, data)
    finally
        close(file)
    end
    return true
end

end # module SharedFiles

import ClimaCalibrate.Report

"""
    _print_variables(io, metadata::AbstractVector)

Print the short name, long name, and units of the variable of each element of
`metadata`, followed by its dimensions and number of entries.
"""
function Report._print_variables(io, metadata::AbstractVector)
    all(md -> md isa Metadata, metadata) || return nothing
    for md in metadata
        # Print the header which looks like "pr (precipitation, kg m^-2 s^-1)"
        details = [ClimaAnalysis.long_name(md), ClimaAnalysis.units(md)]
        filter!(!isempty, details)
        heading = ClimaAnalysis.short_name(md)
        isempty(details) || (heading *= " ($(join(details, ", ")))")
        println(io, heading)

        # Print information about the dimension and length of the data
        rows = [
            dim => _describe_dim(md, dim) for dim in ClimaAnalysis.dim_names(md)
        ]
        push!(rows, "entries" => _describe_entries(md))
        for line in eachline(IOBuffer(sprint(Report.print_aligned, rows)))
            println(io, "  ", line)
        end
    end
    return nothing
end

"""
    _describe_dim(metadata::Metadata, dim)

Return the number of values of the dimension `dim` of `metadata`, a shortened
list of the values, and their units as a string.

The values of the temporal dimension are dates if possible, and are shown
without units.
"""
function _describe_dim(metadata::Metadata, dim)
    is_time =
        ClimaAnalysis.has_time(metadata) &&
        dim == ClimaAnalysis.time_name(metadata)
    values = is_time ? dates_or_times(metadata) : metadata.dims[dim]
    units = is_time ? "" : ClimaAnalysis.dim_units(metadata, dim)
    shown =
        length(values) > 4 ? [values[1], values[2], "…", values[end]] : values
    description = "$(length(values)) values [$(join(shown, ", "))]"
    return isempty(units) ? description : "$description in $units"
end

"""
    _describe_entries(metadata::Metadata)

Return a string with the number of entries of the flattened variable described
by `metadata` and, if any, the number of entries dropped when flattening.
"""
function _describe_entries(metadata::Metadata)
    n_kept = ClimaAnalysis.flattened_length(metadata)
    n_total = ClimaAnalysis.flattened_length(metadata; ignore_dropped = false)
    n_dropped = n_total - n_kept
    return iszero(n_dropped) ? "$n_kept" : "$n_kept ($n_dropped dropped)"
end

"""
    dates_or_times(var_or_metadata::Union{OutputVar, Metadata})

Return the temporal dimension of `var_or_metadata` as dates if possible and time
otherwise.

This function assumes that a temporal dimension exists in `var_or_metadata`.
"""
function dates_or_times(var_or_metadata::Union{OutputVar, Metadata})
    temporal_dim = try
        ClimaAnalysis.dates(var_or_metadata)
    catch
        ClimaAnalysis.times(var_or_metadata)
    end
    return temporal_dim
end

"""
    find_season_and_year(date::Dates.DateTime)

For a given date, find the appopriate season and year.

This function was copied from ClimaAnalysis.
"""
function find_season_and_year(date::Dates.DateTime)
    if Dates.Month(3) <= Dates.Month(date) <= Dates.Month(5)
        return ("MAM", Dates.year(date))
    elseif Dates.Month(6) <= Dates.Month(date) <= Dates.Month(8)
        return ("JJA", Dates.year(date))
    elseif Dates.Month(9) <= Dates.Month(date) <= Dates.Month(11)
        return ("SON", Dates.year(date))
    else
        corrected_year =
            Dates.month(date) == 12 ? Dates.year(date) + 1 : Dates.year(date)
        return ("DJF", corrected_year)
    end
end

"""
    _get_indices_of_metadata(metadata::Iterable{ClimaAnalysis.Var.Metadata})

For a vector of `ClimaAnalysis.Var.Metadata`, return a list of ranges for
indexing.

This is useful when you have a vector created from the concatenation of
multiple flattened `OutputVar`s. The returned ranges let you extract the slice
that belongs to each metadata entry and pass it to `ClimaAnalysis.unflatten` to
reconstruct the original `OutputVar`.
"""
function _get_indices_of_metadata(metadata)
    ranges = UnitRange{Int}[]
    index = 1
    for md in metadata
        data_size = ClimaAnalysis.flattened_length(md)
        start_idx = index
        index += data_size
        push!(ranges, start_idx:(start_idx + data_size - 1))
    end
    return ranges
end

"""
    _reconstruct_vars(data, all_metadata)

Reconstruct a vector of `OutputVar`s from a flat vector `data` and
`all_metadata`, an iterable of `ClimaAnalysis.Var.Metadata`.

The vector `data` is the vertical concatenation of the flattened variables. For
example, this can be a column of a `SampleCollection`, a stacked observation
sample, or the diagonal of a covariance matrix.
"""
function _reconstruct_vars(data, all_metadata)
    ranges = _get_indices_of_metadata(all_metadata)
    return OutputVar[
        ClimaAnalysis.unflatten(metadata, view(data, range)) for
        (metadata, range) in zip(all_metadata, ranges)
    ]
end

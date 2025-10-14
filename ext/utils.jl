"""
    _check_dates_in_var(vars::Iterable{OutputVar}, start_date, end_date)

Check `start_date` and `end_date` are in `vars`.
"""
function _check_dates_in_var(vars, start_date, end_date)
    for var in vars
        start_date in ClimaAnalysis.dates(var) || error(
            "$start_date is not a date in the OutputVar with the short name $(get(var.attributes, "short_name", nothing))",
        )
        end_date in ClimaAnalysis.dates(var) || error(
            "$end_date is not a date in the OutputVar with the short name $(get(var.attributes, "short_name", nothing))",
        )
    end
    return nothing
end

"""
    _vars_to_iterable(vars::Union{OutputVar, Iterable{OutputVar}})

Make `vars` into an iterable if `vars` is an `OutputVar`. Otherwise, return `vars`.
"""
function _vars_to_iterable(vars)
    vars isa OutputVar && (vars = (vars,))
    return vars
end

"""
    split_by_season_from_seconds(time_dim,
                               reference_date;
                               seasons = ("MAM", "JJA", "SON", "DJF"))

Split `time_dim` into a vector of vectors, where each vector represent a single
season.

The order of the seasons can be chosen with the `seasons` keyword argument.

This function differs from `ClimaAnalysis.Utils.split_by_season` as that function
expects dates, while this function expect times in term of seconds.
"""
function split_by_season_from_seconds(
    time_dim,
    reference_date;
    seasons = ("MAM", "JJA", "SON", "DJF"),
)
    reference_date isa AbstractString &&
        (reference_date = Dates.DateTime(reference_date))
    grouped_dates =
        ClimaAnalysis.Utils.time_to_date.(reference_date, time_dim) |>
        (dates -> ClimaAnalysis.Utils.split_by_season(dates, seasons = seasons))
    grouped_times = [
        ClimaAnalysis.Utils.date_to_time.(reference_date, x) for
        x in grouped_dates
    ]
    return grouped_times
end

"""
    find_seasons(start_date, end_date)

Find all the seasons between `start_date` and `end_date`.
"""
function find_seasons(start_date, end_date)
    start_date <= end_date ||
        error("$start_date should not be later than $end_date")
    (first_season, first_year) =
        ClimaAnalysis.Utils.find_season_and_year(start_date)
    # Because the year is determined from the second month, we need to handle
    # the case when the season is DJF
    first_year = first_season == "DJF" ? first_year - 1 : first_year
    season_to_month = Dict("MAM" => 3, "JJA" => 6, "SON" => 9, "DJF" => 12)
    first_date_of_season =
        Dates.DateTime(first_year, season_to_month[first_season], 1)
    curr_date = first_date_of_season
    seasons = String[]
    while curr_date <= end_date
        season, _ = ClimaAnalysis.Utils.find_season_and_year(curr_date)
        push!(seasons, season)
        curr_date += Dates.Month(3) # Season change every three months
    end
    return seasons
end

"""
    check_time_dim(var::OutputVar)

Check time dimension exists, unit for the time dimension is second, and a
start date is present.

This function is borrowed from ClimaAnalysis.
"""
function _check_time_dim(var::OutputVar)
    ClimaAnalysis.has_time(var) || error("Time is not a dimension in var")
    ClimaAnalysis.dim_units(var, ClimaAnalysis.time_name(var)) == "s" ||
        error("Unit for time is not second")
    haskey(var.attributes, "start_date") ||
        error("Start date is not found in var")
    return nothing
end

"""
    dates_or_times(var::OutputVar)

Return the temporal dimension of `var` as dates if possible and time otherwise.

This function assumes that a temporal dimension exists in `var`.
"""
function dates_or_times(var_or_metadata::Union{OutputVar, Metadata})
    temporal_dim = try
        ClimaAnalysis.dates(var_or_metadata)
    catch
        ClimaAnalysis.times(var_or_metadata)
    end
    return temporal_dim
end

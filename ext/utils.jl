# TODO: Move this to climaanalysis_helper.jl and
# rename it too?
"""
    find_seasons(start_date, end_date)

Find all the seasons between `start_date` and `end_date`.
"""
function find_seasons(start_date, end_date)
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

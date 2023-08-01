import CairoMakie: Makie
import ClimaComms
import ClimaCore: Fields, Geometry, Spaces, InputOutput
import Statistics: mean

# Adapted from contours_and_plots.jl in ClimaAtmos

function time_from_filename(file)
    arr = split(basename(file), ".")
    day = parse(Float64, replace(arr[1], "day" => ""))
    sec = parse(Float64, arr[2])
    return day * (60 * 60 * 24) + sec
end

function read_hdf5_file(file_path)
    reader = InputOutput.HDF5Reader(
        file_path,
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
    )
    diagnostics = InputOutput.read_field(reader, "diagnostics")
    close(reader)
    return diagnostics
end

horizontal_space(diagnostics) =
    Spaces.horizontal_space(axes(diagnostics.temperature))

is_on_sphere(diagnostics) =
    eltype(Fields.coordinate_field(horizontal_space(diagnostics))) <:
    Geometry.LatLongPoint

get_column_1(diagnostics) =
    isnothing(diagnostics) ? (nothing, nothing) :
    column_view_diagnostics(diagnostics, ((1, 1), 1))

function column_view_diagnostics(diagnostics, column)
    ((i, j), h) = column
    is_extruded_field(object) =
        axes(object) isa Spaces.ExtrudedFiniteDifferenceSpace
    column_view(field) = Fields.column(field, i, j, h)
    column_zs(object) =
        is_extruded_field(object) ?
        vec(parent(Fields.coordinate_field(column_view(object)).z)) / 1000 :
        nothing
    column_values(object) =
        is_extruded_field(object) ? vec(parent(column_view(object))) : nothing
    objects = Fields._values(diagnostics)

    column_zs_and_values = map(column_zs, objects), map(column_values, objects)

    # Assume that all variables have the same horizontal coordinates.
    coords = Fields.coordinate_field(column_view(diagnostics.temperature))

    coord_strings = map(filter(!=(:z), propertynames(coords))) do symbol
        # Add 0 to every horizontal coordinate value so that -0.0 gets
        # printed without the unnecessary negative sign.
        value = round(mean(getproperty(coords, symbol)); sigdigits = 6) + 0
        return "$symbol = $value"
    end
    col_string = "column data from $(join(coord_strings, ", "))"

    return column_zs_and_values, col_string
end

column_at_coord_getter(latitude, longitude) =
    diagnostics -> begin
        isnothing(diagnostics) && return (nothing, nothing)
        column = if is_on_sphere(diagnostics)
            horz_space = horizontal_space(diagnostics)
            horz_coords = Fields.coordinate_field(horz_space)
            FT = eltype(eltype(horz_coords))
            target_column_coord =
                Geometry.LatLongPoint(FT(latitude), FT(longitude))
            distance_to_target(((i, j), h)) =
                Geometry.great_circle_distance(
                    Spaces.column(horz_coords, i, j, h)[],
                    target_column_coord,
                    horz_space.global_geometry,
                )
            argmin(distance_to_target, Spaces.all_nodes(horz_space))
        else
            ((1, 1), 1) # If the data is not on a sphere, extract column 1.
        end
        return column_view_diagnostics(diagnostics, column)
    end

"""
    latitude_contour_plot(file_path, variable; longitude = 0, out_path = nothing)

Saves a contour plot of latitude versus height for the given variable at the
given longitude. Default longitude = 0
"""
function latitude_contour_plot(
    file_path,
    variable;
    longitude = 0,
    out_path = nothing,
)
    diagnostics = read_hdf5_file(file_path)
    latitudes = -90:5:90
    cols = map(latitudes) do lat
        column_at_coord_getter(lat, longitude)(diagnostics)[1]
    end
    variable_cols = map(cols) do (zs, values)
        getproperty(values, variable)
    end
    zs = getproperty(cols[1][1], variable)
    values = hcat(variable_cols...)

    figure = Makie.Figure()
    axis = Makie.Axis(
        figure[1, 1],
        xlabel = "Latitude",
        ylabel = "Height (km)",
        title = "$variable at $longitude long",
    )
    Makie.contourf!(axis, latitudes, zs, values')
    Makie.contourf!(axis, latitudes, zs, values')
    if isnothing(out_path)
        out_path =
            chop(file_path, head = 0, tail = 5) * "_$(variable)_$longitude.png"
    end
    Makie.save(out_path, figure)
    return nothing
end

import ClimaCalibrate.EnsembleBuilder as EnsembleBuilder
import ClimaCalibrate: g_ens_matrix

include("checkers.jl")

"""
    MetadataInfo{METADATA <: Metadata}

An object that stores the metadata of the observation, the index of the
metadata, and the corresponding range to fill out in the G ensemble matrix.
"""
struct MetadataInfo{METADATA <: Metadata}
    """The index of the metadata of all metadata"""
    index::Int64

    """The indices of column of the current G ensemble matrix to fill out"""
    range::UnitRange{Int64}

    """A single metadata from the metadata for the current minibatch"""
    metadata::METADATA
end

"""
    GEnsembleBuilder{FT <: AbstractFloat}

An object to help build G ensemble matrix by using the metadata stored in the
`EKP.EnsembleKalmanProcess` object. Metadata must come from `ClimaAnalysis`.

`GEnsembleBuilder` takes in preprocessed `OutputVar`s and automatically
construct the corresponding G ensemble matrix for the current iteration of the
calibration.
"""
struct GEnsembleBuilder{
    FT <: AbstractFloat,
    METADATAINFO <: MetadataInfo,
    T <: Tuple{Vararg{AbstractChecker}},
}
    """G ensemble matrix that is returned by the observation map"""
    g_ens::Matrix{FT}

    """Dictionary which map short name to a vector of metadata associated with
       the short name"""
    metadata_by_short_name::Dict{String, Vector{METADATAINFO}}

    """A vector of metadata info ordered by how the observations are combined"""
    metadata_vec::Vector{METADATAINFO}

    """A bit matrix which keeps track of which entries are filled out in the
    G ensemble matrix. The size of this matrix is the number of metadata for the
    minibatch by the number of ensemble members"""
    completed::BitMatrix

    """A list of checkers used to check the OutputVar to the list of metadata"""
    checkers::T
end

"""
    GEnsembleBuilder(ekp::EKP.EnsembleKalmanProcess{FT})
        where {FT <: AbstractFloat}

Construct a `GEnsembleBuilder` where the element type of the G ensemble matrix
is `FT`.
"""
function EnsembleBuilder.GEnsembleBuilder(
    ekp::EKP.EnsembleKalmanProcess{FT},
) where {FT <: AbstractFloat}
    obs_series = EKP.get_observation_series(ekp)
    N = EKP.get_N_iterations(ekp) + 1
    metadatas = ObservationRecipe.get_metadata_for_nth_iteration(obs_series, N)
    eltype(metadatas) <: ClimaAnalysis.Var.Metadata || error(
        "GEnsembleBuilder is only compatible with metadata from ClimaAnalysis",
    )

    # Check all metadata contain a short name
    metadata_short_names =
        collect(ClimaAnalysis.short_name(metadata) for metadata in metadatas)
    any(==(""), metadata_short_names) && error(
        "One of the observations does not has a short name; add a short name to the OutputVar before making an observation from it",
    )

    G_ens = g_ens_matrix(ekp)
    completed = falses(length(metadatas), EKP.get_N_ens(ekp))

    short_name_to_metadata_map = Dict{String, Vector{MetadataInfo}}()
    minibatch_indices = _get_minibatch_indices_for_nth_iteration(obs_series, N)

    # Assume G_ens and minibatch_indices are one-indexed
    @assert first(size(G_ens)) == last(last(minibatch_indices))

    all_metadata_vec = MetadataInfo[]
    for (i, (metadata, range)) in enumerate(zip(metadatas, minibatch_indices))
        short_name = ClimaAnalysis.short_name(metadata)
        metadata_with_short_name_vec =
            get!(short_name_to_metadata_map, short_name, MetadataInfo[])
        metadata_info = MetadataInfo(i, range, metadata)
        push!(metadata_with_short_name_vec, metadata_info)
        push!(all_metadata_vec, metadata_info)
    end

    return GEnsembleBuilder(
        G_ens,
        short_name_to_metadata_map,
        all_metadata_vec,
        completed,
        (
            ShortNameChecker(),
            DimNameChecker(),
            DimUnitsChecker(),
            UnitsChecker(),
            DimValuesChecker(),
        ),
    )
end

"""
    EnsembleBuilder.fill_g_ens_col!(g_ens_builder::GEnsembleBuilder,
                                    col_idx,
                                    vars::OutputVar...;
                                    checkers = (,)
                                    verbose = false)

Fill the `col_idx`th of the G ensemble matrix from the `OutputVar`s `vars` and
`ekp`.

The column of the G ensemble matrix does not need to be filled out completely by
this function. As such, you can call this function with the same `col_idx`, but
different `vars` to gradually fill out the column of the G ensemble matrix.

It is assumed that the times or dates of a single `OutputVar` is a superset of
the times or dates of one or more metadata in the minibatch.

This function relies on the short names in the metadata. This function will not
behave correctly if the short names are mislabled or not present.

Furthermore, this function assumes that all observations are generated using
`ObservationRecipe.Observation` which guarantees that the metadata exists and
the correct placement of metadata.
"""
function EnsembleBuilder.fill_g_ens_col!(
    g_ens_builder::GEnsembleBuilder,
    col_idx,
    vars::OutputVar...;
    checkers = (),
    verbose = false,
)
    # Check all OutputVars contain a short name
    var_short_names = collect(ClimaAnalysis.short_name(var) for var in vars)
    any(==(""), var_short_names) && error(
        "One of the OutputVars does not has a short name; add a short name to the OutputVar before creating it",
    )

    metadata_by_short_name = g_ens_builder.metadata_by_short_name
    for (i, var) in enumerate(vars)
        var_short_name = ClimaAnalysis.short_name(var)
        metadata_info_vec = if haskey(metadata_by_short_name, var_short_name)
            metadata_by_short_name[var_short_name]
        else
            empty(metadata_by_short_name |> first |> last)
        end
        use_var = false
        # Try every metadata_info in the vector, because a single var can be
        # used for multiple metadata
        for metadata_info in metadata_info_vec
            use_var |= _try_fill_g_ens_col_with_var!(
                g_ens_builder,
                col_idx,
                var,
                metadata_info;
                checkers = checkers,
                verbose = verbose,
            )
        end
        use_var || @warn(
            "OutputVar at index $i with the short name $(ClimaAnalysis.short_name(var)) was passed as an input, but did not match with any of the metadata"
        )
    end
    return nothing
end

"""
    EnsembleBuilder.fill_g_ens_col!(g_ens_builder::GEnsembleBuilder,
                                    col_idx,
                                    val::AbstractFloat)

Fill the `col_idx`th column of the G ensemble matrix with `val`.

This is useful if you want to completely fill a column of a G ensemble matrix
with `NaN`s if a simulation crashed.
"""
function EnsembleBuilder.fill_g_ens_col!(
    g_ens_builder::GEnsembleBuilder,
    col_idx,
    val::AbstractFloat,
)
    g_ens_builder.g_ens[:, col_idx] .= val
    g_ens_builder.completed[:, col_idx] .= true
    return nothing
end

"""
    _try_fill_g_ens_col_with_var!(g_ens_builder::GEnsembleBuilder,
                                  col_idx,
                                  var::OutputVar,
                                  metadata_info::MetadataInfo)

Try to fill the `col_idx`th of the G ensemble matrix using `metadata_info` and
`var`. Return `true` if sucessful and `false` if not.
"""
function _try_fill_g_ens_col_with_var!(
    g_ens_builder::GEnsembleBuilder,
    col_idx,
    var::OutputVar,
    metadata_info::MetadataInfo;
    checkers = (),
    verbose = false,
)
    (; metadata, range, index) = metadata_info

    # Call checkers in g_ens_builder and user passed checkers
    _is_compatible_with_metadata(
        g_ens_builder.checkers,
        var,
        metadata;
        verbose = verbose,
    ) || return false
    _is_compatible_with_metadata(checkers, var, metadata; verbose = verbose) ||
        return false

    match_dates_var = _match_dates(var, metadata)
    g_ens_builder.g_ens[range, col_idx] .=
        ClimaAnalysis.flatten(match_dates_var, metadata).data
    if g_ens_builder.completed[index, col_idx]
        @warn(
            "This portion of the G ensemble matrix ($range, $col_idx) is already filled out by another OutputVar. Replacing the contents with the new OutputVar"
        )
    end
    g_ens_builder.completed[index, col_idx] = true
    return true
end

"""
    _is_compatible_with_metadata(
        checkers,
        var::OutputVar,
        metadata::Metadata;
        checkers = (),
        verbose = false,
    )

Return `true` if `var` can be flattened with `metadata` and fill out the
column of the G ensemble matrix corresponding to the `metadata`.
"""
function _is_compatible_with_metadata(
    checkers,
    var::OutputVar,
    metadata::Metadata;
    verbose = false,
)
    return all(
        Checker.check(checker, var, metadata; verbose = verbose) for
        checker in checkers
    )
end

"""
    _match_dates(var::OutputVar, metadata::Metadata)

Return a `OutputVar` using `var` whose dates are the same as the dates in
`metadata`.

If not all dates in `var` can be found in the dates in `metadata`, then
`nothing` is returned.

!!! warning "Matching dates"
    This function will match the dates even if the dates in `var` is out of
    order or there are more dates in `var` than the dates in `metadata`.
    For example, if `var` contains monthly averages and `metadata` represents
    seasonal averages, then this function will not return an error, because
    all dates in `metadata` is in the set of all dates in `var`.
"""
function _match_dates(var::OutputVar, metadata::Metadata)
    # Metadata comes from observational data and var is simulation data
    # when calibrating
    obs_dates = ClimaAnalysis.dates(metadata)
    sim_dates = ClimaAnalysis.dates(var)
    for dates in (obs_dates, sim_dates)
        allunique(dates) || error("Dates in $dates are not unique")
    end
    sim_indices_for_obs_dates = indexin(obs_dates, sim_dates)

    any(isnothing, sim_indices_for_obs_dates) && error(
        "There are dates in the metadata ($(setdiff(obs_dates, sim_dates))) that are not present in the dates in var",
    )

    return ClimaAnalysis.view_select(
        var,
        by = ClimaAnalysis.Index(),
        time = sim_indices_for_obs_dates,
    )
end

"""
    EnsembleBuilder.is_complete(g_ens_builder::GEnsembleBuilder)

Return `true` if all the entries of the G ensemble matrix is filled out and
`false` otherwise.
"""
function EnsembleBuilder.is_complete(g_ens_builder::GEnsembleBuilder)
    return all(g_ens_builder.completed)
end

"""
    EnsembleBuilder.get_g_ensemble(g_ens_builder::GEnsembleBuilder)

Return the G ensemble matrix from `g_ens_builder`.

This function does not check that the G ensemble matrix is completed. See
[`ClimaCalibrate.EnsembleBuilder.is_complete`](@ref) to check if the G ensemble
matrix is completely filled out.
"""
function EnsembleBuilder.get_g_ensemble(g_ens_builder::GEnsembleBuilder)
    return g_ens_builder.g_ens
end

"""
    ranges_by_short_name(g_ens_builder::GEnsembleBuilder, short_name)

Return a vector of ranges for the G ensemble matrix that correspond with the
short name.
"""
function EnsembleBuilder.ranges_by_short_name(
    g_ens_builder::GEnsembleBuilder,
    short_name,
)
    metadata_by_short_name = g_ens_builder.metadata_by_short_name
    haskey(metadata_by_short_name, short_name) || return UnitRange{Int64}[]
    metadata_infos = metadata_by_short_name[short_name]
    return collect(metadata_info.range for metadata_info in metadata_infos)
end

"""
    metadata_by_short_name(g_ens_builder::GEnsembleBuilder, short_name)

Return a vector of metadata that correspond with `short_name`.
"""
function EnsembleBuilder.metadata_by_short_name(
    g_ens_builder::GEnsembleBuilder,
    short_name,
)
    metadata_by_short_name = g_ens_builder.metadata_by_short_name
    haskey(metadata_by_short_name, short_name) || return Metadata[]
    metadata_infos = metadata_by_short_name[short_name]
    return collect(metadata_info.metadata for metadata_info in metadata_infos)
end

"""
    missing_short_names(g_ens_builder::GEnsembleBuilder, col_idx)

Return a set of the short names of the metadata that are not filled out for the
`col_idx`th column of `g_ens_builder`.
"""
function EnsembleBuilder.missing_short_names(
    g_ens_builder::GEnsembleBuilder,
    col_idx,
)
    completed_col = g_ens_builder.completed[:, col_idx]
    short_names = Set{String}()
    for i in eachindex(completed_col)
        completed_col[i] && continue
        push!(
            short_names,
            ClimaAnalysis.short_name(g_ens_builder.metadata_vec[i].metadata),
        )
    end
    return short_names
end

"""
    Base.show(io::IO, g_ens_builder::GEnsembleBuilder)

Show the index of the metadata, the short name of the metadata, the
corresponding indices for the columns of the G ensemble matrix, and how many
members are completed for the metadata.
"""
function Base.show(io::IO, g_ens_builder::GEnsembleBuilder)
    # Collect all strings first to determine column widths
    headers = (
        "Index",
        "Short name",
        "G ensemble indices",
        "Completed ($(last(size(g_ens_builder.completed))) ensemble members)",
    )
    rows = NTuple{length(headers), String}[]
    for (i, metadata_info) in enumerate(g_ens_builder.metadata_vec)
        short_name = ClimaAnalysis.short_name(metadata_info.metadata)
        range_str = "$(first(metadata_info.range)):$(last(metadata_info.range))"
        nums_completed = sum(g_ens_builder.completed[i, :])
        push!(
            rows,
            (
                string(metadata_info.index),
                short_name,
                range_str,
                string(nums_completed),
            ),
        )
    end

    # Calculate maximum width for each column
    col_widths = collect(
        max(length(headers[i]), maximum(length(row[i]) for row in rows)) for
        i in eachindex(headers)
    )

    # Print header
    for i in eachindex(headers)
        if i < length(headers)
            printstyled(io, rpad(headers[i], col_widths[i] + 2), bold = true)
        else
            printstyled(io, headers[i], bold = true)
        end
    end
    print(io, "\n")
    println(io, "-"^(sum(col_widths) + 2 * length(headers) - 2))

    # Print information about each metadata
    for row in rows
        for i in eachindex(row)
            if i < length(row)
                print(rpad(row[i], col_widths[i] + 2))
            else
                print(row[i])
            end
        end
        print(io, "\n")
    end
end

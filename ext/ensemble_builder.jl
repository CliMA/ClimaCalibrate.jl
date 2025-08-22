import ClimaCalibrate.EnsembleBuilder as EnsembleBuilder
import ClimaCalibrate: g_ens_mat

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
construct the corresponding G ensembe matrix for the current iteration of the
calibration.
"""
struct GEnsembleBuilder{FT <: AbstractFloat, METADATAINFO <: MetadataInfo}
    """G ensemble matrix that is returned by the observation map"""
    g_ens::Matrix{FT}

    """Dictionary which map short name to a vector of metadata associated with
       the short name"""
    metadata_by_short_name::Dict{String, Vector{METADATAINFO}}

    """A bit matrix which keeps track of which entries are filled out in the
    G ensemble matrix. The size of this matrix is the number of metadata for the
    minibatch by the number of ensemble members"""
    completed::BitMatrix
end

"""
    GEnsembleBuilder(ekp::EKP.EnsembleKalmanProcess,
                     ::Type{FT})

Construct a `GEnsembleBuilder` where the element type of the G ensemble matrix
is `FT`.
"""
function EnsembleBuilder.GEnsembleBuilder(
    ekp::EKP.EnsembleKalmanProcess,
    ::Type{FT},
) where {FT <: AbstractFloat}
    pkgversion(ClimaAnalysis) > v"0.5.19" || error("Using GEnsembleBuilder requires a version of ClimaAnalysis above 0.5.19")
    obs_series = EKP.get_observation_series(ekp)
    N = EKP.get_N_iterations(ekp) + 1
    metadatas = ObservationRecipe.get_metadata_for_nth_iteration(obs_series, N)
    eltype(metadatas) <: ClimaAnalysis.Var.Metadata || error(
        "GEnsembleBuilder is only compatible with metadata from ClimaAnalysis",
    )

    # Check all metadata contain a short name
    metadata_short_names =
        collect(ClimaAnalysis.short_name(metadata) for metadata in metadatas)
    # TODO: Add a check or warning for this when making observation; this can be circumvented
    # with combine_observations though
    any(==(""), metadata_short_names) && error(
        "One of the observations does not has a short name; add a short name to the OutputVar before making an observation from it",
    )

    G_ens = g_ens_mat(ekp, FT)
    completed = falses(length(metadatas), EKP.get_N_ens(ekp))

    short_name_to_metadata_map = Dict{String, Vector{MetadataInfo}}()
    obs_series = EKP.get_observation_series(ekp)
    minibatch_indices = _get_minibatch_indices_for_nth_iteration(obs_series, N)

    # Assume G_ens and minibatch_indices are one-indexed
    @assert first(size(G_ens)) == last(last(minibatch_indices))

    for (i, (metadata, range)) in enumerate(zip(metadatas, minibatch_indices))
        short_name = ClimaAnalysis.short_name(metadata)
        metadata_vec =
            get!(short_name_to_metadata_map, short_name, MetadataInfo[])
        push!(metadata_vec, MetadataInfo(i, range, metadata))
    end

    return GEnsembleBuilder(G_ens, short_name_to_metadata_map, completed)
end

"""
    EnsembleBuilder.fill_g_ens_col!(g_ens::GEnsembleBuilder,
                                      col_idx,
                                      ekp,
                                      vars::OutputVar...)

Fill the `col_idx`th of the G ensemble matrix from the `OutputVar`s `vars` and
`ekp`.

The column of the G ensemble matrix does not need to be filled out completely
by this function. As such, you can call this function with the same `col`, but
different `vars` to gradually fill out the column of the G ensemble matrix.

It is assumed that the times or dates of a single `OutputVar` is a superset of
the times or dates of one or more metadata in the minibatch.

This function relies on the short names in the
metadata. This function will not behave correctly if the short names are
mislabled or not present.

Furthermore, this function assumes that all observations are generated using
`ObservationRecipe.Observation` which guarantees the metadata exists and the
correct placement of metadata.
"""
function EnsembleBuilder.fill_g_ens_col!(
    g_ens_builder::GEnsembleBuilder,
    col_idx,
    vars::OutputVar...,
)
    # Check all OutputVars contain a short name
    var_short_names = collect(ClimaAnalysis.short_name(var) for var in vars)
    any(==(""), var_short_names) && error(
        "One of the OutputVars does not has a short name; add a short name to the OutputVar before creating it",
    )

    for (i, var) in enumerate(vars)
        use_var = false
        var_short_name = ClimaAnalysis.short_name(var)
        metadata_info_vec =
            if haskey(g_ens_builder.metadata_by_short_name, var_short_name)
                g_ens_builder.metadata_by_short_name[var_short_name]
            else
                empty(g_ens_builder.metadata_by_short_name |> first |> last)
            end
        for metadata_info in metadata_info_vec
            metadata = metadata_info.metadata
            _is_compatible_for_g_ens(var, metadata) || continue
            match_dates_var = _match_dates(var, metadata)
            flattened_data =
                ClimaAnalysis.flatten(match_dates_var, metadata).data
            g_ens_builder.g_ens[metadata_info.range, col_idx] = flattened_data
            if g_ens_builder.completed[metadata_info.index, col_idx]
                @warn(
                    "This portion of the G ensemble matrix ($(metadata_info.range), $col_idx) is already filled out by another OutputVar. Replacing the contents with the new OutputVar"
                )
            end
            g_ens_builder.completed[metadata_info.index, col_idx] = true
            use_var = true
        end
        use_var || @warn(
            "OutputVar at index $i with the short name $(ClimaAnalysis.short_name(var)) was passed as an input, but did not match with any of the metadata"
        )
    end
    return nothing
end

"""
    _is_compatible_for_g_ens(var::OutputVar, metadata::Metadata)

Return `true` if `var` can be flattened with `metadata` and fill out the
column of the G ensemble matrix corresponding to the `metadata`.
"""
function _is_compatible_for_g_ens(var::OutputVar, metadata::Metadata)
    # TODO: Think about whether this can be splitted into multiple functions or
    # structs for checking stuff since it would be nice to make this extensible
    # instead of dumping every single check in this function
    ClimaAnalysis.short_name(var) == ClimaAnalysis.short_name(metadata) ||
        return false

    # Check dimensions are all available
    Set(ClimaAnalysis.conventional_dim_name.(keys(var.dims))) ==
    Set(ClimaAnalysis.conventional_dim_name.(keys(metadata.dims))) ||
        return false

    # Instead of trying to determine whether dates or times should be used, we
    # try dates first and if it does not work, then use the time dimension
    # instead
    function dates_or_times(var)
        temporal_dim = try
            ClimaAnalysis.dates(var)
        catch
            ClimaAnalysis.times(var)
        end
        return temporal_dim
    end

    # Check if dimensions are the same. For the temporal dimension, the only
    # requirement is that the temporal dimension of var is a superset of the
    # temporal dimension of metadata, because _match_dates will get the correct
    # dates for us
    for var_dim_name in keys(var.dims)
        md_dim_name = ClimaAnalysis.Var.find_corresponding_dim_name_in_var(
            var_dim_name,
            metadata,
        )
        if ClimaAnalysis.conventional_dim_name(var_dim_name) != "time"
            all(isapprox(var.dims[var_dim_name], metadata.dims[md_dim_name])) ||
                return false
        else
            dates_or_times(metadata) ⊆ dates_or_times(var) || return false
        end
    end

    return true
end

"""
    _match_dates(var::OutputVar, metadata::Metadata)

Return a `OutputVar` whose dates are the same as the dates in `metadata`.

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
    common_date_indices = indexin(obs_dates, sim_dates)

    any(isnothing, common_date_indices) && error(
        "There are dates in the metadata ($(setdiff(obs_dates, sim_dates))) that are not present in the dates in var",
    )

    var_time_name = ClimaAnalysis.time_name(var)
    dims = deepcopy(var.dims)
    dims[var_time_name] = dims[var_time_name][common_date_indices]

    time_idx = var.dim2index[var_time_name]
    time_indices = ntuple(
        x -> ifelse(x == time_idx, common_date_indices, Colon()),
        ndims(var.data),
    )
    data = view(var.data, time_indices...)
    return ClimaAnalysis.remake(var, dims = dims, data = data)
end

"""
    EnsembleBuilder.is_complete(g_ens::GEnsembleBuilder)

Return `true` if all the entries of the G ensemble matrix is filled out and
`false` otherwise.
"""
function EnsembleBuilder.is_complete(g_ens_builder::GEnsembleBuilder)
    return all(g_ens_builder.completed)
end

"""
    EnsembleBuilder.get_g_ensemble(g_ens_builder::GEnsembleBuilder)

Get the G ensemble matrix from `g_ens_builder`.

This function does not check that the G ensemble matrix is completed. See
blah to check if the G ensemble matrix is filled out.
"""
function EnsembleBuilder.get_g_ensemble(g_ens_builder::GEnsembleBuilder)
    return g_ens_builder.g_ens
end



"""
    diagnose(g_ens::GEnsembleBuilder,
             var::OutputVar,
             metadata = get_metadata_vec(g_ens, ClimaAnalysis.short_name(var))

Diagnose if `var` is compatible with the `metadata`.
"""
function diagnose(g_ens::GEnsembleBuilder, var::OutputVar) end

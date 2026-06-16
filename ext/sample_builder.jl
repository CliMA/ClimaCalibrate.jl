import ClimaCalibrate.SampleBuilder
import ClimaCalibrate.SampleBuilder:
    build_samples,
    build_samples_by_times,
    num_samples,
    reconstruct_col,
    choose_obs,
    reconstruct_obs,
    get_obs,
    get_obs_metadata,
    get_samples,
    get_metadata

import ClimaAnalysis: OutputVar

"""
    SampleCollection

An object for storing a collection of samples and their associated metadata as
matrices.

The collection of samples is represented as a matrix. Each column of the matrix
of samples represents one sample which is a vertical concatenation of one or
more flattened `ClimaAnalysis.OutputVar`s. Each column of `metadata` holds the
corresponding `ClimaAnalysis.Var.Metadata`, one for each
`ClimaAnalysis.OutputVar`.

For each row of the matrix of the metadata, for dimensions that are not ignored,
it is guaranteed that
1. the short names are the same,
2. the flattened vector size are the same,
3. the units are the same,
4. the dimensions between both are the same,
5. the number of dimensions are the same,
6. the dimension units are the same,
7. the dimension values are the same,
8. the coordinates where the NaNs are dropped are the same.
"""
struct SampleCollection{
    FT <: AbstractFloat,
    METADATA <: ClimaAnalysis.Var.Metadata,
}
    """A matrix of FT values where each column represents a single sample. A
    single sample may represent multiple variables."""
    samples::Matrix{FT}

    """A matrix of ClimaAnalysis.Metadata."""
    metadata::Matrix{METADATA}
end

"""
    build_samples(var::OutputVar; FT = Float32, dims = $FLATTENED_DIMS)

Return a `SampleCollection` with a single sample consisting of `var`.

The matrix of samples has element type `FT` (defaults to `Float32`). `OutputVar`s
are flattened in the order of `dims`.
"""
function SampleBuilder.build_samples(
    var::OutputVar;
    FT = Float32,
    dims = FLATTENED_DIMS,
)
    return build_samples(OutputVar[var]; FT, dims)
end

"""
    build_samples(
        var_sample::Vector;
        FT = Float32,
        dims = $FLATTENED_DIMS,
    )

Return a `SampleCollection` with a single sample consisting of the `OutputVar`s
in `var_sample`.

The matrix of samples has element type `FT` (defaults to `Float32`). `OutputVar`s
are flattened in the order of `dims`.
"""
function SampleBuilder.build_samples(
    var_sample::Vector;
    FT = Float32,
    dims = FLATTENED_DIMS,
)
    return build_samples(reshape(var_sample, :, 1); FT, dims)
end

"""
    build_samples(
        var_samples::Matrix;
        FT = Float32,
        dims = $FLATTENED_DIMS,
        ignore_dims = (),
    )

Return a `SampleCollection` from `var_samples`.

The matrix of samples has element type `FT` (defaults to `Float32`).

The `OutputVar`s are flattened in the order of `dims`. When validating between
`OutputVar`s, checks for dimensions in `ignore_dims` are skipped.

It is the user's responsibility to ensure that ignoring those dimensions is
appropriate. For example, `build_samples_by_times` ignores the time dimension
because each sample is windowed to a different time range.
"""
function SampleBuilder.build_samples(
    var_samples::Matrix;
    FT = Float32,
    dims = FLATTENED_DIMS,
    ignore_dims = (),
)
    iszero(length(var_samples)) &&
        error("There are no OutputVars in var_samples")

    # Create samples matrix
    first_sample = view(var_samples, :, 1)
    flattened_var_sample = ClimaAnalysis.flatten.(first_sample; dims)
    sample_length = sum(ClimaAnalysis.flattened_length.(flattened_var_sample))
    n_samples = size(var_samples, 2)
    samples_mat = zeros(FT, (sample_length, n_samples))

    # Create metadata matrix
    metadata_mat =
        Array{ClimaAnalysis.Var.Metadata, 2}(undef, size(var_samples))

    # Fill out all columns
    for (i, var_sample) in enumerate(eachcol(var_samples))
        index = 1
        for (j, var) in enumerate(var_sample)
            flattened_var = ClimaAnalysis.flatten(var; dims)

            # Pairwise checks against the first column
            i == 1 || _validate(flattened_var, metadata_mat[j, 1], ignore_dims)

            data_size = ClimaAnalysis.flattened_length(flattened_var)
            start_idx = index
            index += data_size
            samples_mat[start_idx:(start_idx + data_size - 1), i] .=
                flattened_var.data
            metadata_mat[j, i] = flattened_var.metadata

        end
    end
    return SampleCollection(samples_mat, metadata_mat)
end

"""
    _validate(flattened_var, metadata, ignore_dims)

Validate the metadata between `flattened_var` and `metadata` are the same.

Dimensions in `ignore_dims` are ignored when validating dimensions.

The checks are that
1. the short names are the same,
2. the flattened vector size are the same,
3. the units are the same,
4. the dimensions between both are the same,
5. the number of dimensions are the same,
6. the dimension units are the same,
7. the dimension values are the same,
8. the coordinates where the NaNs are dropped are the same.
"""
function _validate(flattened_var, metadata, ignore_dims)
    # Check short names
    metadata_short_name = ClimaAnalysis.short_name(metadata)
    flattened_var_short_name = ClimaAnalysis.short_name(flattened_var)
    metadata_short_name == flattened_var_short_name || error(
        "Short names are not the same. Got $metadata_short_name and $flattened_var_short_name",
    )

    # Check length of flattened vector
    metadata_flattened_length = ClimaAnalysis.flattened_length(metadata)
    flattened_var_length = ClimaAnalysis.flattened_length(flattened_var)
    metadata_flattened_length == flattened_var_length || error(
        "Length of flattened OutputVars are not the same. Got $metadata_flattened_length and $flattened_var_length",
    )

    # Check units
    metadata_units = ClimaAnalysis.units(metadata)
    flattened_var_units = ClimaAnalysis.units(flattened_var)
    metadata_units == flattened_var_units || error(
        "Units are not the same. Got $metadata_units and $flattened_var_units",
    )

    # Check type and number of dimensions are the same
    # Before the call to _validate, all OutputVars are flatten in the same
    # order, so `flatten_dim_order` is a way to retrieve the dimension names
    # in an ordered list
    ClimaAnalysis.conventional_dim_name.(
        ClimaAnalysis.flatten_dim_order(metadata)
    ) ==
    ClimaAnalysis.conventional_dim_name.(
        ClimaAnalysis.flatten_dim_order(flattened_var)
    ) || error(
        "Dimensions found are not equal across different samples. See OutputVar with short name $(ClimaAnalysis.short_name(metadata))",
    )

    ignore_dims = ClimaAnalysis.conventional_dim_name.(ignore_dims)
    var_dim_names = ClimaAnalysis.dim_names(flattened_var)
    metadata_dim_names = ClimaAnalysis.dim_names(metadata)
    for var_dim_name in var_dim_names
        ClimaAnalysis.conventional_dim_name(var_dim_name) in ignore_dims &&
            continue

        # Find the corresponding dimension in the metadata (matched by name)
        metadata_dim_name = ClimaAnalysis.find_corresponding_dim_name(
            var_dim_name,
            metadata_dim_names,
        )

        # Check dimension values
        same_dim_length =
            length(flattened_var.dims[var_dim_name]) ==
            length(metadata.dims[metadata_dim_name])
        same_dim_vals =
            same_dim_length && isapprox(
                flattened_var.dims[var_dim_name],
                metadata.dims[metadata_dim_name],
            )
        same_dim_vals || error(
            "Dimension values are not the same for $var_dim_name: Got $(flattened_var.dims[var_dim_name]) and $(metadata.dims[metadata_dim_name])",
        )

        # Check dimension units
        var_dim_units = ClimaAnalysis.dim_units(flattened_var, var_dim_name)
        metadata_dim_units =
            ClimaAnalysis.dim_units(metadata, metadata_dim_name)
        var_dim_units == metadata_dim_units || error(
            "Dimensions units are not the same. Got $var_dim_units and $metadata_dim_units",
        )
    end

    # Check that the coordinates where the NaNs are dropped are the same across
    # samples
    # It is possible for two FlatVars to have the same length, but the NaNs are
    # at different coordinates
    _same_coords_for_dropped_vals(metadata, flattened_var, ignore_dims) ||
        error(
            "Coordinates where values are dropped are not the same. Check that the dropped values (e.g. NaNs) appear at the same coordinates across different samples",
        )

    return nothing
end

"""
    _same_coords_for_dropped_vals()

Return `true` if `metadata` and `flat_var` have the same coordinates where the values are
dropped when flattening.
"""
function _same_coords_for_dropped_vals(
    metadata::ClimaAnalysis.Var.Metadata,
    flattened_var::ClimaAnalysis.Var.FlatVar,
    ignore_dims,
)
    @static if pkgversion(ClimaAnalysis) > v"0.5.22"
        return ClimaAnalysis.arecompatible(
            metadata,
            flattened_var.metadata;
            ignore_dims,
        )
    else
        return metadata.drop_mask == flattened_var.metadata.drop_mask
    end
end

"""
    build_samples_by_times(var::OutputVar, time_ranges; FT = Float32, dims = $FLATTENED_DIMS)

Generate samples from a single `OutputVar` by windowing its time dimension with
`time_ranges` (one sample per time range).

The matrix of samples has element type `FT` (defaults to `Float32`).
"""
function SampleBuilder.build_samples_by_times(
    var::OutputVar,
    time_ranges;
    FT = Float32,
    dims = FLATTENED_DIMS,
)
    return build_samples_by_times(OutputVar[var], time_ranges; FT, dims)
end

"""
    build_samples_by_times(
        vars::Vector,
        time_ranges;
        FT = Float32,
        dims = $FLATTENED_DIMS,
    )

Generate samples from a vector of `OutputVar`s by windowing the times of the
`OutputVar`s in `vars` using `time_ranges`.

Each sample has every `OutputVar` in `vars`, but the times are windowed
according to `time_ranges`. The matrix of samples has element type `FT` (defaults
to `Float32`).
"""
function SampleBuilder.build_samples_by_times(
    vars::Vector,
    time_ranges;
    FT = Float32,
    dims = FLATTENED_DIMS,
)
    # Validate if time range make sense
    isempty(time_ranges) && error("Time ranges are empty")
    for time_range in time_ranges
        length(time_range) == 2 ||
            error("The time range ($time_range) is not of length 2")
        time_left, time_right = time_range
        time_left <= time_right || error(
            "The starting date/time ($time_left) should be before the ending date/time ($time_right)",
        )
    end

    # Check each var has a time dimension and dates are all unique
    for var in vars
        ClimaAnalysis.has_time(var) || error(
            "OutputVar with short name $(ClimaAnalysis.short_name(var)) doesn't have a time dimension",
        )
        var_times = ClimaAnalysis.times(var)
        allunique(var_times) || error(
            "Not all times of OutputVar with short name $(ClimaAnalysis.short_name(var)) are unique",
        )
    end

    # Each sample (column) consisting of multiple OutputVars corresponds to a
    # single time range
    # Instead of checking if the times in time_range exist in var, it is easier
    # to window the OutputVar with the times and see if that fails. Note that it
    # is cumbersome to check if the times in time_range exist in var, because
    # the times can either be floating point times (relative time) or
    # Dates.DateTime (absolute time)
    var_samples = [
        ClimaAnalysis.window(
            var,
            "time",
            left = t_left,
            right = t_right,
            by = ClimaAnalysis.MatchValue(),
        ) for var in vars, (t_left, t_right) in time_ranges
    ]

    # By the construction of var_samples, the only dimension that differs
    # from the samples is the time dimension
    return build_samples(var_samples; FT, dims, ignore_dims = ("time",))
end

"""
    num_samples(sample_collection::SampleCollection)

Return the number of samples in `sample_collection`.
"""
function SampleBuilder.num_samples(sample_collection::SampleCollection)
    return size(sample_collection.samples, 2)
end

"""
    reconstruct_col(sample_collection::SampleCollection, i::Integer)

Reconstruct the `i`th column of `sample_collection` as a vector of `OutputVar`s.
"""
function SampleBuilder.reconstruct_col(
    sample_collection::SampleCollection,
    i::Integer,
)
    total = num_samples(sample_collection)
    1 <= i <= total || error(
        "The number of samples is $total, but the $(i)th sample is requested",
    )
    return _reconstruct_vars(
        view(sample_collection.samples, :, i),
        view(sample_collection.metadata, :, i),
    )
end

"""
    ObservedSampleCollection

An object to store a [`SampleCollection`](@ref) and the observation to be used
in the calibration.
"""
struct ObservedSampleCollection{SC <: SampleCollection}
    """A collection of samples in the form of a matrix and the associated
    metadata."""
    sample_collection::SC

    """The index of the column of the matrix of samples to use as the true
    observation."""
    i::Int
end

"""
    choose_obs(sample_collection::SampleCollection, i::Integer)

Choose the `i`th sample to be the observation for the calibration.
"""
function SampleBuilder.choose_obs(
    sample_collection::SampleCollection,
    i::Integer,
)
    total_samples = num_samples(sample_collection)
    1 <= i <= total_samples || error(
        "The number of samples is $total_samples, but the $(i)th sample is requested to use as the observation",
    )
    return ObservedSampleCollection(sample_collection, i)
end

"""
    reconstruct_obs(observed_sample_collection::ObservedSampleCollection)

Reconstruct the observation as a vector of `OutputVar`s.
"""
function SampleBuilder.reconstruct_obs(
    observed_sample_collection::ObservedSampleCollection,
)
    return reconstruct_col(
        observed_sample_collection.sample_collection,
        observed_sample_collection.i,
    )
end

"""
    get_obs(observed_sample_collection::ObservedSampleCollection)

Return the observation as a vector of values.

This may be a view into the observation stored in `observed_sample_collection`.
"""
function SampleBuilder.get_obs(
    observed_sample_collection::ObservedSampleCollection,
)
    return view(
        observed_sample_collection.sample_collection.samples,
        :,
        observed_sample_collection.i,
    )
end

"""
    get_obs_metadata(observed_sample_collection::ObservedSampleCollection)

Return the metadata associated with the observation as a vector of
`ClimaAnalysis.Var.Metadata`.

This may be a view into the metadata stored in `observed_sample_collection`.
"""
function SampleBuilder.get_obs_metadata(
    observed_sample_collection::ObservedSampleCollection,
)
    return view(
        observed_sample_collection.sample_collection.metadata,
        :,
        observed_sample_collection.i,
    )
end

"""
    get_samples(observed_sample_collection::ObservedSampleCollection)

Return the matrix of samples stored in `observed_sample_collection`.

Mutating this matrix also mutates the matrix in `observed_sample_collection`.
"""
function SampleBuilder.get_samples(
    observed_sample_collection::ObservedSampleCollection,
)
    return observed_sample_collection.sample_collection.samples
end

"""
    get_metadata(observed_sample_collection::ObservedSampleCollection)

Return the matrix of `ClimaAnalysis.Var.Metadata` in
`observed_sample_collection`.

Mutating this matrix also mutates the matrix in `observed_sample_collection`.
"""
function SampleBuilder.get_metadata(
    observed_sample_collection::ObservedSampleCollection,
)
    return observed_sample_collection.sample_collection.metadata
end

function Base.show(io::IO, sc::SampleCollection)
    sample_len, n_samples = size(sc.samples)
    n_vars = size(sc.metadata, 1)
    FT = eltype(sc.samples)

    printstyled(io, "SampleCollection", bold = true)
    print(io, " ($(sample_len)×$(n_samples) matrix of $FT)")
    print(
        io,
        "\n$n_samples sample(s), each $sample_len value(s) from $n_vars variable(s)",
    )

    # All columns share the same variables, so read metadata from column 1. The
    # ranges give the rows of the sample matrix that belong to each variable.
    first_metadata = view(sc.metadata, :, 1)
    ranges = _get_indices_of_metadata(first_metadata)

    headers = ("Short name", "Units", "Indices", "Dimensions")
    rows = NTuple{length(headers), String}[]
    for (md, idx_range) in zip(first_metadata, ranges)
        dim_order = ClimaAnalysis.flatten_dim_order(md)
        dim_str =
            join(("$dim ($(length(md.dims[dim])))" for dim in dim_order), ", ")
        push!(
            rows,
            (
                ClimaAnalysis.short_name(md),
                ClimaAnalysis.units(md),
                "$(first(idx_range)):$(last(idx_range))",
                dim_str,
            ),
        )
    end

    col_widths = collect(
        max(length(headers[i]), maximum(length(row[i]) for row in rows)) for
        i in eachindex(headers)
    )

    print(io, "\n")
    for i in eachindex(headers)
        text =
            i < length(headers) ? rpad(headers[i], col_widths[i] + 2) :
            headers[i]
        printstyled(io, text, bold = true)
    end
    print(io, "\n")
    println(io, "-"^(sum(col_widths) + 2 * length(headers) - 2))
    for (k, row) in enumerate(rows)
        for i in eachindex(row)
            print(
                io,
                i < length(row) ? rpad(row[i], col_widths[i] + 2) : row[i],
            )
        end
        k < length(rows) && print(io, "\n")
    end
    return nothing
end

function Base.show(io::IO, osc::ObservedSampleCollection)
    show(io, osc.sample_collection)
    print(io, "\n")
    printstyled(io, "Observation: ", bold = true)
    print(io, "sample $(osc.i) of $(num_samples(osc.sample_collection))")
    return nothing
end

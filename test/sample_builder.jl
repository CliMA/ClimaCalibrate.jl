using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.ObservationRecipe
import EnsembleKalmanProcesses
import OrderedCollections: OrderedDict
import Statistics: mean
import LinearAlgebra: Diagonal, I
import Statistics
import NaNStatistics: nanvar, nanmean

import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions

import ClimaCalibrate.SampleBuilder

# Since functions defined in ext are not exported, we need to access them
# like this
ext = Base.get_extension(ClimaCalibrate, :ClimaCalibrateClimaAnalysisExt)

import ClimaAnalysis.Template:
    TemplateVar,
    make_template_var,
    add_attribs,
    add_dim,
    add_time_dim,
    add_lon_dim,
    add_lat_dim,
    add_data,
    ones_data,
    zeros_data,
    one_to_n_data,
    initialize

"""
    equal_samples(sample_collection1, sample_collection2)

Return `true` if the contents of `sample_collection1` and `sample_collection2` are the same.
"""
function equal_samples(sample_collection1, sample_collection2)
    is_equal = sample_collection1.samples == sample_collection2.samples
    metadata_arr1 = sample_collection1.metadata
    metadata_arr2 = sample_collection2.metadata
    is_equal &= size(metadata_arr1) == size(metadata_arr2)
    !is_equal && return is_equal

    for (metadata1, metadata2) in zip(metadata_arr1, metadata_arr2)
        is_equal &= equal_metadata(metadata1, metadata2)
    end
    return is_equal
end

"""
    equal_metadata(
        metadata1::ClimaAnalysis.Var.Metadata,
        metadata2::ClimaAnalysis.Var.Metadata,
    )

Return `true` if the contents of `metadata1` and `metadata2` are the same.

Note that we need to define our own equality function since ClimaAnalysis
v0.5.22 does not provide its own equality function for
`ClimaAnalysis.Var.Metadata`.
"""
function equal_metadata(
    metadata1::ClimaAnalysis.Var.Metadata,
    metadata2::ClimaAnalysis.Var.Metadata,
)
    names = fieldnames(typeof(metadata1))
    is_equal = true
    for name in names
        field1 = getproperty(metadata1, name)
        field2 = getproperty(metadata2, name)
        is_equal &= field1 == field2
    end
    return is_equal
end

"""
    to_col_vec(v::Vector)

Return `v` as a matrix representing a column vector.
"""
function to_col_vec(v::Vector)
    return reshape(v, :, 1)
end

"""
    to_col_vec(v::Vector)

Return `v` as a matrix representing a row vector.
"""
function to_row_vec(v::Vector)
    return reshape(v, 1, :)
end

@testset "Construct single sample" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2008),
            [Dates.DateTime(2008, i) for i in 1:12],
        )
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(short_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize

    sample_collection1 = SampleBuilder.build_samples(var)
    sample_collection2 = SampleBuilder.build_samples([var])
    sample_collection3 = SampleBuilder.build_samples(to_col_vec([var]))

    # All samples formed should be exactly the same
    @test equal_samples(sample_collection1, sample_collection2)
    @test equal_samples(sample_collection2, sample_collection3)

    # Check only one of the samples
    flat_var = ClimaAnalysis.flatten(var; dims = ext.FLATTENED_DIMS)
    @test eltype(sample_collection1.samples) == Float32
    @test sample_collection1.samples == hcat(flat_var.data)
    @test equal_metadata(sample_collection1.metadata[1], flat_var.metadata)

    # Different order of dimensions
    dims = ("lat", "lon", "time")
    sample_collection = SampleBuilder.build_samples(var; dims)
    flat_var = ClimaAnalysis.flatten(var; dims)
    @test ClimaAnalysis.flatten_dim_order(sample_collection.metadata[1]) ==
          ("lat", "lon", "time")
    @test sample_collection.samples == hcat(flat_var.data)

    # Use Float64
    sample_collection = SampleBuilder.build_samples(var; FT = Float64)
    @test eltype(sample_collection.samples) == Float64

    # Error handling
    @test_throws r"There are no OutputVars in var_samples" SampleBuilder.build_samples([],)
end

@testset "Multiple samples" begin
    # Make a vector of OutputVars of length n that differ by one month
    # Also, the data of each var is equal to the data of previous OutputVar plus
    # one
    function make_vars_vector(var::ClimaAnalysis.OutputVar, n)
        vars = [var]
        for _ in 1:(n - 1)
            push!(
                vars,
                ClimaAnalysis.transform_dates(
                    vars[end],
                    date -> date + Dates.Month(1),
                ) + 1,
            )
        end
        return vars
    end

    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2008),
            [Dates.DateTime(2008, i) for i in 1:12],
        )
    pr_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(long_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    pr_vars = make_vars_vector(pr_var, 3)

    rsut_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(long_name = "rsut", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    rsut_vars = make_vars_vector(rsut_var, 3)

    rsus_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_attribs(long_name = "rsus", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    rsus_vars = make_vars_vector(rsus_var, 3)

    # Test with multiple samples with a a single OutputVar each
    vars = [pr_var, rsut_var, rsus_var]
    num_samples = 5
    sample_collection =
        SampleBuilder.build_samples(repeat(vars, 1, num_samples); FT = Float32)

    flat_pr = ClimaAnalysis.flatten(pr_var; dims = ext.FLATTENED_DIMS)
    flat_rsut = ClimaAnalysis.flatten(rsut_var; dims = ext.FLATTENED_DIMS)
    flat_rsus = ClimaAnalysis.flatten(rsus_var; dims = ext.FLATTENED_DIMS)
    total_length =
        sum(ClimaAnalysis.flattened_length.([flat_pr, flat_rsut, flat_rsus]))

    # Check type, shape, and observation index
    @test eltype(sample_collection.samples) == Float32
    @test size(sample_collection.samples) == (total_length, num_samples)
    @test size(sample_collection.metadata) == (length(vars), num_samples)

    expected_col = Float32[flat_pr.data; flat_rsut.data; flat_rsus.data]
    for i in 1:num_samples
        # Check data is the concatenation of each flattened var
        @test sample_collection.samples[:, i] == expected_col

        # Check metadata matches each flattened var
        @test equal_metadata(sample_collection.metadata[1, i], flat_pr.metadata)
        @test equal_metadata(
            sample_collection.metadata[2, i],
            flat_rsut.metadata,
        )
        @test equal_metadata(
            sample_collection.metadata[3, i],
            flat_rsus.metadata,
        )
    end

    # Test with permutation of dimensions
    nan_pr_var = deepcopy(pr_var)
    nan_pr_var.data[begin] = NaN
    sample_collection = SampleBuilder.build_samples(
        to_row_vec([
            nan_pr_var,
            permutedims(nan_pr_var, ("lon", "lat", "time")),
        ]);
        FT = Float32,
    )
    # Metadata is different between the different samples, so we won't check it
    @test sample_collection.samples[:, 1] == sample_collection.samples[:, 2]

    # Test with different dimension names
    diff_dim_pr_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("longitude", lon, units = "degrees") |>
        add_dim("latitude", lat, units = "degrees") |>
        add_attribs(long_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    sample_collection = SampleBuilder.build_samples(
        to_row_vec([
            pr_var,
            permutedims(diff_dim_pr_var, ("lon", "lat", "time")),
        ]);
        FT = Float32,
    )
    @test sample_collection.samples[:, 1] == sample_collection.samples[:, 2]

    # Test with ignore_dims
    samples_mat = stack(
        [pr_var, rsut_var, rsus_var] for
        (pr_var, rsut_var, rsus_var) in zip(pr_vars, rsut_vars, rsus_vars)
    )
    sample_collection = SampleBuilder.build_samples(
        samples_mat;
        FT = Float32,
        ignore_dims = ("time",),
    )

    num_samples_vec = size(samples_mat, 2)
    total_length = sum(
        ClimaAnalysis.flattened_length.(
            ClimaAnalysis.flatten.(first(eachcol(samples_mat)))
        ),
    )

    # Check type, shape, and observation index
    @test eltype(sample_collection.samples) == Float32
    @test size(sample_collection.samples) == (total_length, num_samples_vec)
    @test size(sample_collection.metadata) == (length(vars), num_samples_vec)

    # Each column should contain data from its corresponding sample
    for (i, (pr, rsut, rsus)) in enumerate(zip(pr_vars, rsut_vars, rsus_vars))
        flat_pr_i = ClimaAnalysis.flatten(pr; dims = ext.FLATTENED_DIMS)
        flat_rsut_i = ClimaAnalysis.flatten(rsut; dims = ext.FLATTENED_DIMS)
        flat_rsus_i = ClimaAnalysis.flatten(rsus; dims = ext.FLATTENED_DIMS)
        expected_col_i =
            Float32[flat_pr_i.data; flat_rsut_i.data; flat_rsus_i.data]
        @test sample_collection.samples[:, i] == expected_col_i
    end

    # Columns differ since data is incremented across samples
    @test sample_collection.samples[:, 1] != sample_collection.samples[:, 2]

    # Should throw error when the time dimension is not included in ignore_dims
    @test_throws ErrorException SampleBuilder.build_samples(
        samples_mat;
        FT = Float32,
    )

    # Error handling
    # Empty vectors
    @test_throws r"There are no OutputVars in var_samples" SampleBuilder.build_samples(
        fill(ClimaAnalysis.OutputVar, 0, 2),
    )

    # Short names are not the same
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-90.0, -30.0, 30.0, 90.0]
    var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            short_name = "pr",
            start_date = "2008-1-1",
            units = "mm/day",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    diff_short_name_var = deepcopy(var)
    ClimaAnalysis.set_short_name!(diff_short_name_var, "rsus")
    @test_throws r"Short names are not the same" SampleBuilder.build_samples(
        to_row_vec([var, diff_short_name_var]),
    )

    # Flattened vector size are not the same
    diff_length_var = ClimaAnalysis.average_lat(var)
    @test_throws r"Length of flattened OutputVars are not the same" SampleBuilder.build_samples(
        to_row_vec([var, diff_length_var]),
    )

    # NaNs are not the same across physical space
    nan_var1 = deepcopy(var)
    nan_var2 = deepcopy(var)
    nan_var1.data[begin] = NaN
    nan_var2.data[end] = NaN
    @test_throws r"Coordinates where values are dropped are not the same" SampleBuilder.build_samples(
        to_row_vec([nan_var1, nan_var2]),
    )

    # Units are not the same
    diff_units_var = ClimaAnalysis.set_units(var, "K")
    @test_throws r"Units are not the same" SampleBuilder.build_samples(
        to_row_vec([var, diff_units_var]),
    )

    # Number of dimensions are not the same
    diff_ndims_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lat", lat, units = "degrees") |>
        add_dim("lon", [0.0], units = "degrees") |>
        add_attribs(
            short_name = "pr",
            start_date = "2008-1-1",
            units = "mm/day",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    @test_throws r"Dimensions found are not equal across different samples" SampleBuilder.build_samples(
        to_row_vec([var, diff_ndims_var]),
    )

    # Type of the dimensions are not the same
    diff_dim_type_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_attribs(
            short_name = "pr",
            start_date = "2008-1-1",
            units = "mm/day",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    @test_throws r"Dimensions found are not equal across different samples" SampleBuilder.build_samples(
        to_row_vec([var, diff_dim_type_var]),
    )

    # Dimension units are not the same
    diff_dim_units_var = deepcopy(var)
    ClimaAnalysis.set_dim_units!(diff_dim_units_var, "latitude", "deg")
    @test_throws r"Dimensions units are not the same" SampleBuilder.build_samples(
        to_row_vec([var, diff_dim_units_var]),
    )

    # Dimension values are not the same
    diff_dim_vals_var = deepcopy(var)
    ClimaAnalysis.latitudes(diff_dim_vals_var) .+= 100
    @test_throws r"Dimension values are not the same" SampleBuilder.build_samples(
        to_row_vec([var, diff_dim_vals_var]),
    )
end

@testset "Getters" begin
    function make_var(short_name, value)
        lat = [-90.0, -30.0, 30.0, 90.0]
        lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
        time =
            ClimaAnalysis.Utils.date_to_time.(
                Dates.DateTime(2008),
                [Dates.DateTime(2008, i) for i in 1:12],
            )
        var =
            TemplateVar() |>
            add_dim("time", time, units = "s") |>
            add_dim("lon", lon, units = "degrees") |>
            add_dim("lat", lat, units = "degrees") |>
            add_attribs(short_name = short_name, start_date = "2008-1-1") |>
            one_to_n_data(collected = true) |>
            initialize
        var.data .= value
        return var
    end

    pr_var1 = make_var("pr", 1.0)
    pr_var2 = make_var("pr", 2.0)

    rsut_var1 = make_var("rsut", 3.0)
    rsut_var2 = make_var("rsut", 4.0)

    sample_collection =
        SampleBuilder.build_samples([pr_var1 pr_var2; rsut_var1 rsut_var2])

    # Get the number of samples
    @test SampleBuilder.num_samples(sample_collection) == 2

    osc = SampleBuilder.choose_obs(sample_collection, 2)

    # Get the matrix of samples and metadata
    @test SampleBuilder.get_samples(osc) === sample_collection.samples
    @test SampleBuilder.get_metadata(osc) === sample_collection.metadata

    # Get the observation (second column of the matrix of samples)
    @test SampleBuilder.get_obs(osc) == sample_collection.samples[:, 2]

    # Get the metadata of the observation
    obs_metadata = SampleBuilder.get_obs_metadata(osc)
    @test length(obs_metadata) == 2
    for (md, full_md) in zip(obs_metadata, sample_collection.metadata[:, 2])
        @test equal_metadata(md, full_md)
    end

    # The observation column should correspond to the second sample
    flat_pr2 = ClimaAnalysis.flatten(pr_var2; dims = ext.FLATTENED_DIMS)
    flat_rsut2 = ClimaAnalysis.flatten(rsut_var2; dims = ext.FLATTENED_DIMS)
    @test SampleBuilder.get_obs(osc) == Float32[flat_pr2.data; flat_rsut2.data]
    @test equal_metadata(obs_metadata[1], flat_pr2.metadata)
    @test equal_metadata(obs_metadata[2], flat_rsut2.metadata)

    # Error handling
    @test_throws ErrorException SampleBuilder.choose_obs(sample_collection, 0)
    @test_throws ErrorException SampleBuilder.choose_obs(sample_collection, 3)
end

@testset "Samples by windowing using time ranges" begin
    lat = [-90.0, -30.0, 30.0, 90.0]

    # Test with absolute time (Dates.DateTime)
    dates =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2008),
            [Dates.DateTime(2008, i) for i in 1:12],
        )
    pr_var_abs =
        TemplateVar() |>
        add_dim("time", dates, units = "s") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(long_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize

    rsut_var_abs =
        TemplateVar() |>
        add_dim("time", dates, units = "s") |>
        add_attribs(long_name = "rsut", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize

    time_ranges = [
        (Dates.DateTime(2008, 1), Dates.DateTime(2008, 3)),
        (Dates.DateTime(2008, 4), Dates.DateTime(2008, 6)),
        (Dates.DateTime(2008, 7), Dates.DateTime(2008, 9)),
    ]
    sample_collection = SampleBuilder.build_samples_by_times(
        [pr_var_abs, rsut_var_abs],
        time_ranges,
    )

    @test size(sample_collection.metadata) == (2, length(time_ranges))

    # Check each column of the matrix of samples is what we expect
    for (i, (t_left, t_right)) in enumerate(time_ranges)
        windowed_pr = ClimaAnalysis.window(
            pr_var_abs,
            "time",
            left = t_left,
            right = t_right,
            by = ClimaAnalysis.MatchValue(),
        )
        windowed_rsut = ClimaAnalysis.window(
            rsut_var_abs,
            "time",
            left = t_left,
            right = t_right,
            by = ClimaAnalysis.MatchValue(),
        )
        flat_pr = ClimaAnalysis.flatten(windowed_pr)
        flat_rsut = ClimaAnalysis.flatten(windowed_rsut)
        expected_col = Float32[flat_pr.data; flat_rsut.data]
        @test sample_collection.samples[:, i] ≈ expected_col
    end

    # Test with relative time (Float64)
    times = Float64.(0:11)
    pr_var_rel =
        TemplateVar() |>
        add_dim("time", times, units = "s") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(long_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize

    time_ranges = [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0)]
    sample_collection =
        SampleBuilder.build_samples_by_times([pr_var_rel], time_ranges)

    @test size(sample_collection.metadata) == (1, length(time_ranges))

    # Check each column of the matrix of samples is what we expect
    for (i, (t_left, t_right)) in enumerate(time_ranges)
        windowed = ClimaAnalysis.window(
            pr_var_rel,
            "time",
            left = t_left,
            right = t_right,
            by = ClimaAnalysis.MatchValue(),
        )
        flat_windowed = ClimaAnalysis.flatten(windowed)
        @test sample_collection.samples[:, i] ≈ Float32.(flat_windowed.data)
    end

    # Test single OutputVar constructor
    single_sample_collection =
        SampleBuilder.build_samples_by_times(pr_var_rel, time_ranges)
    @test equal_samples(single_sample_collection, sample_collection)

    # Error handling
    # Var without a time dimension
    no_time_var =
        TemplateVar() |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(long_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    @test_throws r"doesn't have a time dimension" SampleBuilder.build_samples_by_times(
        [no_time_var],
        time_ranges,
    )

    # Non-unique times
    dup_time_var =
        TemplateVar() |>
        add_dim("time", [0.0, 0.0, 1.0, 2.0], units = "s") |>
        add_attribs(long_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize
    @test_throws r"Not all times" SampleBuilder.build_samples_by_times(
        [dup_time_var],
        [(0.0, 1.0)],
    )

    # Time range not of length 2
    @test_throws r"is not of length 2" SampleBuilder.build_samples_by_times(
        [pr_var_rel],
        [(0.0, 1.0, 2.0)],
    )

    # Left endpoint greater than right endpoint
    @test_throws r"starting date/time" SampleBuilder.build_samples_by_times(
        [pr_var_rel],
        [(5.0, 1.0)],
    )

    # Times do not exist
    @test_throws ErrorException SampleBuilder.build_samples_by_times(
        [pr_var_rel],
        [(0.5, 1.5)],
    )

    # Empty time ranges
    @test_throws r"Time ranges are empty" SampleBuilder.build_samples_by_times(
        [pr_var_rel],
        [],
    )

    # Uneven lengths
    @test_throws ErrorException SampleBuilder.build_samples_by_times(
        [pr_var_rel],
        [(1.0, 2.0), (3.0, 5.0)],
    )
end

@testset "Reconstruct" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2008),
            [Dates.DateTime(2008, i) for i in 1:12],
        )
    pr_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(long_name = "pr", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize

    rsut_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(long_name = "rsut", start_date = "2008-1-1") |>
        one_to_n_data(collected = true) |>
        initialize

    vars1 = [pr_var, rsut_var]
    vars2 = [pr_var + 1, rsut_var + 1]
    sample_collection = SampleBuilder.build_samples(hcat(vars1, vars2))
    osc = SampleBuilder.choose_obs(sample_collection, 1)

    reconstructed_col1 = SampleBuilder.reconstruct_col(sample_collection, 1)
    reconstructed_col2 = SampleBuilder.reconstruct_col(sample_collection, 2)
    reconstructed_obs = SampleBuilder.reconstruct_obs(osc)

    for (reconstructed_col, vars) in (
        (reconstructed_col1, vars1),
        (reconstructed_col2, vars2),
        (reconstructed_obs, vars1),
    )
        @test length(reconstructed_col) == length(vars)
        for (reconstructed, var) in zip(reconstructed_col, vars)
            @test reconstructed.attributes == var.attributes
            @test reconstructed.dim_attributes == var.dim_attributes
            @test reconstructed.data == Float32.(var.data)
        end
    end

    # Reconstruct with NaNs
    nan_pr_var = deepcopy(pr_var)
    nan_pr_var.data[1:3] .= NaN
    nan_sample_collection = SampleBuilder.build_samples([nan_pr_var])
    reconstructed_nan_pr_var =
        first(SampleBuilder.reconstruct_col(nan_sample_collection, 1))
    @test nan_pr_var.attributes == reconstructed_nan_pr_var.attributes
    @test nan_pr_var.dims == reconstructed_nan_pr_var.dims
    @test nan_pr_var.dim_attributes == reconstructed_nan_pr_var.dim_attributes
    @test isequal(nan_pr_var.data, reconstructed_nan_pr_var.data)

    # Error handling
    @test_throws ErrorException SampleBuilder.reconstruct_col(
        sample_collection,
        0,
    )
    @test_throws ErrorException SampleBuilder.reconstruct_col(
        sample_collection,
        3,
    )
end

@testset "Show" begin
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time =
        ClimaAnalysis.Utils.date_to_time.(
            Dates.DateTime(2008),
            [Dates.DateTime(2008, i) for i in 1:12],
        )
    pr_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            short_name = "pr",
            start_date = "2008-1-1",
            units = "mm/day",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    rsut_var =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            short_name = "rsut",
            start_date = "2008-1-1",
            units = "W m-2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize

    # pr is time(12) × lon(5) × lat(4) = 240 values, rsut is time(12) × lat(4) =
    # 48 values, so each sample has 288 values and the matrix is 288 × 2
    sample_collection =
        SampleBuilder.build_samples([pr_var pr_var; rsut_var rsut_var])
    str = sprint(show, sample_collection)

    # Header line with matrix size and element type
    @test occursin("SampleCollection", str)
    @test occursin("288×2 matrix of Float32", str)

    # Summary line
    @test occursin("2 sample(s), each 288 value(s) from 2 variable(s)", str)

    # Table headers
    @test occursin("Short name", str)
    @test occursin("Units", str)
    @test occursin("Indices", str)
    @test occursin("Dimensions", str)

    # Per-variable rows: short name, units, and index ranges
    @test occursin("pr", str)
    @test occursin("rsut", str)
    @test occursin("mm/day", str)
    @test occursin("W m-2", str)
    @test occursin("1:240", str)
    @test occursin("241:288", str)

    # Dimension order with grid sizes
    @test occursin("time (12)", str)
    @test occursin("lon (5)", str)
    @test occursin("lat (4)", str)

    # ObservedSampleCollection repeats the collection then names the observation
    osc = SampleBuilder.choose_obs(sample_collection, 2)
    osc_str = sprint(show, osc)
    @test occursin("288×2 matrix of Float32", osc_str)
    @test occursin("Observation: sample 2 of 2", osc_str)
end

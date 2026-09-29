using Test
import Dates
import ClimaAnalysis
import ClimaCalibrate
import ClimaCalibrate.SampleBuilder
import ClimaCalibrate.SampleBuilder:
    AbstractTransform, LatitudeWeighting, PerCollectionWeighting

import ClimaAnalysis.Template:
    TemplateVar, add_dim, add_attribs, one_to_n_data, initialize

# Since functions defined in ext are not exported, we need to access them like
# this
ext = Base.get_extension(ClimaCalibrate, :ClimaCalibrateClimaAnalysisExt)

@testset "PerCollectionWeighting transform" begin
    var =
        TemplateVar() |>
        add_dim("lat", [-90.0, 0.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "hi") |>
        one_to_n_data(collected = true) |>
        initialize
    sample_collection = SampleBuilder.build_samples([var var]; FT = Float64)

    # Check apply_transform
    transform = PerCollectionWeighting(3.0)
    weighted_collection =
        SampleBuilder.apply_transform(sample_collection |> transform)
    @test SampleBuilder.get_samples(weighted_collection) ==
          3.0 .* SampleBuilder.get_samples(sample_collection)

    # Check apply_transform!
    deepcopy_collection = deepcopy(sample_collection)
    returned_collection =
        SampleBuilder.apply_transform!(deepcopy_collection, transform)
    @test SampleBuilder.get_samples(returned_collection) ==
          SampleBuilder.get_samples(weighted_collection)
    @test SampleBuilder.get_samples(deepcopy_collection) ==
          SampleBuilder.get_samples(weighted_collection)
end

@testset "PerVariableWeighting transform" begin
    pr_var =
        TemplateVar() |>
        add_dim("lat", [-90.0, 0.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "pr") |>
        one_to_n_data(collected = true) |>
        initialize
    tas_var =
        TemplateVar() |>
        add_dim("lon", [-90.0, 0.0, 30.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "tas") |>
        one_to_n_data(collected = true) |>
        initialize

    sample_collection = SampleBuilder.build_samples(
        [pr_var pr_var; tas_var tas_var];
        FT = Float64,
    )

    # Test both constructors (vector and dict)
    vec_transform = SampleBuilder.PerVariableWeighting([2.0, 3.0])
    by_transform =
        SampleBuilder.PerVariableWeighting(Dict("pr" => 2.0, "tas" => 3.0))
    custom_by_transform = SampleBuilder.PerVariableWeighting(
        Dict(true => 20.0, false => 30.0);
        by = md -> ClimaAnalysis.has_latitude(md),
    )

    # Check each transform return the same result
    weighted_collection1 =
        SampleBuilder.apply_transform(sample_collection |> vec_transform)
    weighted_collection2 =
        SampleBuilder.apply_transform(sample_collection |> by_transform)
    @test SampleBuilder.get_samples(weighted_collection1) ==
          SampleBuilder.get_samples(weighted_collection2)

    weighted_collection3 =
        SampleBuilder.apply_transform(sample_collection |> custom_by_transform)
    samples = SampleBuilder.get_samples(sample_collection)
    @test SampleBuilder.get_samples(weighted_collection3) ==
          [20.0 .* samples[1:3, :]; 30.0 .* samples[4:7, :]]

    # Check apply_transform!
    for (transform, weighted_collection) in (
        (vec_transform, weighted_collection1),
        (by_transform, weighted_collection2),
        (custom_by_transform, weighted_collection3),
    )
        deepcopy_collection = deepcopy(sample_collection)
        returned_collection =
            SampleBuilder.apply_transform!(deepcopy_collection, transform)
        @test SampleBuilder.get_samples(returned_collection) ==
              SampleBuilder.get_samples(weighted_collection)
        @test SampleBuilder.get_samples(deepcopy_collection) ==
              SampleBuilder.get_samples(weighted_collection)
    end

    # Compare against PerCollectionWeighting
    per_var_weighted_collection = SampleBuilder.apply_transform(
        sample_collection |> SampleBuilder.PerVariableWeighting([3.0, 3.0]),
    )
    per_collection_weighted_collection = SampleBuilder.apply_transform(
        sample_collection |> PerCollectionWeighting(3.0),
    )
    @test SampleBuilder.get_samples(per_var_weighted_collection) ==
          SampleBuilder.get_samples(per_collection_weighted_collection)

    # Error handling
    # Cannot use a by function with a vector
    @test_throws ErrorException SampleBuilder.PerVariableWeighting(
        [2.0, 3.0],
        ClimaAnalysis.short_name,
    )
    # Cannot pass nothing for the by keyword argument when using a dictionary
    @test_throws ErrorException SampleBuilder.PerVariableWeighting(
        Dict("pr" => 2.0, "tas" => 3.0);
        by = nothing,
    )

    # The samples are not mutated when an error is thrown. The missing keys are
    # for the second variable, so weighting partway would change the first.
    for transform in (
        # A short name is not provided
        SampleBuilder.PerVariableWeighting(Dict("pr" => 2.0)),
        # A key from a custom `by` is not provided
        SampleBuilder.PerVariableWeighting(
            Dict(true => 20.0);
            by = md -> ClimaAnalysis.has_latitude(md),
        ),
        # The number of weights is not the same as the number of variables
        SampleBuilder.PerVariableWeighting([2.0]),
        SampleBuilder.PerVariableWeighting([2.0, 3.0, 4.0]),
    )
        deepcopy_collection = deepcopy(sample_collection)
        @test_throws ErrorException SampleBuilder.apply_transform!(
            deepcopy_collection,
            transform,
        )
        @test SampleBuilder.get_samples(deepcopy_collection) == samples
    end

    # Every missing key is in the error message
    @test_throws "[\"pr\", \"tas\"]" SampleBuilder.apply_transform!(
        deepcopy(sample_collection),
        SampleBuilder.PerVariableWeighting(Dict("ta" => 2.0)),
    )
end

@testset "LatitudeWeighting transform" begin
    pr_var =
        TemplateVar() |>
        add_dim("lat", [-90.0, 0.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "pr") |>
        one_to_n_data(collected = true) |>
        initialize
    tas_var =
        TemplateVar() |>
        add_dim("lon", [-90.0, 0.0, 30.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "tas") |>
        one_to_n_data(collected = true) |>
        initialize
    ta_var =
        TemplateVar() |>
        add_dim("lat", [0.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "ta") |>
        one_to_n_data(collected = true) |>
        initialize

    # The rows of the samples are pr (1:3), tas (4:7), and ta (8:9)
    sample_collection = SampleBuilder.build_samples(
        [pr_var pr_var; tas_var tas_var; ta_var ta_var];
        FT = Float64,
    )
    samples = SampleBuilder.get_samples(sample_collection)

    # With min_cosd_lat = 0.25, the weights at the latitudes 0 and ±90 degrees
    # are sqrt(1 / 1) = 1 and sqrt(1 / 0.25) = 2 respectively
    default_transform = LatitudeWeighting(min_cosd_lat = 0.25)
    only_pr_transform = LatitudeWeighting(["pr"]; min_cosd_lat = 0.25)
    custom_by_transform = LatitudeWeighting(
        [true];
        by = ClimaAnalysis.has_latitude,
        min_cosd_lat = 0.25,
    )

    # Every variable with a latitude is weighted
    weighted_collection1 =
        SampleBuilder.apply_transform(sample_collection |> default_transform)
    @test SampleBuilder.get_samples(weighted_collection1) == [
        [2.0, 1.0, 2.0] .* samples[1:3, :];
        samples[4:7, :];
        [1.0, 2.0] .* samples[8:9, :]
    ]

    # Only the selected variable is weighted
    weighted_collection2 =
        SampleBuilder.apply_transform(sample_collection |> only_pr_transform)
    @test SampleBuilder.get_samples(weighted_collection2) ==
          [[2.0, 1.0, 2.0] .* samples[1:3, :]; samples[4:9, :]]

    # Selecting every variable with a latitude is the same as the default
    weighted_collection3 =
        SampleBuilder.apply_transform(sample_collection |> custom_by_transform)
    @test SampleBuilder.get_samples(weighted_collection3) ==
          SampleBuilder.get_samples(weighted_collection1)

    # The selected variables can also be a tuple or a set
    for selected in (("pr",), Set(["pr"]))
        weighted_collection = SampleBuilder.apply_transform(
            sample_collection |>
            LatitudeWeighting(selected; min_cosd_lat = 0.25),
        )
        @test SampleBuilder.get_samples(weighted_collection) ==
              SampleBuilder.get_samples(weighted_collection2)
    end

    # Check apply_transform!
    for (transform, weighted_collection) in (
        (default_transform, weighted_collection1),
        (only_pr_transform, weighted_collection2),
        (custom_by_transform, weighted_collection3),
    )
        deepcopy_collection = deepcopy(sample_collection)
        returned_collection =
            SampleBuilder.apply_transform!(deepcopy_collection, transform)
        @test SampleBuilder.get_samples(returned_collection) ==
              SampleBuilder.get_samples(weighted_collection)
        @test SampleBuilder.get_samples(deepcopy_collection) ==
              SampleBuilder.get_samples(weighted_collection)
    end

    # The weights are broadcast over the other dimensions of a variable
    lat = [-90.0, -30.0, 30.0, 90.0]
    lon = [-60.0, -30.0, 0.0, 30.0, 60.0]
    time = ClimaAnalysis.Utils.date_to_time.(
        Dates.DateTime(2007, 12),
        [Dates.DateTime(i, 12, 1) for i in 2007:2009],
    )
    var3d =
        TemplateVar() |>
        add_dim("time", time, units = "s") |>
        add_dim("lon", lon, units = "degrees") |>
        add_dim("lat", lat, units = "degrees") |>
        add_attribs(
            short_name = "hi",
            long_name = "hello",
            start_date = "2007-12-1",
            blah = "blah2",
        ) |>
        one_to_n_data(collected = true) |>
        initialize
    sample_collection3d = SampleBuilder.build_samples([var3d]; FT = Float64)
    weighted_collection3d = SampleBuilder.apply_transform(
        sample_collection3d,
        LatitudeWeighting(min_cosd_lat = 0.15),
    )
    lat_weights_per_column = sqrt.(
        ClimaAnalysis.flatten(ext._lat_weights_var(var3d, min_cosd_lat = 0.15)).data,
    )
    @test isequal(
        SampleBuilder.get_samples(weighted_collection3d),
        SampleBuilder.get_samples(sample_collection3d) .*
        lat_weights_per_column,
    )

    # Error handling
    # min_cosd_lat must be positive
    @test_throws ErrorException LatitudeWeighting(min_cosd_lat = 0.0)
    @test_throws ErrorException LatitudeWeighting(["pr"]; min_cosd_lat = -0.1)
    # The by keyword argument cannot be nothing when an iterable is passed
    @test_throws ErrorException LatitudeWeighting(["pr"]; by = nothing)

    pr_other_lats_var =
        TemplateVar() |>
        add_dim("lat", [-60.0, 0.0, 60.0], units = "degrees") |>
        add_attribs(short_name = "pr") |>
        one_to_n_data(collected = true) |>
        initialize
    pr_radians_var =
        TemplateVar() |>
        add_dim("lat", [-1.0, 0.0, 1.0], units = "radians") |>
        add_attribs(short_name = "pr") |>
        one_to_n_data(collected = true) |>
        initialize

    transforms_to_check = [
        # No variable has a latitude dimension
        (
            default_transform,
            SampleBuilder.build_samples([tas_var tas_var]; FT = Float64),
            "None of the variables have a latitude dimension",
        ),
        # No variable is selected
        (
            LatitudeWeighting(["rsut"]),
            sample_collection,
            "None of the variables are selected",
        ),
        # A selected variable does not have a latitude dimension
        (
            LatitudeWeighting(["pr", "tas"]),
            sample_collection,
            "is selected for latitude weighting, but it does not have a latitude dimension",
        ),
        # The latitudes of a variable are not the same across samples
        (
            default_transform,
            SampleBuilder.build_samples(
                [ta_var ta_var; pr_var pr_other_lats_var];
                FT = Float64,
                ignore_dims = ("lat",),
            ),
            "are not the same as the latitudes of the first sample",
        ),
        # The latitudes are not in degrees
        (
            default_transform,
            SampleBuilder.build_samples(
                [ta_var ta_var; pr_radians_var pr_radians_var];
                FT = Float64,
            ),
            "The unit for latitude is missing or is not degree",
        ),
    ]

    # The samples are not mutated when an error is thrown
    for (transform, error_collection, msg) in transforms_to_check
        deepcopy_collection = deepcopy(error_collection)
        @test_throws msg SampleBuilder.apply_transform!(
            deepcopy_collection,
            transform,
        )
        @test SampleBuilder.get_samples(deepcopy_collection) ==
              SampleBuilder.get_samples(error_collection)
    end
end

@testset "TransformedSampleCollection accessors" begin
    var =
        TemplateVar() |>
        add_dim("lat", [-90.0, 0.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "pr") |>
        one_to_n_data(collected = true) |>
        initialize
    sample_collection = SampleBuilder.build_samples([var var var]; FT = Float64)

    # A SampleCollection has no transforms
    @test SampleBuilder.base(sample_collection) === sample_collection
    @test SampleBuilder.transform_sequence(sample_collection) == ()
    @test SampleBuilder.apply_transform(sample_collection) === sample_collection

    lat_weighting = LatitudeWeighting()
    per_var_weighting = SampleBuilder.PerVariableWeighting([2.0])
    per_collection_weighting = PerCollectionWeighting(3.0)
    transformed_collection =
        sample_collection |>
        lat_weighting |>
        per_var_weighting |>
        per_collection_weighting

    @test transformed_collection isa SampleBuilder.TransformedSampleCollection
    @test SampleBuilder.base(transformed_collection) === sample_collection
    @test SampleBuilder.num_samples(transformed_collection) == 3
    @test SampleBuilder.transform_sequence(transformed_collection) ==
          (lat_weighting, per_var_weighting, per_collection_weighting)
    @test SampleBuilder.apply_transform(transformed_collection) isa
          ext.SampleCollection

    # Calling a transform is the same as piping into it
    @test SampleBuilder.transform_sequence(
        per_collection_weighting(sample_collection),
    ) == (per_collection_weighting,)

    @test sprint(show, transformed_collection) == """
        TransformedSampleCollection (3 transform(s) not yet applied)
        SampleCollection (3×3 matrix of Float64)
        3 sample(s), each 3 value(s) from 1 variable(s)
          ↳ LatitudeWeighting(min_cosd_lat = 0.1)
          ↳ PerVariableWeighting([2.0])
          ↳ PerCollectionWeighting(3.0)"""
end

@testset "Composition of transformations" begin
    pr_var =
        TemplateVar() |>
        add_dim("lat", [-90.0, 0.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "pr") |>
        one_to_n_data(collected = true) |>
        initialize
    tas_var =
        TemplateVar() |>
        add_dim("lon", [-90.0, 0.0, 30.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "tas") |>
        one_to_n_data(collected = true) |>
        initialize
    ta_var =
        TemplateVar() |>
        add_dim("lat", [0.0, 90.0], units = "degrees") |>
        add_attribs(short_name = "ta") |>
        one_to_n_data(collected = true) |>
        initialize

    # The rows of the samples are pr (1:3), tas (4:7), and ta (8:9)
    sample_collection = SampleBuilder.build_samples(
        [pr_var pr_var; tas_var tas_var; ta_var ta_var];
        FT = Float64,
    )
    samples = copy(SampleBuilder.get_samples(sample_collection))

    # With min_cosd_lat = 0.25, the weights at the latitudes 0 and ±90 degrees
    # are 1 and 2 respectively
    lat_weighting = LatitudeWeighting(min_cosd_lat = 0.25)
    per_var_weighting = SampleBuilder.PerVariableWeighting([2.0, 3.0, 4.0])
    per_collection_weighting = PerCollectionWeighting(5.0)
    expected_samples =
        5.0 .* [
            2.0 .* [2.0, 1.0, 2.0] .* samples[1:3, :];
            3.0 .* samples[4:7, :];
            4.0 .* [1.0, 2.0] .* samples[8:9, :]
        ]

    # The ordering of weighting does not matter, so every order gives the same
    # samples
    for (t1, t2, t3) in (
        (lat_weighting, per_var_weighting, per_collection_weighting),
        (lat_weighting, per_collection_weighting, per_var_weighting),
        (per_var_weighting, lat_weighting, per_collection_weighting),
        (per_var_weighting, per_collection_weighting, lat_weighting),
        (per_collection_weighting, lat_weighting, per_var_weighting),
        (per_collection_weighting, per_var_weighting, lat_weighting),
    )
        transformed_collection = sample_collection |> t1 |> t2 |> t3
        @test SampleBuilder.transform_sequence(transformed_collection) ==
              (t1, t2, t3)
        @test SampleBuilder.get_samples(
            SampleBuilder.apply_transform(transformed_collection),
        ) == expected_samples
    end

    # Applying the transforms eagerly one at a time gives the same samples
    eager_collection =
        SampleBuilder.apply_transform(sample_collection, lat_weighting)
    eager_collection =
        SampleBuilder.apply_transform(eager_collection, per_var_weighting)
    eager_collection = SampleBuilder.apply_transform(
        eager_collection,
        per_collection_weighting,
    )
    @test SampleBuilder.get_samples(eager_collection) == expected_samples

    # Eagerly applying a transform to a TransformedSampleCollection also
    # applies the transforms in the chain
    mixed_collection = SampleBuilder.apply_transform(
        sample_collection |> lat_weighting |> per_var_weighting,
        per_collection_weighting,
    )
    @test SampleBuilder.get_samples(mixed_collection) == expected_samples

    # Repeating a transform applies it again
    @test SampleBuilder.get_samples(
        SampleBuilder.apply_transform(
            sample_collection |>
            PerCollectionWeighting(2.0) |>
            PerCollectionWeighting(3.0),
        ),
    ) == 6.0 .* samples
    @test SampleBuilder.get_samples(
        SampleBuilder.apply_transform(
            sample_collection |> lat_weighting |> lat_weighting,
        ),
    ) == [
        [4.0, 1.0, 4.0] .* samples[1:3, :];
        samples[4:7, :];
        [1.0, 4.0] .* samples[8:9, :]
    ]

    # Latitude weighting pr and ta separately is the same as weighting every
    # variable with a latitude
    @test SampleBuilder.get_samples(
        SampleBuilder.apply_transform(
            sample_collection |>
            LatitudeWeighting(["pr"]; min_cosd_lat = 0.25) |>
            LatitudeWeighting(["ta"]; min_cosd_lat = 0.25),
        ),
    ) == SampleBuilder.get_samples(
        SampleBuilder.apply_transform(sample_collection |> lat_weighting),
    )

    # None of the compositions mutate the base sample collection
    @test SampleBuilder.get_samples(sample_collection) == samples
end

@testset "Implementation of custom transform" begin
    # Add a constant to the samples without mutating them and count the number
    # of times the transform is applied. Annotating the sample collection as a
    # SampleCollection checks that the transform is only applied to one.
    struct ShiftTransform <: AbstractTransform
        shift::Float64
        num_calls::Base.RefValue{Int}
        ShiftTransform(shift) = new(shift, Ref(0))
    end

    function SampleBuilder.apply_transform!(
        sample_collection::ext.SampleCollection,
        transform::ShiftTransform,
    )
        transform.num_calls[] += 1
        return ext.SampleCollection(
            SampleBuilder.get_samples(sample_collection) .+ transform.shift,
            SampleBuilder.get_metadata(sample_collection),
        )
    end

    var =
        TemplateVar() |>
        add_dim("lat", [-60.0, 0.0, 60.0], units = "degrees") |>
        add_attribs(short_name = "hi", start_date = "2007-12-1") |>
        one_to_n_data(collected = true) |>
        initialize
    sample_collection = SampleBuilder.build_samples([var]; FT = Float64)
    samples = copy(SampleBuilder.get_samples(sample_collection))

    # Building and showing the transformed sample collection do not apply the
    # transform
    transform = ShiftTransform(1.0)
    transformed_collection = sample_collection |> transform |> transform
    @test transformed_collection isa SampleBuilder.TransformedSampleCollection
    @test SampleBuilder.base(transformed_collection) === sample_collection
    @test SampleBuilder.num_samples(transformed_collection) ==
          SampleBuilder.num_samples(sample_collection)
    @test SampleBuilder.transform_sequence(transformed_collection) ==
          (transform, transform)
    @test occursin("↳ " * repr(transform), sprint(show, transformed_collection))
    @test transform.num_calls[] == 0

    # The transform is applied once for each time it appears in the chain and
    # its return value is used
    shifted_collection = SampleBuilder.apply_transform(transformed_collection)
    @test SampleBuilder.get_samples(shifted_collection) == samples .+ 2.0
    @test transform.num_calls[] == 2
    @test SampleBuilder.get_samples(sample_collection) == samples

    # Eager application
    eager_collection =
        SampleBuilder.apply_transform(sample_collection, ShiftTransform(1.0))
    @test SampleBuilder.get_samples(eager_collection) == samples .+ 1.0

    # Transforms are applied in the order of the chain, including built-in
    # transforms
    shift_transform = ShiftTransform(1.0)
    per_collection_weighting = PerCollectionWeighting(2.0)
    shift_then_weight =
        sample_collection |> shift_transform |> per_collection_weighting
    weight_then_shift =
        sample_collection |> per_collection_weighting |> shift_transform
    @test SampleBuilder.transform_sequence(shift_then_weight) ==
          (shift_transform, per_collection_weighting)
    @test SampleBuilder.get_samples(
        SampleBuilder.apply_transform(shift_then_weight),
    ) == 2.0 .* (samples .+ 1.0)
    @test SampleBuilder.get_samples(
        SampleBuilder.apply_transform(weight_then_shift),
    ) == 2.0 .* samples .+ 1.0
    @test SampleBuilder.get_samples(sample_collection) == samples
end

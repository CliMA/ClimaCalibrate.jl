import ClimaCalibrate.EnsembleBuilder:
    AbstractMatcher,
    ShortNameMatcher,
    DimNameMatcher,
    DimUnitsMatcher,
    UnitsMatcher,
    DimValuesMatcher,
    SignMatcher,
    SequentialIndicesMatcher

"""
    EnsembleBuilder.match(
        ::ShortNameMatcher,
        var::OutputVar,
        metadata::Metadata;
        verbose = false,
    )

Return `true` if `var` and `metadata` have the same short name, `false`
otherwise.
"""
function EnsembleBuilder.match(
    ::ShortNameMatcher,
    var::OutputVar,
    metadata::Metadata;
    verbose = false,
)
    # Do not need to check if the short name is there, since we already know that
    # the short name of metadata exists
    return ClimaAnalysis.short_name(var) == ClimaAnalysis.short_name(metadata)
end

"""
    EnsembleBuilder.match(
        ::DimNameMatcher,
        var::OutputVar,
        metadata::Metadata;
        verbose = false,
    )

Return `true` if `var` and `metadata` have the same dimensions, `false`
otherwise.
"""
function EnsembleBuilder.match(
    ::DimNameMatcher,
    var::OutputVar,
    metadata::Metadata;
    verbose = false,
)
    return issetequal(
        ClimaAnalysis.conventional_dim_name.(keys(var.dims)),
        ClimaAnalysis.conventional_dim_name.(keys(metadata.dims)),
    )
end

"""
    EnsembleBuilder.match(
        ::DimUnitsMatcher,
        var::OutputVar,
        metadata::Metadata;
        verbose = false,
    )

Return `true` if the units of the dimensions in `var` and `metadata` are the
same, `false` otherwise. This function assumes `var` and `metadata` have the
same dimensions.
"""
function EnsembleBuilder.match(
    ::DimUnitsMatcher,
    var::OutputVar,
    metadata::Metadata;
    verbose = false,
)
    for var_dim_name in keys(var.dims)
        md_dim_name = ClimaAnalysis.Var.find_corresponding_dim_name_in_var(
            var_dim_name,
            metadata,
        )
        var_dim_units = ClimaAnalysis.dim_units(var, var_dim_name)
        md_dim_units = ClimaAnalysis.dim_units(metadata, md_dim_name)
        var_dim_units == md_dim_units || return false
        if var_dim_units == "" || md_dim_units == ""
            @warn(
                "Units for $(ClimaAnalysis.conventional_dim_name(var_dim_name)) is missing in var or metadata"
            )
        end
    end
    return true
end

"""
    EnsembleBuilder.match(
        ::UnitsMatcher,
        var::OutputVar,
        metadata::Metadata;
        verbose = false,
    )

Return `true` if `var` and `metadata` have the same units, `false` otherwise.
"""
function EnsembleBuilder.match(
    ::UnitsMatcher,
    var::OutputVar,
    metadata::Metadata;
    verbose = false,
)
    var_units = ClimaAnalysis.units(var)
    metadata_units = ClimaAnalysis.units(metadata)
    if var_units == "" || metadata_units == ""
        @warn("Var or metadata may be missing units")
    end
    return var_units == metadata_units
end

"""
    EnsembleBuilder.match(
        ::DimValuesMatch,
        var::OutputVar,
        metadata::Metadata;
        verbose = false,
    )

Match if the values of the dimensions in `var` and `metadata` are compatible
for the purpose of filling out the G ensemble matrix.

The nontemporal dimensions are compatible if the values are approximately the
same. The temporal dimensions are compatible if the temporal dimension of
`metadata` is a subset of the temporal dimension of `var`.
"""
function EnsembleBuilder.match(
    ::DimValuesMatcher,
    var::OutputVar,
    metadata::Metadata;
    verbose = false,
)
    for var_dim_name in keys(var.dims)
        md_dim_name = ClimaAnalysis.Var.find_corresponding_dim_name_in_var(
            var_dim_name,
            metadata,
        )
        if ClimaAnalysis.conventional_dim_name(var_dim_name) != "time"
            all(isapprox(var.dims[var_dim_name], metadata.dims[md_dim_name])) ||
                return false
        else
            # For the temporal dimension, only check if the times of metadata is
            # a subset of the times of var, because _match_dates will get the
            # correct dates for us
            dates_or_times(metadata) ⊆ dates_or_times(var) || return false
        end
    end
    return true
end

"""
    EnsembleBuilder.match(
        ::SignMatcher,
        var::OutputVar,
        metadata::Metadata;
        verbose = false,
    )

TODO
"""
function EnsembleBuilder.match(
    ::SignMatcher,
    var::OutputVar,
    metadata::Metadata;
    verbose = false,
)
    error("Not yet implemented")
end

"""
    EnsembleBuilder.match(
        ::SequentialIndicesMatcher,
        var::OutputVar,
        metadata::Metadata;
        verbose = false,
    )

TODO
"""
function EnsembleBuilder.match(
    ::SequentialIndicesMatcher,
    var::OutputVar,
    metadata::Metadata;
    verbose = false,
)
    error("Not yet implemented")
end

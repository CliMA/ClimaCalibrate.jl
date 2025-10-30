import ClimaCalibrate.Checker:
    AbstractChecker,
    ShortNameChecker,
    DimNameChecker,
    DimUnitsChecker,
    UnitsChecker,
    DimValuesChecker,
    SequentialIndicesChecker,
    SignChecker
import ClimaCalibrate.Checker

"""
    Checker.check(
        ::ShortNameChecker,
        var::OutputVar,
        metadata::Metadata;
        data = nothing,
        verbose = false,
    )

Return `true` if `var` and `metadata` have the same short name, `false`
otherwise.
"""
function Checker.check(
    ::ShortNameChecker,
    var::OutputVar,
    metadata::Metadata;
    data = nothing,
    verbose = false,
)
    # Do not need to check if the short name is there, since we already know that
    # the short name of metadata exists
    var_short_name = ClimaAnalysis.short_name(var)
    metadata_short_name = ClimaAnalysis.short_name(metadata)
    same_short_name = var_short_name == metadata_short_name
    verbose &&
        !same_short_name &&
        @info "Short names are not the same between the OutputVar ($var_short_name) and metadata ($metadata_short_name)"
    return same_short_name
end

"""
    Checker.check(
        ::DimNameChecker,
        var::OutputVar,
        metadata::Metadata;
        data = nothing,
        verbose = false,
    )

Return `true` if `var` and `metadata` have the same dimensions, `false`
otherwise.
"""
function Checker.check(
    ::DimNameChecker,
    var::OutputVar,
    metadata::Metadata;
    data = nothing,
    verbose = false,
)
    var_dim_names = ClimaAnalysis.conventional_dim_name.(keys(var.dims))
    metadata_dim_names =
        ClimaAnalysis.conventional_dim_name.(keys(metadata.dims))
    same_dim_names = issetequal(var_dim_names, metadata_dim_names)
    verbose &&
        !same_dim_names &&
        @info "Dimensions are not the same between the OutputVar ($var_dim_names) and metadata ($metadata_dim_names)"
    return same_dim_names
end

"""
    Checker.check(
        ::DimUnitsChecker,
        var::OutputVar,
        metadata::Metadata;
        data = nothing,
        verbose = false,
    )

Return `true` if the units of the dimensions in `var` and `metadata` are the
same, `false` otherwise. This function assumes `var` and `metadata` have the
same dimensions.
"""
function Checker.check(
    ::DimUnitsChecker,
    var::OutputVar,
    metadata::Metadata;
    data = nothing,
    verbose = false,
)
    for var_dim_name in keys(var.dims)
        md_dim_name = ClimaAnalysis.Var.find_corresponding_dim_name_in_var(
            var_dim_name,
            metadata,
        )
        var_dim_units = ClimaAnalysis.dim_units(var, var_dim_name)
        md_dim_units = ClimaAnalysis.dim_units(metadata, md_dim_name)
        same_dim_units = var_dim_units == md_dim_units
        verbose &&
            !same_dim_units &&
            @info "Units of dimension ($var_dim_name) are not the same between OutputVar ($var_dim_units) and metadata ($md_dim_units)"
        same_dim_units || return false
        if var_dim_units == "" || md_dim_units == ""
            @warn(
                "Units for $(ClimaAnalysis.conventional_dim_name(var_dim_name)) is missing in var or metadata"
            )
        end
    end
    return true
end

"""
    Checker.check(
        ::UnitsChecker,
        var::OutputVar,
        metadata::Metadata;
        data = nothing,
        verbose = false,
    )

Return `true` if `var` and `metadata` have the same units, `false` otherwise.
"""
function Checker.check(
    ::UnitsChecker,
    var::OutputVar,
    metadata::Metadata;
    data = nothing,
    verbose = false,
)
    var_units = ClimaAnalysis.units(var)
    metadata_units = ClimaAnalysis.units(metadata)
    if var_units == "" || metadata_units == ""
        @warn("Var or metadata may be missing units")
    end
    same_units = var_units == metadata_units
    verbose &&
        !same_units &&
        @info "Units of OutputVar ($var_units) are not the same as units of metadata ($metadata_units)"
    return same_units
end

"""
    Checker.check(
        ::DimValuesMatch,
        var::OutputVar,
        metadata::Metadata;
        data = nothing,
        verbose = false,
    )

Return `true` if the values of the dimensions in `var` and `metadata` are
compatible for the purpose of filling out the G ensemble matrix, `false`
otherwise.

The nontemporal dimensions are compatible if the values are approximately the
same. The temporal dimensions are compatible if the temporal dimension of
`metadata` is a subset of the temporal dimension of `var`.
"""
function Checker.check(
    ::DimValuesChecker,
    var::OutputVar,
    metadata::Metadata;
    data = nothing,
    verbose = false,
)
    for var_dim_name in keys(var.dims)
        md_dim_name = ClimaAnalysis.Var.find_corresponding_dim_name_in_var(
            var_dim_name,
            metadata,
        )
        if ClimaAnalysis.conventional_dim_name(var_dim_name) != "time"
            same_dim_values = all(
                isapprox(var.dims[var_dim_name], metadata.dims[md_dim_name]),
            )
            if !same_dim_values
                verbose &&
                    @info "Values of dimension ($var_dim_name) in OutputVar ($(var.dims[var_dim_name])) is not the same as values of dimension in metadata ($(metadata.dims[md_dim_name])))"
                return false
            end
        else
            # For the temporal dimension, only check if the times of metadata is
            # a subset of the times of var, because _match_dates will get the
            # correct dates for us
            var_dates_or_times = dates_or_times(var)
            metadata_dates_or_times = dates_or_times(metadata)
            subset_of_var_dates = metadata_dates_or_times ⊆ var_dates_or_times
            if !subset_of_var_dates
                verbose &&
                    @info "Dates/times of metadata ($metadata_dates_or_times) is not a subset of the dates/times of the OutputVar ($var_dates_or_times)"
                return false
            end
        end
    end
    return true
end

"""
    Checker.check(
        ::SequentialIndicesChecker,
        var::OutputVar,
        metadata::Metadata;
        data = nothing,
        verbose = false,
    )

Return `true` if the dates of `var` map to sequential indices of the dates of
`metadata`, `false` otherwise.

!!! note "Use this check"
    It is recommended to always enable this check when possible.

!!! note "Why use this check?"
    This check is helpful in ensuring that the dates are matched correctly
    between `var` and `metadata`. For example, without this check, if the
    simulation data contain monthly averages and metadata track seasonal
    averages, then no error is thrown, because all dates in `metadata` are in
    all the dates in `var`.
"""
function Checker.check(
    ::SequentialIndicesChecker,
    var::OutputVar,
    metadata::Metadata;
    data = nothing,
    verbose = false,
)
    obs_dates = ClimaAnalysis.dates(metadata)
    sim_dates = ClimaAnalysis.dates(var)
    sim_indices_for_obs_dates = indexin(obs_dates, sim_dates)

    length(sim_indices_for_obs_dates) == 1 && @warn(
        "There is only one date in the metadata. SequentialIndicesChecker will always return true"
    )

    # Do not need to check for nothing in sim_indices_for_obs_dates because
    # of DimValuesChecker
    for i in eachindex(sim_indices_for_obs_dates)[2:end]
        if sim_indices_for_obs_dates[i] != sim_indices_for_obs_dates[i - 1] + 1
            verbose &&
                @info "Dates of OutputVar ($sim_dates) do not map to sequential indices ($sim_indices_for_obs_dates) of the dates of the metadata ($obs_dates)."
            return false
        end
    end
    return true
end

"""
    Checker.check(
        ::SignChecker,
        var::OutputVar,
        metadata::Metadata;
        data,
        verbose = false,
    )

Return `true` if the absolute difference of the proportion of positive values in
`var.data` and the proportion of positive values in `data` is less than the threshold
defined in `SignChecker`, `false` otherwise.
"""
function Checker.check(
    checker::SignChecker,
    var::OutputVar,
    metadata::Metadata;
    data,
    verbose = false,
)
    obs_pos_proportion = mean(data .> 0)

    # This is inaccurate, because not all the values in var.data will end up in
    # the G ensemble matrix. See _match_dates for one case. However, the mean
    # should not change that much with additional times.
    valid = @. !isnan(var.data)
    sim_pos_proportion = sum(valid .& (var.data .> 0)) / sum(valid)

    same_sign = abs(obs_pos_proportion - sim_pos_proportion) < checker.threshold
    !same_sign &&
        verbose &&
        @info "Proportion of positive values in the simulation data ($sim_pos_proportion) is not the same as the proportion of positive values in the observational data ($obs_pos_proportion)"
    return same_sign
end

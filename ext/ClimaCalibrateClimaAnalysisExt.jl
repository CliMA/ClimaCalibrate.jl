module ClimaCalibrateClimaAnalysisExt

import ClimaAnalysis
import ClimaAnalysis: OutputVar
import ClimaAnalysis.Var: Metadata

import EnsembleKalmanProcesses as EKP

import Dates
import Statistics
import Statistics: mean
import NaNStatistics: nanmean, nanvar

import LinearAlgebra: Diagonal, I, diagind

# Used by both by SampleBuilder and ObservationRecipe for default flattening
# of OutputVars
const FLATTENED_DIMS =
    ("longitude", "latitude", "pressure_level", "x", "y", "z", "time")

include("utils.jl")
include("checkers.jl")
include("sample_builder.jl")
include("observation_recipe.jl")
include("ensemble_builder.jl")

end

module ClimaAnalysisExt

import ClimaAnalysis
import ClimaAnalysis: OutputVar
import ClimaAnalysis.Var: Metadata

import EnsembleKalmanProcesses as EKP

import Dates
import Statistics
import Statistics: mean
import NaNStatistics: nanmean, nanvar

import LinearAlgebra: Diagonal, I, diagind

include("utils.jl")
include("observation_recipe.jl")
include("checkers.jl")
include("ensemble_builder.jl")

end

module VorLap
#TODO change to import
import CSV, DataFrames, HDF5, LinearAlgebra, Interpolations, Plots, DelimitedFiles,PlotlyJS
using Printf
const modulepath = splitdir(@__FILE__)[1]

Plots.plotlyjs()  # switch backend to PlotlyJS for interactive 3D plots

include("$modulepath/structs.jl")
include("$modulepath/vorlap_utils.jl")
include("$modulepath/fileio.jl")

end
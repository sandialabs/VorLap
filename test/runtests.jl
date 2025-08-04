path = splitdir(@__FILE__)[1]
using Test

@testset "Run Gold Test" begin
    include("$path/testvorlap.jl")
end
using DistributedEmitterOpt
using Test

@testset "DistributedEmitterOpt.jl" begin
    @testset "Gradient Tests" begin
        include("gradient_tests.jl")
    end

    @testset "Constraint Tests" begin
        include("constraint_tests.jl")
    end

    @testset "Constrained Optimization Tests" begin
        include("constrained_optimization_tests.jl")
    end

    @testset "Objective Baselines Tests" begin
        include("objective_baselines_tests.jl")
    end

    @testset "Logger Tests" begin
        include("logger_tests.jl")
    end

    @testset "Material Dataset Tests" begin
        include("material_datasets_tests.jl")
    end

    @testset "PML Tests" begin
        include("pml_tests.jl")
    end
end

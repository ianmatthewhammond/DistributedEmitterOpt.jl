using DistributedEmitterOpt
using Test

@testset "DistributedEmitterOpt.jl" begin
    @testset "Gradient Tests" begin
        include("gradient_tests.jl")
    end

    @testset "Surface measure tests" begin
        include("surface_tv_void_tests.jl")
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

    @testset "Mesh Repair Tests" begin
        include("mesh_repair_tests.jl")
    end

    @testset "Logger Tests" begin
        include("logger_tests.jl")
    end

    @testset "Projection Derivative Tests" begin
        include("projection_derivative_tests.jl")
    end

    @testset "PML Tests" begin
        include("pml_tests.jl")
    end

    # Separate namespaces: these files define their own fixtures and constants.
    for (name, file) in (("Optimizer Safety Tests", "plain_optimizer_safety_tests.jl"),
                         ("Checkpoint Tolerance Tests", "raw_tolerance_checkpoint_tests.jl"),
                         ("Checkpoint State Tests", "checkpoint_state_tests.jl"),
                         ("Mixed Output Gradient Tests", "mixed_output_gradient_tests.jl"))
        @testset "$name" begin
            sandbox = Module(gensym(:Tests))
            Core.eval(sandbox, :(include(path) = Base.include($sandbox, path)))
            Base.include(sandbox, joinpath(@__DIR__, file))
        end
    end

    include("sidewall_runtests.jl")

    # Each in its own process: NLopt.jl and the standalone library both export
    # `nlopt_*` symbols, and sharing a process would invalidate the parity check.
    @testset "Standalone Backend Tests" begin
        if STANDALONE_LIBRARY_READY
            for file in ("test_mmaccsa_backend.jl", "test_backend_parity.jl")
                cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $(joinpath(@__DIR__, file))`
                @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
            end
        else
            @test_skip "needs NLopt-MMA-CCSA"
        end
    end
end

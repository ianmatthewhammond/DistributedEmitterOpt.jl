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

    # Oblique incidence (docs/oblique_fields.md). DEO_SLOW_TESTS=1 adds the longest files: the
    # general-angle gradient ladders, the reflection/cell-reduction symmetry suite, and the
    # MUMPSSolver, PardisoSolver and CUDSSSolver gradients (skipped without MUMPS.jl, without
    # Pardiso.jl and MKL, and without CUDSS.jl and a CUDA GPU). mixed_output_gradient_tests.jl
    # runs above, in its own namespace.
    @testset "Oblique incidence" begin
        for file in ("oblique_fresnel_tests.jl", "oblique_film_reference_tests.jl",
                     "oblique_normal_equivalence_tests.jl", "oblique_order_guard_tests.jl", "oblique_api_tests.jl",
                     "oblique_mesh_reuse_tests.jl", "oblique_solver_guard_tests.jl",
                     "backend_adjoint_tests.jl", "oblique_negative_control_tests.jl",
                     "oblique_symmetric_adjoint_control_tests.jl", "oblique_legacy_gradient_tests.jl",
                     "cudss_solver_tests.jl")
            include(file)
        end
        if get(ENV, "DEO_SLOW_TESTS", "0") == "1"
            include("oblique_gradient_tests.jl")
            include("oblique_symmetry_tests.jl")
            include("mumps_gradient_tests.jl")
            include("pardiso_oblique_gradient_tests.jl")
            include("cudss_gradient_tests.jl")
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

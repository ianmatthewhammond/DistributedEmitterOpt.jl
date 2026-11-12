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

    @testset "Logger Tests" begin
        include("logger_tests.jl")
    end

    @testset "PML Tests" begin
        include("pml_tests.jl")
    end

    # Oblique incidence (docs/oblique_fields.md). DEO_SLOW_TESTS=1 adds the longest files: the
    # general-angle gradient ladders, the reflection/cell-reduction symmetry suite, and the
    # MUMPSSolver and PardisoSolver gradients (skipped without MUMPS.jl, and without Pardiso.jl
    # and MKL).
    @testset "Oblique incidence" begin
        for file in ("oblique_fresnel_tests.jl", "oblique_film_reference_tests.jl",
                     "oblique_normal_equivalence_tests.jl", "oblique_order_guard_tests.jl", "oblique_api_tests.jl",
                     "oblique_mesh_reuse_tests.jl", "oblique_solver_guard_tests.jl",
                     "backend_adjoint_tests.jl", "oblique_negative_control_tests.jl",
                     "oblique_symmetric_adjoint_control_tests.jl", "oblique_legacy_gradient_tests.jl",
                     "mixed_output_gradient_tests.jl")
            include(file)
        end
        if get(ENV, "DEO_SLOW_TESTS", "0") == "1"
            include("oblique_gradient_tests.jl")
            include("oblique_symmetry_tests.jl")
            include("mumps_gradient_tests.jl")
            include("pardiso_oblique_gradient_tests.jl")
        end
    end

    include("sidewall_runtests.jl")
end

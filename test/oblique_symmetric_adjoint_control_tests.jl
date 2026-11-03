# Added check. The correct oblique operator is NOT complex symmetric: A(kappa)^T = A(-kappa).
# An adjoint that assumes symmetry (solves with conj(A) instead of A') is right at normal
# incidence and wrong at any tilt. This control must PASS at zero tilt and FAIL at a tilt;
# if it passes at a tilt the operator has lost its kappa dependence.
include("oblique_common.jl")
struct SymmetricAssumptionSolver <: DEO.AbstractSolver end
DEO.lu!(::SymmetricAssumptionSolver, A) = lu(A)
DEO.filter_lu!(::SymmetricAssumptionSolver, A) = lu(A)
DEO.solve!(::SymmetricAssumptionSolver, F, b::Vector) = F \ b
DEO.solve_adjoint!(::SymmetricAssumptionSolver, F, b::Vector) = conj(F \ conj(b))   # = A' \ b only if A = A^T

@testset "Symmetric-operator assumption in the adjoint" begin
    for (name, fc, should_pass) in (("normal", FieldConfig(532; tilt_x=0, pol=:y), true),
                                    ("tilt x 20", FieldConfig(532; tilt_x=20, pol=:x), false),
                                    ("tilt x 20, y 10", FieldConfig(532; tilt_x=20, tilt_y=10, pol=:x), false))
        good = angular_test_problem([fc], [fc])
        bad = OptimizationProblem(good.pde, good.objective, good.sim, SymmetricAssumptionSolver();
            foundry_mode=true, control=good.control)
        _, e = angular_ladder(bad)
        println("SYMMETRIC-ASSUMPTION $name best relative error = $(minimum(e))")
        should_pass ? (@test minimum(e) < 1e-6) : (@test minimum(e) > 1e-2)
    end
end

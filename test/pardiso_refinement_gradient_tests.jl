# PardisoSolver's `refinement_steps` on whole problems: objective and adjoint gradient with
# refinement at MKL's automatic rule (0) and at two steps (2), against the default solver, to
# 1e-10. One problem has a 45° pump and a normal emission, so it holds full-storage (mtype 13)
# and complex symmetric (mtype 6) Maxwell factors side by side; the other is at normal
# incidence, all mtype 6. Every Maxwell factor must carry the value asked for. Skipped where
# Pardiso.jl or MKL is not available.
include("oblique_common.jl")
Base.find_package("Pardiso") !== nothing && @eval import Pardiso

# Objective and gradient at one random design, on a problem of its own with the given solver.
function refinement_evaluation(prob, solver; seed=1116)
    p = 0.4 .+ 0.2 .* rand(MersenneTwister(seed), length(prob.p))
    q = OptimizationProblem(prob.pde, prob.objective, prob.sim, solver;
        foundry_mode=true, control=prob.control)
    grad = zeros(length(p))
    g = objective_and_gradient!(grad, p, q)
    return (; g, grad, q)
end

refinement_pools(pool::DEO.SolverCachePool) = [pool]
refinement_pools(bundle::DEO.SolverPoolBundle) = [bundle.default; collect(values(bundle.by_pol))]
# (mtype, iparm(8)) of every Maxwell factor of a problem after an evaluation.
maxwell_refinement(prob) = [(c.A_factor.mtype, Int(Pardiso.get_iparm(c.A_factor.ps, 8)))
    for pool in refinement_pools(prob.pool) for (_, c) in pool.caches if c.A_factor !== nothing]

if !(Base.get_extension(DEO, :PardisoExt) !== nothing && Pardiso.mkl_is_available())
    @info "Pardiso.jl with MKL is not available; skipping the PardisoSolver refinement gradient tests"
else
    @testset "PardisoSolver refinement_steps on whole problems" begin
        pump = [FieldConfig(532; tilt_y=45, pol=:x, weight=0.5), FieldConfig(532; tilt_y=45, pol=:y, weight=0.5)]
        mixed = angular_test_problem(pump, [FieldConfig(532; tilt_y=0, pol=:y)])
        fc = FieldConfig(532; tilt_x=0, pol=:y)
        normal = angular_test_problem([fc], [fc])
        for (label, prob, types) in (("45° pump, normal emission", mixed, [6, 13]), ("normal incidence", normal, [6]))
            ref = refinement_evaluation(prob, PardisoSolver())
            defaults = maxwell_refinement(ref.q)
            @test sort(unique(first.(defaults))) == types
            println("PARDISO refinement, $label: default (mtype, iparm(8)) per Maxwell factor = $defaults")
            for steps in (0, 2)
                e = refinement_evaluation(prob, PardisoSolver(refinement_steps=steps))
                set = maxwell_refinement(e.q)
                @test sort(unique(first.(set))) == types
                @test all(==(steps), last.(set))
                dg = abs(e.g - ref.g) / abs(ref.g)
                dgrad = norm(e.grad - ref.grad) / norm(ref.grad)
                println("PARDISO refinement, $label: refinement_steps=$steps against the default: objective=$dg gradient=$dgrad")
                @test dg < 1e-10
                @test dgrad < 1e-10
            end
        end
    end
end
GC.gc()

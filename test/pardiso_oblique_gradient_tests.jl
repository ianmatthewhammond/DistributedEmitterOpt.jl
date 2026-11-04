# PardisoSolver (MKL Pardiso through Pardiso.jl) on oblique problems: adjoint gradients against
# central differences for a 45° pump (s and p, weight 0.5 each) with an oblique and with a
# normal emission, and objectives and gradients of the automatic Maxwell type against forced
# mtype 13 and UMFPACK, at 45° and at normal incidence. The automatic type must be 13 for tilted
# fields and 6 for normal ones, including the normal emission of a tilted pump. The old Pardiso
# adjoint, conj(A⁻¹ conj b) on the same factorization, must fail the 45° ladder. Skipped where
# Pardiso.jl or MKL is not available.
include("oblique_common.jl")
Base.find_package("Pardiso") !== nothing && @eval import Pardiso

# The previous Pardiso adjoint, kept as a negative control.
struct PardisoShortcutSolver <: DEO.AbstractSolver end
DEO.lu!(::PardisoShortcutSolver, A) = DEO.lu!(PardisoSolver(), A)
DEO.lu!(::PardisoShortcutSolver, f, A) = DEO.lu!(PardisoSolver(), f, A)
DEO.filter_lu!(::PardisoShortcutSolver, A) = DEO.filter_lu!(PardisoSolver(), A)
DEO.filter_lu!(::PardisoShortcutSolver, f, A) = DEO.filter_lu!(PardisoSolver(), f, A)
DEO.solve!(::PardisoShortcutSolver, f, b::Vector) = DEO.solve!(PardisoSolver(), f, b)
DEO.solve_adjoint!(::PardisoShortcutSolver, f, b::Vector) = conj(DEO.solve!(PardisoSolver(), f, conj(b)))
DEO.release_factor!(::PardisoShortcutSolver, f) = DEO.release_factor!(PardisoSolver(), f)

with_solver(prob, solver) = OptimizationProblem(prob.pde, prob.objective, prob.sim, solver;
    foundry_mode=true, control=prob.control)

# Objective and gradient at one random design for each solver, each on its own problem.
function evaluations(prob, solvers; seed=1103)
    p = 0.4 .+ 0.2 .* rand(MersenneTwister(seed), length(prob.p))
    map(solvers) do solver
        q = with_solver(prob, solver)
        grad = zeros(length(p))
        t = @elapsed g = objective_and_gradient!(grad, p, q)
        (; g, grad, t, q)
    end
end

# The Pardiso mtype of each field configuration's Maxwell factor after an evaluation.
pools(pool::DEO.SolverCachePool) = [pool]
pools(bundle::DEO.SolverPoolBundle) = [bundle.default; collect(values(bundle.by_pol))]
maxwell_mtypes(prob) = Dict(k => c.A_factor.mtype for pool in pools(prob.pool)
    for (k, c) in pool.caches if c.A_factor !== nothing)

# Automatic type against forced mtype 13 and UMFPACK; returns the automatic types.
function compare_types(label, prob)
    umfpack, auto, full = evaluations(prob, (UmfpackSolver(), PardisoSolver(),
        PardisoSolver(assume_symmetric_maxwell=false)))
    for (name, x, y) in (("auto vs UMFPACK", auto, umfpack), ("auto vs forced 13", auto, full),
                         ("forced 13 vs UMFPACK", full, umfpack))
        dg = abs(x.g - y.g) / abs(y.g)
        dgrad = norm(x.grad - y.grad) / norm(y.grad)
        println("PARDISO $label $name: objective=$dg gradient=$dgrad")
        @test dg < 1e-10
        @test dgrad < 1e-10
    end
    println("PARDISO $label seconds: umfpack=$(umfpack.t) auto=$(auto.t) forced13=$(full.t)")
    @test all(==(13), values(maxwell_mtypes(full.q)))
    types = maxwell_mtypes(auto.q)
    println("PARDISO $label automatic types: $types")
    return types
end

if !(Base.get_extension(DEO, :PardisoExt) !== nothing && Pardiso.mkl_is_available())
    @info "Pardiso.jl with MKL is not available; skipping the PardisoSolver oblique tests"
else
    @testset "PardisoSolver on oblique problems" begin
        pump = [FieldConfig(532; tilt_y=45, pol=:x, weight=0.5), FieldConfig(532; tilt_y=45, pol=:y, weight=0.5)]

        # 45° pump, oblique emission: every factor is unsymmetric.
        emission = [FieldConfig(532; tilt_y=-20, pol=:y)]
        oblique = angular_test_problem(pump, emission)
        sim = DEO.default_sim(oblique.sim)
        @test sim.per_y && !sim.per_x
        g, errs = angular_ladder(with_solver(oblique, PardisoSolver()))
        println("PARDISO 45° s+p pump, -20° p emission: g=$g ladder=$errs")
        @test g > 0 && isfinite(g)
        @test minimum(errs) < 1e-4
        @test errs[end] < 1e-4
        _, bad = angular_ladder(with_solver(oblique, PardisoShortcutSolver()))
        println("PARDISO shortcut adjoint (negative control): ladder=$bad")
        @test minimum(bad) > 1e-2
        types = compare_types("oblique emission", oblique)
        @test all(==(13), values(types))

        # 45° pump, normal emission on the same half cell: the emission factor is symmetric.
        normal_emission = [FieldConfig(532; tilt_y=0, pol=:y)]
        mixed = angular_test_problem(pump, normal_emission)
        g, errs = angular_ladder(with_solver(mixed, PardisoSolver()))
        println("PARDISO 45° s+p pump, normal p emission: g=$g ladder=$errs")
        @test minimum(errs) < 1e-4
        @test errs[end] < 1e-4
        types = compare_types("normal emission", mixed)
        @test types[cache_key(pump[1])] == 13
        @test types[cache_key(pump[2])] == 13
        @test types[cache_key(normal_emission[1])] == 6

        # Normal incidence on the quarter cell: every factor is symmetric.
        fc = FieldConfig(532; tilt_x=0, pol=:y)
        normal = angular_test_problem([fc], [fc])
        types = compare_types("normal", normal)
        @test !isempty(types) && all(==(6), values(types))
    end
end
GC.gc()

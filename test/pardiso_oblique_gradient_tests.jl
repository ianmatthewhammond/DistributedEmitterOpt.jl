# PardisoSolver (MKL Pardiso through Pardiso.jl) on oblique problems: adjoint gradients against
# central differences for a 45° pump (s and p, weight 0.5 each) with an oblique emission, and
# objectives and gradients against UMFPACK at 45° and at normal incidence. The old Pardiso
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

# Relative objective and gradient differences against UMFPACK at one random design.
function against_umfpack(prob, solver; seed=1103)
    p = 0.4 .+ 0.2 .* rand(MersenneTwister(seed), length(prob.p))
    gu = zeros(length(p)); gp = zeros(length(p))
    tu = @elapsed fu = objective_and_gradient!(gu, p, prob)
    tp = @elapsed fp = objective_and_gradient!(gp, p, with_solver(prob, solver))
    return abs(fp - fu) / abs(fu), norm(gp - gu) / norm(gu), tu, tp
end

if !(Base.get_extension(DEO, :PardisoExt) !== nothing && Pardiso.mkl_is_available())
    @info "Pardiso.jl with MKL is not available; skipping the PardisoSolver oblique tests"
else
    @testset "PardisoSolver on oblique problems" begin
        pump = [FieldConfig(532; tilt_y=45, pol=:x, weight=0.5), FieldConfig(532; tilt_y=45, pol=:y, weight=0.5)]
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
        df, dg, tu, tp = against_umfpack(oblique, PardisoSolver())
        println("PARDISO vs UMFPACK oblique: objective=$df gradient=$dg seconds umfpack=$tu pardiso=$tp")
        @test df < 1e-8
        @test dg < 1e-8

        normal = angular_test_problem([FieldConfig(532; tilt_x=0, pol=:y)], [FieldConfig(532; tilt_x=0, pol=:y)])
        for (name, s) in (("mtype 13", PardisoSolver()), ("mtype 6", PardisoSolver(assume_symmetric_maxwell=true)))
            df, dg, tu, tp = against_umfpack(normal, s)
            println("PARDISO vs UMFPACK normal $name: objective=$df gradient=$dg seconds umfpack=$tu pardiso=$tp")
            @test df < 1e-8
            @test dg < 1e-8
        end
    end
end
GC.gc()

# MUMPSSolver in the optimization loop: objectives and gradients against UMFPACK at normal
# incidence (unsymmetric and symmetric storage) and at 45° (s+p pump with an oblique and with a
# normal emission), at two designs so that the second evaluation runs on refactorized factors;
# and an adjoint ladder against central differences at 45°. Skipped where MUMPS.jl is not
# installed.
include("oblique_common.jl")
if Base.find_package("MUMPS") !== nothing
    @eval using MUMPS
    MUMPS.MPI.Initialized() || MUMPS.MPI.Init()
end

with_solver(prob, solver) = OptimizationProblem(prob.pde, prob.objective, prob.sim, solver;
    foundry_mode=true, control=prob.control)

function against_umfpack(label, prob, solver)
    rng = MersenneTwister(1103)
    reference, candidate = with_solver(prob, UmfpackSolver()), with_solver(prob, solver)
    for design in 1:2
        p = 0.4 .+ 0.2 .* rand(rng, length(prob.p))
        gu = zeros(length(p)); gm = zeros(length(p))
        fu = objective_and_gradient!(gu, p, reference)
        t = @elapsed fm = objective_and_gradient!(gm, p, candidate)
        df, dg = abs(fm - fu) / abs(fu), norm(gm - gu) / norm(gu)
        println("MUMPS $label design $design: objective=$df gradient=$dg seconds=$t")
        @test df < 1e-10
        @test dg < 1e-10
    end
end

if Base.get_extension(DEO, :MUMPSExt) === nothing
    @info "MUMPS.jl is not available; skipping the MUMPSSolver gradient tests"
else
    @testset "MUMPSSolver gradients" begin
        fc = FieldConfig(532; tilt_x=0, pol=:y)
        normal = angular_test_problem([fc], [fc])
        against_umfpack("normal", normal, MUMPSSolver())
        against_umfpack("normal, symmetric storage", normal, MUMPSSolver(symmetric=true))

        pump = [FieldConfig(532; tilt_y=45, pol=:x, weight=0.5), FieldConfig(532; tilt_y=45, pol=:y, weight=0.5)]
        oblique = angular_test_problem(pump, [FieldConfig(532; tilt_y=-20, pol=:y)])
        against_umfpack("45° pump, -20° emission", oblique, MUMPSSolver())
        mixed = angular_test_problem(pump, [FieldConfig(532; tilt_y=0, pol=:y)])
        against_umfpack("45° pump, normal emission", mixed, MUMPSSolver())

        g, errs = angular_ladder(with_solver(oblique, MUMPSSolver()))
        println("MUMPS 45° s+p pump, -20° p emission: g=$g ladder=$errs")
        @test minimum(errs) < 1e-4
        @test errs[end] < 1e-4
    end
end
GC.gc()

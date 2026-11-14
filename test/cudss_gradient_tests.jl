# CUDSSSolver (NVIDIA cuDSS through CUDSS.jl) in the optimization loop, the same checks as the
# PardisoSolver oblique tests: adjoint gradients against central differences for a 45° pump
# (s and p, weight 0.5 each) with an oblique emission, and objectives and gradients against
# UMFPACK at 45°, for a 45° pump with a normal emission, and at normal incidence. The structure
# must be "G" for tilted fields and "S" for normal ones, including the normal emission of a
# tilted pump. The shortcut adjoint conj(A⁻¹ conj b) on the general factor must fail the 45°
# ladder. With precision = :single, objectives and gradients still match UMFPACK to 1e-9. Skipped where CUDSS.jl or a CUDA GPU is not available, unless DEO_REQUIRE_CUDSS=1.
include("oblique_common.jl")
Base.find_package("CUDSS") !== nothing && @eval using CUDSS

cudss_gpu_ready() = Base.get_extension(DEO, :CUDSSExt) !== nothing && CUDSS.CUDACore.functional()
if isdefined(@__MODULE__, :CUDSS) && CUDSS.CUDACore.functional() && Base.get_extension(DEO, :CUDSSExt) === nothing
    error("CUDSS.jl and a CUDA GPU are available, but the CUDSSExt extension did not load")
end

# The shortcut adjoint, kept as a negative control: exact only for complex symmetric matrices.
struct CUDSSShortcutSolver <: DEO.AbstractSolver end
DEO.lu!(::CUDSSShortcutSolver, A) = DEO.lu!(CUDSSSolver(), A)
DEO.lu!(::CUDSSShortcutSolver, f, A) = DEO.lu!(CUDSSSolver(), f, A)
DEO.filter_lu!(::CUDSSShortcutSolver, A) = DEO.filter_lu!(CUDSSSolver(), A)
DEO.filter_lu!(::CUDSSShortcutSolver, f, A) = DEO.filter_lu!(CUDSSSolver(), f, A)
DEO.solve!(::CUDSSShortcutSolver, f, b::Vector) = DEO.solve!(CUDSSSolver(), f, b)
DEO.solve_adjoint!(::CUDSSShortcutSolver, f, b::Vector) = conj(DEO.solve!(CUDSSSolver(), f, conj(b)))
DEO.release_factor!(::CUDSSShortcutSolver, f) = DEO.release_factor!(CUDSSSolver(), f)

cudss_with(prob, solver) = OptimizationProblem(prob.pde, prob.objective, prob.sim, solver;
    foundry_mode=true, control=prob.control)

# Objective and gradient at two random designs (the second on refactorized factors).
function cudss_against_umfpack(label, prob; solver=CUDSSSolver(), tol=1e-10)
    rng = MersenneTwister(1103)
    reference, candidate = cudss_with(prob, UmfpackSolver()), cudss_with(prob, solver)
    for design in 1:2
        p = 0.4 .+ 0.2 .* rand(rng, length(prob.p))
        gu = zeros(length(p)); gc = zeros(length(p))
        tu = @elapsed fu = objective_and_gradient!(gu, p, reference)
        tc = @elapsed fc = objective_and_gradient!(gc, p, candidate)
        df, dg = abs(fc - fu) / abs(fu), norm(gc - gu) / norm(gu)
        println("CUDSS $label design $design: objective=$df gradient=$dg seconds umfpack=$tu cudss=$tc")
        @test df < tol
        @test dg < tol
    end
    return candidate
end

cudss_pools(pool::DEO.SolverCachePool) = [pool]
cudss_pools(bundle::DEO.SolverPoolBundle) = [bundle.default; collect(values(bundle.by_pol))]
cudss_structures(prob) = Dict(k => c.A_factor.structure for pool in cudss_pools(prob.pool)
    for (k, c) in pool.caches if c.A_factor !== nothing)

if !cudss_gpu_ready()
    get(ENV, "DEO_REQUIRE_CUDSS", "0") == "1" &&
        error("DEO_REQUIRE_CUDSS=1, but the CUDSSExt extension is not loaded or no CUDA GPU is functional")
    @info "CUDSS.jl with a CUDA GPU is not available; skipping the CUDSSSolver gradient tests"
else
    @testset "CUDSSSolver gradients" begin
        pump = [FieldConfig(532; tilt_y=45, pol=:x, weight=0.5), FieldConfig(532; tilt_y=45, pol=:y, weight=0.5)]

        # 45° pump, oblique emission: every factor is general.
        oblique = angular_test_problem(pump, [FieldConfig(532; tilt_y=-20, pol=:y)])
        g, errs = angular_ladder(cudss_with(oblique, CUDSSSolver()))
        println("CUDSS 45° s+p pump, -20° p emission: g=$g ladder=$errs")
        @test g > 0 && isfinite(g)
        @test minimum(errs) < 1e-4
        @test errs[end] < 1e-4
        _, bad = angular_ladder(cudss_with(oblique, CUDSSShortcutSolver()))
        println("CUDSS shortcut adjoint (negative control): ladder=$bad")
        @test minimum(bad) > 1e-2
        q = cudss_against_umfpack("oblique emission", oblique)
        s = cudss_structures(q)
        println("CUDSS oblique emission structures: $s")
        @test !isempty(s) && all(==("G"), values(s))

        # 45° pump, normal emission: the emission factor is symmetric.
        normal_emission = [FieldConfig(532; tilt_y=0, pol=:y)]
        mixed = angular_test_problem(pump, normal_emission)
        q = cudss_against_umfpack("normal emission", mixed)
        s = cudss_structures(q)
        println("CUDSS normal emission structures: $s")
        @test s[cache_key(pump[1])] == "G"
        @test s[cache_key(pump[2])] == "G"
        @test s[cache_key(normal_emission[1])] == "S"

        # Normal incidence on the quarter cell: every factor is symmetric.
        fc = FieldConfig(532; tilt_x=0, pol=:y)
        normal = angular_test_problem([fc], [fc])
        q = cudss_against_umfpack("normal", normal)
        s = cudss_structures(q)
        println("CUDSS normal structures: $s")
        @test !isempty(s) && all(==("S"), values(s))

        # precision = :single: single-precision "S" factors refined in double to 1e-11 give the
        # same objective and gradient; tilted ("G") factors stay double.
        q = cudss_against_umfpack("normal, single precision", normal; solver=CUDSSSolver(precision=:single), tol=1e-9)
        @test all(c -> typeof(c.A_factor).parameters[2] === ComplexF32,
                  (c for pool in cudss_pools(q.pool) for c in values(pool.caches) if c.A_factor !== nothing))
        cudss_against_umfpack("normal emission, single precision", mixed; solver=CUDSSSolver(precision=:single), tol=1e-9)
    end
end
GC.gc()

# Forward and adjoint solves on non-symmetric complex matrices for every solver backend that
# loads. The adjoint must be the conjugate-transposed solve Aᴴ x = b: oblique Bloch operators
# are not complex symmetric (A(κ)ᵀ = A(−κ)). A backend that cannot do that solve must refuse
# rather than return conj(A⁻¹ conj b). Each factor serves a forward, an adjoint and a second
# forward solve with distinct right-hand sides, before and after a refactorization: a backend
# that keeps solver state can hand back a right-hand side unsolved. The storage and adjoint
# decisions that live in src are checked whether or not the MUMPS and Pardiso extensions are
# available.
using DistributedEmitterOpt, LinearAlgebra, SparseArrays, Random, Test
const DEO = DistributedEmitterOpt

# MUMPS.jl loads only when it is installed; GridapPardiso is used only if already loaded.
if Base.find_package("MUMPS") !== nothing
    @eval using MUMPS
    MUMPS.MPI.Initialized() || MUMPS.MPI.Init()
end

relres(r, b) = norm(r) / norm(b)

# Forward, adjoint and forward again with distinct right-hand sides: the worst relative residual.
function solve_sequence(s, f, M, rng)
    b1, b2, b3 = (randn(rng, eltype(M), size(M, 1)) for _ in 1:3)
    return max(relres(M * DEO.solve!(s, f, b1) - b1, b1),
               relres(M' * DEO.solve_adjoint!(s, f, b2) - b2, b2),
               relres(M * DEO.solve!(s, f, b3) - b3, b3))
end

@testset "Backend forward and adjoint solves" begin
    A = sparse(ComplexF64[2+1im 1+2im; -0.3+0.7im 3-0.4im])
    b = ComplexF64[1+0.2im, -0.7+0.3im]
    rng = MersenneTwister(1103)
    n = 60
    B = sprandn(rng, ComplexF64, n, n, 0.1) + 6I
    bB = randn(rng, ComplexF64, n)
    S = A + transpose(A)                       # complex symmetric
    mumps = Base.get_extension(DistributedEmitterOpt, :MUMPSExt) !== nothing
    pardiso = Base.get_extension(DistributedEmitterOpt, :PardisoExt) !== nothing
    println("EXTENSIONS MUMPS=$mumps Pardiso=$pardiso")

    backends = Any[("UMFPACK", UmfpackSolver())]
    mumps && push!(backends, ("MUMPS", MUMPSSolver()))
    for (name, s) in backends, (M, rhs) in ((A, b), (B, bB))
        @test norm(M - transpose(M)) > 1e-2 * norm(M)
        f = DEO.lu!(s, M)
        fwd = relres(M * DEO.solve!(s, f, rhs) - rhs, rhs)
        adj = relres(M' * DEO.solve_adjoint!(s, f, rhs) - rhs, rhs)
        seq = solve_sequence(s, f, M, rng)
        # Refactorization on the same sparsity pattern
        M2 = M + spdiagm(0 => fill(0.5 - 0.25im, size(M, 1)))
        f = DEO.lu!(s, f, M2)
        seq2 = solve_sequence(s, f, M2, rng)
        println("BACKEND $name n=$(size(M, 1)) forward=$fwd adjoint=$adj sequence=$seq refactor sequence=$seq2")
        @test max(fwd, adj, seq, seq2) < 1e-11
        DEO.release_factor!(s, f)
    end

    # Real systems (the filter): the adjoint is the transposed solve.
    R = sprandn(rng, n, n, 0.1) + 6I
    bR = randn(rng, n)
    Rs = R + transpose(R)
    for (name, s) in backends
        f = DEO.filter_lu!(s, Rs)
        @test solve_sequence(s, f, Rs, rng) < 1e-11
        f = DEO.filter_lu!(s, f, 2Rs)
        @test solve_sequence(s, f, 2Rs, rng) < 1e-11
        DEO.release_factor!(s, f)
        name == "UMFPACK" || continue
        f = DEO.lu!(s, R)
        @test relres(transpose(R) * DEO.solve_adjoint!(s, f, bR) - bR, bR) < 1e-11
    end

    if mumps
        ext = Base.get_extension(DistributedEmitterOpt, :MUMPSExt)
        s = MUMPSSolver()
        # Symmetric storage (sym = 2): MUMPS gets one triangle of the full matrix.
        @test_throws ArgumentError DEO.lu!(MUMPSSolver(symmetric=true), A)
        s2 = MUMPSSolver(symmetric=true)
        for M in (S, B + transpose(B))
            f = DEO.lu!(s2, M)
            @test f.sym == 2
            @test solve_sequence(s2, f, M, rng) < 1e-11
            @test DEO.lu!(s2, f, 2M) === f         # same pattern: the analysis is reused
            @test solve_sequence(s2, f, 2M, rng) < 1e-11
            DEO.release_factor!(s2, f)
        end
        # The MPI backend, the only one before MUMPS.jl 1.6.1, next to the sequential default
        if ext.HAS_SEQUENTIAL
            f = DEO.lu!(s, B)
            @test f.sequential
            DEO.release_factor!(s, f)
            f = ext.new_factor(s, B, 0; sequential=false)
            @test !f.sequential
            @test solve_sequence(s, f, B, rng) < 1e-11
            @test DEO.lu!(s, f, 2B) === f
            @test solve_sequence(s, f, 2B, rng) < 1e-11
            DEO.release_factor!(s, f)
        end
        # Real unsymmetric storage (sym = 0), as DEO never asks for it: the transposed solve
        f = ext.new_factor(s, R, 0)
        @test solve_sequence(s, f, R, rng) < 1e-11
        DEO.release_factor!(s, f)
        # A new pattern, also one made by changing the factorized matrix in place, gets a new
        # instance, and the old one is released.
        f = DEO.lu!(s, B)
        @test DEO.lu!(s, f, B + spdiagm(0 => fill(0.5 - 0.25im, n))) === f
        C = copy(B)
        f = DEO.lu!(s, C)
        i = findfirst(iszero, Vector(C[:, 1]))
        C[i, 1] = 0.3 - 0.1im
        g = DEO.lu!(s, f, C)
        @test g !== f
        @test_throws ArgumentError DEO.solve!(s, f, bB)
        @test solve_sequence(s, g, C, rng) < 1e-11
        DEO.release_factor!(s, g)
        DEO.release_factor!(s, g)                  # idempotent
        @test_throws ArgumentError DEO.solve_adjoint!(s, g, bB)
        # A failed factorization raises instead of leaving a factor that returns garbage.
        @test_throws ErrorException DEO.lu!(s, sparse(ComplexF64[1 1; 1 1]))
        # Negative control: MUMPS.jl's own solve! on an instance that has solved already returns
        # the new right-hand side unsolved, which is what the sequences above guard against.
        m = MUMPS.Mumps{ComplexF64}(0)
        MUMPS.suppress_display!(m)
        MUMPS.associate_matrix!(m, B)
        MUMPS.factorize!(m)
        MUMPS.associate_rhs!(m, bB)
        MUMPS.solve!(m)
        b2 = 2bB .+ 1
        MUMPS.associate_rhs!(m, b2)
        MUMPS.solve!(m)
        @test vec(MUMPS.get_sol(m)) == b2
        MUMPS.finalize!(m)
    end
    if pardiso
        s = PardisoSolver()
        f = DEO.lu!(s, A)
        @test relres(A * DEO.solve!(s, f, b) - b, b) < 1e-11
        @test_throws ArgumentError DEO.solve_adjoint!(s, f, b)
        f = DEO.lu!(s, S)
        @test relres(S' * DEO.solve_adjoint!(s, f, b) - b, b) < 1e-11
    end

    # Decisions in src, independent of the extensions
    @test DEO.pardiso_shortcut_adjoint_ok(S)
    @test !DEO.pardiso_shortcut_adjoint_ok(A)
    @test DEO.check_pardiso_adjoint(true) === nothing
    @test_throws ArgumentError DEO.check_pardiso_adjoint(false)
    @test DEO.checked_mumps_sym(false, A) == 0
    @test DEO.checked_mumps_sym(true, S) == 2
    @test_throws ArgumentError DEO.checked_mumps_sym(true, A)
end
GC.gc()

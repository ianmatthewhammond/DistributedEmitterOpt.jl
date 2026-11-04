# Forward and adjoint solves on non-symmetric complex matrices for every solver backend that
# loads. The adjoint must be the conjugate-transposed solve Aᴴ x = b: oblique Bloch operators
# are not complex symmetric (A(κ)ᵀ = A(−κ)), and on these matrices the complex-symmetric
# shortcut conj(A⁻¹ conj b) must miss it. Each factor serves a forward, an adjoint and a second
# forward solve with distinct right-hand sides, before and after a refactorization: a backend
# that keeps solver state can hand back a right-hand side unsolved. The storage decisions that
# live in src are checked whether or not the MUMPS and Pardiso extensions are available.
using DistributedEmitterOpt, LinearAlgebra, SparseArrays, Random, Test
const DEO = DistributedEmitterOpt

# MUMPS.jl and Pardiso.jl load only when they are installed; Pardiso also needs MKL.
if Base.find_package("MUMPS") !== nothing
    @eval using MUMPS
    MUMPS.MPI.Initialized() || MUMPS.MPI.Init()
end
Base.find_package("Pardiso") !== nothing && @eval import Pardiso

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
    pardiso = Base.get_extension(DistributedEmitterOpt, :PardisoExt) !== nothing && Pardiso.mkl_is_available()
    println("EXTENSIONS MUMPS=$mumps Pardiso=$pardiso")

    backends = Any[("UMFPACK", UmfpackSolver())]
    mumps && push!(backends, ("MUMPS", MUMPSSolver()))
    pardiso && push!(backends, ("Pardiso", PardisoSolver()))
    for (name, s) in backends, (M, rhs) in ((A, b), (B, bB))
        @test norm(M - transpose(M)) > 1e-2 * norm(M)
        f = DEO.lu!(s, M)
        fwd = relres(M * DEO.solve!(s, f, rhs) - rhs, rhs)
        adj = relres(M' * DEO.solve_adjoint!(s, f, rhs) - rhs, rhs)
        shortcut = relres(M' * conj(DEO.solve!(s, f, conj(rhs))) - rhs, rhs)
        seq = solve_sequence(s, f, M, rng)
        # Refactorization on the same sparsity pattern
        M2 = M + spdiagm(0 => fill(0.5 - 0.25im, size(M, 1)))
        f = DEO.lu!(s, f, M2)
        seq2 = solve_sequence(s, f, M2, rng)
        println("BACKEND $name n=$(size(M, 1)) forward=$fwd adjoint=$adj sequence=$seq refactor sequence=$seq2 shortcut=$shortcut")
        @test max(fwd, adj, seq, seq2) < 1e-11
        @test shortcut > 1e-2
        DEO.release_factor!(s, f)
    end

    if pardiso
        s = PardisoSolver()
        # The analysis is reused on the same pattern. A new pattern, also one made by changing
        # the factorized matrix in place, gets a new analysis, and the old handle is released.
        f = DEO.lu!(s, B)
        @test DEO.lu!(s, f, B + spdiagm(0 => fill(0.5 - 0.25im, n))) === f
        C = copy(B)
        f = DEO.lu!(s, C)
        i = findfirst(iszero, Vector(C[:, 1]))
        @test i !== nothing
        C[i, 1] = 0.3 - 0.1im                  # a new stored entry
        g = DEO.lu!(s, f, C)
        @test g !== f
        @test_throws ArgumentError DEO.solve!(s, f, bB)
        @test solve_sequence(s, g, C, rng) < 1e-11
        DEO.release_factor!(s, g)
        DEO.release_factor!(s, g)              # idempotent
        @test_throws ArgumentError DEO.solve_adjoint!(s, g, bB)
        h = DEO.lu!(s, g, B)                   # a released factor is rebuilt
        @test h !== g
        @test relres(B * DEO.solve!(s, h, bB) - bB, bB) < 1e-11
        DEO.release_factor!(s, h)

        # A complex symmetric matrix gets complex symmetric storage (mtype 6, one triangle) by
        # default and when forced; a non-symmetric one gets mtype 13 unless 6 is forced.
        @test DEO.lu!(s, B).mtype == 13
        @test DEO.lu!(PardisoSolver(assume_symmetric_maxwell=false), S).mtype == 13
        for s6 in (PardisoSolver(), PardisoSolver(assume_symmetric_maxwell=true)),
            M in (S, B + transpose(B))
            f = DEO.lu!(s6, M)
            @test f.mtype == 6
            @test solve_sequence(s6, f, M, rng) < 1e-11
            DEO.release_factor!(s6, f)
        end
        @test_throws ArgumentError DEO.lu!(PardisoSolver(assume_symmetric_maxwell=true), B)
        @test_throws ArgumentError DEO.lu!(PardisoSolver(mtype=4), sparse(ComplexF64[2 1+im; 1-im 3]))
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
    if pardiso
        for (mtype, M) in ((2, Rs + 30I), (11, R))
            s = PardisoSolver(filter_mtype=mtype)
            f = DEO.filter_lu!(s, M)
            @test solve_sequence(s, f, M, rng) < 1e-11
            @test DEO.filter_lu!(s, f, 2M) === f
            @test solve_sequence(s, f, 2M, rng) < 1e-11
            DEO.release_factor!(s, f)
        end
        @test_throws ArgumentError DEO.filter_lu!(PardisoSolver(), R)
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

    # Decisions in src, independent of the extensions
    @test DEO.symmetry_defect(B) ≈ norm(B - transpose(B)) / norm(B)   # also one-sided entries
    @test DEO.symmetry_defect(S) == 0
    @test DEO.symmetry_defect(Rs) == 0
    @test DEO.checked_maxwell_mtype(PardisoSolver(), B + transpose(B)) == 6
    @test DEO.checked_maxwell_mtype(PardisoSolver(), B) == 13
    @test DEO.checked_mumps_sym(false, A) == 0
    @test DEO.checked_mumps_sym(true, S) == 2
    @test_throws ArgumentError DEO.checked_mumps_sym(true, A)
    @test DEO.checked_filter_mtype(PardisoSolver(), Rs) == -2
    @test DEO.checked_filter_mtype(PardisoSolver(filter_mtype=11), R) == 11
    @test_throws ArgumentError DEO.checked_filter_mtype(PardisoSolver(), R)
    @test_throws ArgumentError DEO.checked_filter_mtype(PardisoSolver(filter_mtype=1), Rs)
    @test DEO.checked_pardiso_iparm(DEO.legacy_pardiso_iparm()) == DEO.legacy_pardiso_iparm()
    @test_throws ArgumentError DEO.checked_pardiso_iparm(zeros(Int, 63))
    for (i, v) in ((1, 0), (6, 1), (28, 1), (35, 1))
        iparm = DEO.legacy_pardiso_iparm()
        iparm[i] = v
        @test_throws ArgumentError DEO.checked_pardiso_iparm(iparm)
    end
end
GC.gc()   # collected Pardiso handles run Pardiso.jl's finalizers, also after release_factor!

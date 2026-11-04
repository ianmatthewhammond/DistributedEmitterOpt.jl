# Forward and adjoint solves on non-symmetric complex matrices for every solver backend that
# loads. The adjoint must be the conjugate-transposed solve Aᴴ x = b: oblique Bloch operators
# are not complex symmetric (A(κ)ᵀ = A(−κ)), and on these matrices the complex-symmetric
# shortcut conj(A⁻¹ conj b) must miss it. The storage decisions that live in src are checked
# whether or not the MUMPS and Pardiso extensions are available.
using DistributedEmitterOpt, LinearAlgebra, SparseArrays, Random, Test
const DEO = DistributedEmitterOpt

# MUMPS.jl and Pardiso.jl load only when they are installed; Pardiso also needs MKL.
if Base.find_package("MUMPS") !== nothing
    @eval using MUMPS
    MUMPS.MPI.Initialized() || MUMPS.MPI.Init()
end
Base.find_package("Pardiso") !== nothing && @eval import Pardiso

relres(r, b) = norm(r) / norm(b)

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
        # Refactorization on the same sparsity pattern
        M2 = M + spdiagm(0 => fill(0.5 - 0.25im, size(M, 1)))
        f = DEO.lu!(s, f, M2)
        fwd2 = relres(M2 * DEO.solve!(s, f, rhs) - rhs, rhs)
        adj2 = relres(M2' * DEO.solve_adjoint!(s, f, rhs) - rhs, rhs)
        println("BACKEND $name n=$(size(M, 1)) forward=$fwd adjoint=$adj refactor forward=$fwd2 adjoint=$adj2 shortcut=$shortcut")
        @test max(fwd, adj, fwd2, adj2) < 1e-11
        @test shortcut > 1e-2
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
        @test relres(C * DEO.solve!(s, g, bB) - bB, bB) < 1e-11
        @test relres(C' * DEO.solve_adjoint!(s, g, bB) - bB, bB) < 1e-11
        DEO.release_factor!(s, g)
        DEO.release_factor!(s, g)              # idempotent
        @test_throws ArgumentError DEO.solve_adjoint!(s, g, bB)
        h = DEO.lu!(s, g, B)                   # a released factor is rebuilt
        @test h !== g
        @test relres(B * DEO.solve!(s, h, bB) - bB, bB) < 1e-11
        DEO.release_factor!(s, h)

        # Complex symmetric storage (mtype 6) keeps one triangle.
        s6 = PardisoSolver(assume_symmetric_maxwell=true)
        for (M, rhs) in ((S, b), (B + transpose(B), bB))
            f = DEO.lu!(s6, M)
            @test relres(M * DEO.solve!(s6, f, rhs) - rhs, rhs) < 1e-11
            @test relres(M' * DEO.solve_adjoint!(s6, f, rhs) - rhs, rhs) < 1e-11
            DEO.release_factor!(s6, f)
        end
        @test_throws ArgumentError DEO.lu!(s6, B)
        @test_throws ArgumentError DEO.lu!(PardisoSolver(mtype=4), sparse(ComplexF64[2 1+im; 1-im 3]))
    end

    # Real systems (the filter): the adjoint is the transposed solve.
    R = sprandn(rng, n, n, 0.1) + 6I
    bR = randn(rng, n)
    Rs = R + transpose(R)
    for (name, s) in backends
        f = DEO.filter_lu!(s, Rs)
        @test relres(Rs * DEO.solve!(s, f, bR) - bR, bR) < 1e-11
        @test relres(Rs * DEO.solve_adjoint!(s, f, bR) - bR, bR) < 1e-11
        name == "UMFPACK" || continue
        f = DEO.lu!(s, R)
        @test relres(transpose(R) * DEO.solve_adjoint!(s, f, bR) - bR, bR) < 1e-11
    end
    if pardiso
        for (mtype, M) in ((2, Rs + 30I), (11, R))
            s = PardisoSolver(filter_mtype=mtype)
            f = DEO.filter_lu!(s, M)
            @test relres(M * DEO.solve!(s, f, bR) - bR, bR) < 1e-11
            @test relres(transpose(M) * DEO.solve_adjoint!(s, f, bR) - bR, bR) < 1e-11
            @test DEO.filter_lu!(s, f, 2M) === f
            @test relres(2M * DEO.solve!(s, f, bR) - bR, bR) < 1e-11
            DEO.release_factor!(s, f)
        end
        @test_throws ArgumentError DEO.filter_lu!(PardisoSolver(), R)
    end

    # Decisions in src, independent of the extensions
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

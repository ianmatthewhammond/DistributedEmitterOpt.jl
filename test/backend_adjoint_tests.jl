# Forward and adjoint solves on non-symmetric complex matrices for every solver backend that
# loads. The adjoint must be the conjugate-transposed solve Aᴴ x = b: oblique Bloch operators
# are not complex symmetric (A(κ)ᵀ = A(−κ)). A backend that cannot do that solve must refuse
# rather than return conj(A⁻¹ conj b). The storage and adjoint decisions that live in src are
# checked whether or not the MUMPS and Pardiso extensions are available.
using DistributedEmitterOpt, LinearAlgebra, SparseArrays, Random, Test
const DEO = DistributedEmitterOpt

# MUMPS.jl loads only when it is installed; GridapPardiso is used only if already loaded.
if Base.find_package("MUMPS") !== nothing
    @eval using MUMPS
    MUMPS.MPI.Initialized() || MUMPS.MPI.Init()
end

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
    pardiso = Base.get_extension(DistributedEmitterOpt, :PardisoExt) !== nothing
    println("EXTENSIONS MUMPS=$mumps Pardiso=$pardiso")

    backends = Any[("UMFPACK", UmfpackSolver())]
    mumps && push!(backends, ("MUMPS", MUMPSSolver()))
    for (name, s) in backends, (M, rhs) in ((A, b), (B, bB))
        @test norm(M - transpose(M)) > 1e-2 * norm(M)
        f = DEO.lu!(s, M)
        fwd = relres(M * DEO.solve!(s, f, rhs) - rhs, rhs)
        adj = relres(M' * DEO.solve_adjoint!(s, f, rhs) - rhs, rhs)
        # Refactorization on the same sparsity pattern
        M2 = M + spdiagm(0 => fill(0.5 - 0.25im, size(M, 1)))
        f = DEO.lu!(s, f, M2)
        fwd2 = relres(M2 * DEO.solve!(s, f, rhs) - rhs, rhs)
        adj2 = relres(M2' * DEO.solve_adjoint!(s, f, rhs) - rhs, rhs)
        println("BACKEND $name n=$(size(M, 1)) forward=$fwd adjoint=$adj refactor forward=$fwd2 adjoint=$adj2")
        @test max(fwd, adj, fwd2, adj2) < 1e-11
    end

    # Real non-symmetric system: the adjoint is the transposed solve.
    R = sprandn(rng, n, n, 0.1) + 6I
    bR = randn(rng, n)
    for (name, s) in backends
        f = DEO.filter_lu!(s, R + transpose(R))
        @test relres((R + transpose(R)) * DEO.solve_adjoint!(s, f, bR) - bR, bR) < 1e-11
        name == "UMFPACK" || continue
        f = DEO.lu!(s, R)
        @test relres(transpose(R) * DEO.solve_adjoint!(s, f, bR) - bR, bR) < 1e-11
    end

    if mumps
        @test_throws ArgumentError DEO.lu!(MUMPSSolver(symmetric=true), A)
        f = DEO.lu!(MUMPSSolver(symmetric=true), S)
        @test relres(S' * DEO.solve_adjoint!(MUMPSSolver(), f, b) - b, b) < 1e-11
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

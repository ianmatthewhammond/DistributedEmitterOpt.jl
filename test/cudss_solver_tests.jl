# CUDSSSolver (NVIDIA cuDSS through CUDSS.jl) on small matrices: the structure chosen for each
# matrix ("S" for complex symmetric, "G" otherwise, forced choices and their refusal), forward
# and adjoint solves against dense references (in both orders, so no solve mode leaks), a matrix
# accepted as symmetric only to the tolerance (its adjoint must still be exact), analysis reuse
# on new values, a new analysis on a new pattern, real filter matrices and their refactor, a
# symmetric indefinite matrix with zero diagonal, a released factor, and the residual guard.
# Skipped where CUDSS.jl or a CUDA GPU is not available, unless DEO_REQUIRE_CUDSS=1, which makes
# a missing extension or GPU a failure.
using Test, LinearAlgebra, SparseArrays, Random
using DistributedEmitterOpt
const DEO = DistributedEmitterOpt
Base.find_package("CUDSS") !== nothing && @eval using CUDSS

cudss_ready() = Base.get_extension(DEO, :CUDSSExt) !== nothing && CUDSS.CUDACore.functional()
cudss_required() = get(ENV, "DEO_REQUIRE_CUDSS", "0") == "1"

# A sparse, diagonally dominant complex matrix; `symmetric` makes it A = Aᵀ (not Hermitian).
function cudss_test_matrix(rng, n; symmetric::Bool, T=ComplexF64)
    S = sprand(rng, T, n, n, 4 / n)
    A = symmetric ? S + transpose(S) : S + 0.3 * sprand(rng, T, n, n, 2 / n)
    return SparseMatrixCSC{T,Int}(A + (8 + (T <: Complex ? 2im : 0)) * I)
end

cudss_relerr(x, y) = norm(x - y) / norm(y)

if !cudss_ready()
    cudss_required() && error("DEO_REQUIRE_CUDSS=1, but the CUDSSExt extension is not loaded or no CUDA GPU is functional")
    @info "CUDSS.jl with a CUDA GPU is not available; skipping the CUDSSSolver tests"
else
    isdefined(CUDSS.CUDACore, :allowscalar) && CUDSS.CUDACore.allowscalar(false)
    @testset "CUDSSSolver on small matrices" begin
        rng = MersenneTwister(2026)
        n = 400
        solver = CUDSSSolver()

        # Complex symmetric (normal incidence): LDLᵀ on the lower triangle, adjoint via conj.
        A = cudss_test_matrix(rng, n; symmetric=true)
        @test A == transpose(A) && A != A'
        b = rand(rng, ComplexF64, n)
        f = DEO.lu!(solver, A)
        @test f.structure == "S"
        @test f.npivots isa Int
        @test cudss_relerr(DEO.solve!(solver, f, b), Matrix(A) \ b) < 1e-12
        @test cudss_relerr(DEO.solve_adjoint!(solver, f, b), Matrix(A)' \ b) < 1e-12

        # New values on the same pattern reuse the analysis (same factor object).
        A2 = copy(A); nonzeros(A2) .*= 1 .+ 0.1 .* rand(rng, length(nonzeros(A2)))
        A2 = (A2 + transpose(A2)) / 2
        @test A2.colptr == A.colptr && A2.rowval == A.rowval
        f2 = DEO.lu!(solver, f, A2)
        @test f2 === f
        @test cudss_relerr(DEO.solve!(solver, f2, b), Matrix(A2) \ b) < 1e-12
        @test cudss_relerr(DEO.solve_adjoint!(solver, f2, b), Matrix(A2)' \ b) < 1e-12

        # A new pattern with the same structure gets a new analysis (a new factor).
        A3 = cudss_test_matrix(rng, n; symmetric=true)
        f3 = DEO.lu!(solver, f2, A3)
        @test f3 !== f2 && f3.structure == "S" && f2.released
        @test cudss_relerr(DEO.solve!(solver, f3, b), Matrix(A3) \ b) < 1e-12

        # Not symmetric (oblique Bloch fields): general LDU, conjugate-transpose adjoint. Adjoint
        # before forward and forward again after, so a solve mode cannot leak between calls.
        B = cudss_test_matrix(rng, n; symmetric=false)
        @test DEO.symmetry_defect(B) > 1e-3
        g = DEO.lu!(solver, f3, B)
        @test g !== f3 && g.structure == "G"
        @test cudss_relerr(DEO.solve_adjoint!(solver, g, b), Matrix(B)' \ b) < 1e-12
        @test cudss_relerr(DEO.solve!(solver, g, b), Matrix(B) \ b) < 1e-12
        @test cudss_relerr(DEO.solve_adjoint!(solver, g, b), Matrix(B)' \ b) < 1e-12
        B2 = copy(B); nonzeros(B2) .*= 1 .+ 0.1 .* rand(rng, length(nonzeros(B2)))
        g2 = DEO.lu!(solver, g, B2)
        @test g2 === g
        @test cudss_relerr(DEO.solve_adjoint!(solver, g2, b), Matrix(B2)' \ b) < 1e-12
        @test cudss_relerr(DEO.solve!(solver, g2, b), Matrix(B2) \ b) < 1e-12

        # Symmetric only to the tolerance: "S" is chosen, and the adjoint must still solve the
        # true Aᴴ (the refinement runs on the residual of Aᴴ, not of the stored symmetric part).
        δ = 1e-13
        N = sparse(ComplexF64[1 0 0; 0 δ δ; 0 0 δ])
        @test DEO.symmetry_defect(N) <= DEO.SYMMETRY_RTOL
        e = ComplexF64[0, 1, 0]
        fn = DEO.lu!(solver, N)
        @test fn.structure == "S"
        @test cudss_relerr(DEO.solve!(solver, fn, e), Matrix(N) \ e) < 1e-10
        @test cudss_relerr(DEO.solve_adjoint!(solver, fn, e), Matrix(N)' \ e) < 1e-10

        # Forced choices: "S" is refused for a matrix that is not symmetric; "G" is allowed for
        # a symmetric one and gives the same answers.
        @test_throws ArgumentError DEO.lu!(CUDSSSolver(assume_symmetric_maxwell=true), B)
        h = DEO.lu!(CUDSSSolver(assume_symmetric_maxwell=false), A)
        @test h.structure == "G"
        @test cudss_relerr(DEO.solve_adjoint!(solver, h, b), Matrix(A)' \ b) < 1e-12

        # Symmetric indefinite with a zero diagonal: either an accurate answer or an error,
        # never a silent wrong one.
        Z = spzeros(ComplexF64, n, n)
        for k in 1:2:n-1
            Z[k, k+1] = Z[k+1, k] = 2 + 1im
        end
        Z += 0.01 * (cudss_test_matrix(rng, n; symmetric=true) - (8 + 2im) * I)
        z = rand(rng, ComplexF64, n)
        outcome = try
            fz = DEO.lu!(solver, Z)
            (cudss_relerr(DEO.solve!(solver, fz, z), Matrix(Z) \ z),
             cudss_relerr(DEO.solve_adjoint!(solver, fz, z), Matrix(Z)' \ z))
        catch err
            err isa ErrorException || rethrow()
            println("CUDSS zero-diagonal indefinite case refused: ", sprint(showerror, err))
            :refused
        end
        println("CUDSS zero-diagonal indefinite case: ", outcome)
        @test outcome === :refused || maximum(outcome) < 1e-10

        # Real filter matrices: symmetric → "S", otherwise "G" with a transpose solve; refactor.
        F = real(cudss_test_matrix(rng, n; symmetric=true, T=Float64))
        c = rand(rng, n)
        ff = DEO.filter_lu!(solver, F)
        @test ff.structure == "S"
        @test cudss_relerr(DEO.solve!(solver, ff, c), Matrix(F) \ c) < 1e-12
        @test cudss_relerr(DEO.solve_adjoint!(solver, ff, c), Matrix(F)' \ c) < 1e-12
        F2 = copy(F); nonzeros(F2) .*= 1 .+ 0.1 .* rand(rng, length(nonzeros(F2)))
        F2 = (F2 + transpose(F2)) / 2
        ff2 = DEO.filter_lu!(solver, ff, F2)
        @test ff2 === ff
        @test cudss_relerr(DEO.solve!(solver, ff2, c), Matrix(F2) \ c) < 1e-12
        G = real(cudss_test_matrix(rng, n; symmetric=false, T=Float64))
        fg = DEO.filter_lu!(solver, G)
        @test fg.structure == "G"
        @test cudss_relerr(DEO.solve_adjoint!(solver, fg, c), Matrix(G)' \ c) < 1e-12
        @test cudss_relerr(DEO.solve!(solver, fg, c), Matrix(G) \ c) < 1e-12

        # A released factor refuses to solve; refactoring it builds a new one.
        DEO.release_factor!(solver, g2)
        @test g2.released && g2.solver === nothing
        @test_throws ArgumentError DEO.solve!(solver, g2, b)
        g3 = DEO.lu!(solver, g2, B2)
        @test g3 !== g2 && !g3.released
        @test cudss_relerr(DEO.solve!(solver, g3, b), Matrix(B2) \ b) < 1e-12

        # The residual check is on: a tolerance below roundoff with no refinement must throw.
        strict = CUDSSSolver(residual_rtol=1e-30, refine_steps=0)
        fs = DEO.lu!(strict, A)
        @test_throws ErrorException DEO.solve!(strict, fs, b)
        @test_throws ErrorException DEO.solve_adjoint!(strict, fs, b)
    end
end
GC.gc()

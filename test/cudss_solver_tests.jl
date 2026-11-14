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
# CUDSS.jl installed and a GPU working, but no extension: a broken load, never a skip.
if isdefined(@__MODULE__, :CUDSS) && CUDSS.CUDACore.functional() && Base.get_extension(DEO, :CUDSSExt) === nothing
    error("CUDSS.jl and a CUDA GPU are available, but the CUDSSExt extension did not load")
end

# A sparse, diagonally dominant complex matrix; `symmetric` makes it A = Aᵀ (not Hermitian).
function cudss_test_matrix(rng, n; symmetric::Bool, T=ComplexF64)
    S = sprand(rng, T, n, n, 4 / n)
    A = symmetric ? S + transpose(S) : S + 0.3 * sprand(rng, T, n, n, 2 / n)
    return SparseMatrixCSC{T,Int}(A + (8 + (T <: Complex ? 2im : 0)) * I)
end

cudss_relerr(x, y) = norm(x - y) / norm(y)

# The device arrays of the CSR matrix a factor handed to cuDSS.
cudss_arrays(f) = (pointer(f.stored.rowPtr), pointer(f.stored.colVal), pointer(f.stored.nzVal))

# Three factors that are registered while held (checked under GC.@preserve) and dropped without
# release when this returns; @noinline so the caller's frame never holds them.
@noinline function cudss_dropped_factors(solver, A, ext)
    fs = [DEO.lu!(solver, A) for _ in 1:3]
    GC.@preserve fs begin
        ids = [f.id for f in fs]
        registered = all(id -> haskey(ext.LIVE, id), ids)
    end
    return ids, registered
end

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
        @test f.npivots == -1 || f.npivots >= 0
        println("CUDSS npivots (symmetric test matrix): ", f.npivots)
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

        # Many refactorizations with freshly uploaded values, each solved at once: cuDSS must run on
        # the stream that uploaded the values (a factor of half-uploaded values would miss the
        # residual check, and the solve would throw).
        Ak = copy(A2)
        for k in 1:20
            nonzeros(Ak) .= nonzeros(A2) .* (1 + 0.05k)
            @test DEO.lu!(solver, f2, Ak) === f2
            @test cudss_relerr(DEO.solve!(solver, f2, b), Matrix(Ak) \ b) < 1e-12
        end
        DEO.lu!(solver, f2, A2)

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

        # A refactorization writes the new values into the arrays cuDSS was analysed with, which stay
        # the same allocations ("S" and "G"), even with the GC run and freed device memory handed back
        # to the driver between refactorizations.
        fs2 = DEO.lu!(solver, A2)
        for (fk, Mk) in ((g2, B2), (fs2, A2))
            ptrs = cudss_arrays(fk)
            Ak = copy(Mk)
            for k in 1:10
                nonzeros(Ak) .= nonzeros(Mk) .* (1 + 0.05k)
                GC.gc(true)
                isdefined(CUDSS.CUDACore, :reclaim) && CUDSS.CUDACore.reclaim()
                @test DEO.lu!(solver, fk, Ak) === fk
                @test cudss_arrays(fk) == ptrs
                @test cudss_relerr(DEO.solve!(solver, fk, b), Matrix(Ak) \ b) < 1e-12
                @test cudss_relerr(DEO.solve_adjoint!(solver, fk, b), Matrix(Ak)' \ b) < 1e-12
            end
            @test DEO.lu!(solver, fk, Mk) === fk
        end
        DEO.release_factor!(solver, fs2)

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

        # Invalid options are refused, not silently turned into "no check".
        @test_throws ArgumentError DEO.lu!(CUDSSSolver(residual_rtol=NaN), A)
        @test_throws ArgumentError DEO.lu!(CUDSSSolver(residual_rtol=-1.0), A)
        @test_throws ArgumentError DEO.lu!(CUDSSSolver(refine_steps=-1), A)
        @test_throws ArgumentError DEO.filter_lu!(CUDSSSolver(residual_rtol=Inf), F)

        # Ownership: a factor built in a task that has finished (and been collected) still works,
        # and factors used from several tasks at once give correct answers.
        ext = Base.get_extension(DEO, :CUDSSExt)
        ft = fetch(Threads.@spawn DEO.lu!(solver, A2))
        GC.gc(true); GC.gc(true)
        @test cudss_relerr(DEO.solve!(solver, ft, b), Matrix(A2) \ b) < 1e-12
        @test cudss_relerr(DEO.solve_adjoint!(solver, ft, b), Matrix(A2)' \ b) < 1e-12
        work = [Threads.@spawn begin
                    M = isodd(k) ? A2 : B2
                    fk = DEO.lu!(solver, M)
                    e1 = cudss_relerr(DEO.solve!(solver, fk, b), Matrix(M) \ b)
                    e2 = cudss_relerr(DEO.solve_adjoint!(solver, fk, b), Matrix(M)' \ b)
                    DEO.release_factor!(solver, fk)
                    max(e1, e2)
                end for k in 1:4]
        @test maximum(fetch.(work)) < 1e-12

        # A factor analysed in one task and then refactorized and solved, forward and adjoint, from
        # another: every operation must run on the extension's stream, whichever task calls it.
        fx = fetch(Threads.@spawn DEO.lu!(solver, A2))
        Ax = copy(A2)
        for k in 1:20
            nonzeros(Ax) .= nonzeros(A2) .* (1 + 0.03k)
            @test DEO.lu!(solver, fx, Ax) === fx
            @test cudss_relerr(DEO.solve!(solver, fx, b), Matrix(Ax) \ b) < 1e-12
            @test cudss_relerr(DEO.solve_adjoint!(solver, fx, b), Matrix(Ax)' \ b) < 1e-12
        end
        DEO.release_factor!(solver, fx)

        # Rectangular matrices are refused before any structure is chosen or anything is allocated.
        R = sprand(rng, ComplexF64, n, n + 1, 4 / n)
        @test_throws DimensionMismatch DEO.lu!(CUDSSSolver(assume_symmetric_maxwell=false), R)
        @test_throws DimensionMismatch DEO.lu!(solver, R)
        @test_throws DimensionMismatch DEO.filter_lu!(solver, real(R))

        # Lifetimes: release removes the factor from the registry at once; factors dropped without
        # release are destroyed, under the lock, by a later call (here release_factor! itself) once
        # the GC has collected them. The dropped factors are made in a function, so nothing here
        # keeps them alive.
        @test haskey(ext.LIVE, ft.id)
        DEO.release_factor!(solver, ft)
        @test !haskey(ext.LIVE, ft.id) && ft.solver === nothing && ft.stored === nothing
        ids, registered = cudss_dropped_factors(solver, A2, ext)
        @test registered
        gone = false
        for attempt in 1:5
            GC.gc(true)
            DEO.release_factor!(solver, DEO.lu!(solver, A2))    # drains, then releases its own
            gone = !any(id -> haskey(ext.LIVE, id), ids)
            gone && break
        end
        println("CUDSS dropped factors destroyed: ", gone)
        @test gone

        # A release whose cleanup cannot start keeps the objects registered and queues them; the
        # next call into the extension destroys them.
        fr = DEO.lu!(solver, A2)
        ext.CLEANUP_FAULT[] = true
        try
            DEO.release_factor!(solver, fr)
        finally
            ext.CLEANUP_FAULT[] = false
        end
        @test fr.released && fr.solver === nothing
        @test haskey(ext.LIVE, fr.id)
        @test fr.id in lock(() -> copy(ext.DROPPED), ext.DROPPED_LOCK)
        DEO.release_factor!(solver, fr)                           # released already: a no-op that drains
        @test !haskey(ext.LIVE, fr.id)
        @test !(fr.id in lock(() -> copy(ext.DROPPED), ext.DROPPED_LOCK))
        @test_throws ArgumentError DEO.solve!(solver, fr, b)

        # Options are checked at solve time too, and right-hand sides must match the factor (on a
        # live factor, so neither check can pass on the released-factor error instead).
        live = DEO.lu!(solver, A2)
        @test !live.released
        @test_throws ArgumentError DEO.solve!(CUDSSSolver(residual_rtol=NaN), live, b)
        @test_throws ArgumentError DEO.solve_adjoint!(CUDSSSolver(refine_steps=-1), live, b)
        @test_throws DimensionMismatch DEO.solve!(solver, live, b[1:end-1])
        @test_throws DimensionMismatch DEO.solve_adjoint!(solver, live, zeros(ComplexF64, n + 1))
        @test cudss_relerr(DEO.solve!(solver, live, b), Matrix(A2) \ b) < 1e-12   # still usable
        DEO.release_factor!(solver, live)

        # The residual check is on: a tolerance below roundoff with no refinement must throw.
        strict = CUDSSSolver(residual_rtol=1e-30, refine_steps=0)
        fs = DEO.lu!(strict, A)
        @test_throws ErrorException DEO.solve!(strict, fs, b)
        @test_throws ErrorException DEO.solve_adjoint!(strict, fs, b)
    end
end
GC.gc()

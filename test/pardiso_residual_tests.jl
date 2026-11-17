# PardisoSolver's residual check (`residual_rtol`, 1e-9 by default, 0 = off). After every
# solve, forward or adjoint, Maxwell (mtype 13 and 6) or filter (mtype -2 and 11), the answer
# must satisfy ‖b − Op x‖ ≤ residual_rtol ‖b‖ for the operator that solve inverts. Checked here:
# the residual the backend computes against an independent product, for each operator; that an
# accepted answer and the handle are left bit for bit as they were, and that a solve with the
# check off gives the same bytes; that a wrong answer (all zeros, as Intel acknowledged for
# iparm(8) = 0 on oneMKL 2025.0, or NaN) is solved again with refinement on and comes out
# right, with iparm(8) and iparm(12) put back; that a limit no solve can meet ends in an error
# naming both residuals and leaves the factor usable; a zero right-hand side; a reused analysis
# (the check follows the new values); and the refusal of an invalid tolerance.
# The keyword and its validation are checked whether or not Pardiso.jl is installed; the rest
# is skipped where Pardiso.jl or MKL is not available.
using DistributedEmitterOpt, LinearAlgebra, SparseArrays, Random, Test
const DEO = DistributedEmitterOpt
Base.find_package("Pardiso") !== nothing && @eval import Pardiso

# All 64 iparm entries of a handle.
residual_iparm(f) = [Int(Pardiso.get_iparm(f.ps, i)) for i in 1:64]
residual_counts(f) = (f.residual_checks, f.residual_resolves)

@testset "PardisoSolver residual_rtol" begin
    d = PardisoSolver()
    @test d.residual_rtol === 1e-9
    @test PardisoSolver(residual_rtol=0).residual_rtol === 0.0
    @test DEO.checked_pardiso_residual_rtol(d) === 1e-9
    @test DEO.checked_pardiso_residual_rtol(PardisoSolver(residual_rtol=0)) === 0.0
    for bad in (-1e-9, NaN, Inf)
        @test_throws ArgumentError DEO.checked_pardiso_residual_rtol(PardisoSolver(residual_rtol=bad))
    end

    ext = Base.get_extension(DEO, :PardisoExt)
    if !(ext !== nothing && Pardiso.mkl_is_available())
        @info "Pardiso.jl with MKL is not available; skipping the PardisoSolver residual tests"
    else
        # The second solve asks for twice the steps the handle allows, at least two, and keeps
        # the handle's choice of extended precision (negative values).
        @test ext.resolve_steps.((0, 1, 2, 3, -1, -2, -3)) == (2, 2, 4, 6, -2, -4, -6)

        rng = MersenneTwister(1117)
        n = 400
        B = sprandn(rng, ComplexF64, n, n, 0.02) + 6I          # not symmetric: mtype 13
        S = B + transpose(B)                                   # complex symmetric: mtype 6
        R = sprandn(rng, n, n, 0.02)
        Rs = R + transpose(R) + 12I                            # real symmetric: filter mtype -2
        Ru = R + 12I                                           # real, not symmetric: filter mtype 11
        bc = randn(rng, ComplexF64, n)
        br = randn(rng, n)
        off = PardisoSolver(residual_rtol=0)
        tight = PardisoSolver(residual_rtol=1e-30)             # a limit no solve meets
        cases = (("Maxwell mtype 13", d, DEO.lu!, B, bc, 13),
                 ("Maxwell mtype 6", d, DEO.lu!, S, bc, 6),
                 ("filter mtype -2", d, DEO.filter_lu!, Rs, br, -2),
                 ("filter mtype 11", PardisoSolver(filter_mtype=11), DEO.filter_lu!, Ru, br, 11))

        # A stored lower triangle stands for the symmetric matrix, also where a diagonal entry
        # is missing from the matrix and stored as an explicit zero.
        let U = triu(sprandn(rng, ComplexF64, 60, 60, 0.1), 1), v = randn(rng, ComplexF64, 60)
            Z = U + transpose(U) + sparse([3, 17], [3, 17], ComplexF64[2 - im, 0.5im], 60, 60)
            L = ext.storage(Z, 6)
            @test nnz(L) > nnz(tril(Z))                        # zeros were added on the diagonal
            y = ext.symmetric_mul!(similar(v), L, v)
            @test norm(y - Z * v) <= 1e-13 * norm(Z * v)
        end

        for (name, s, factorize, M, b, mtype) in cases
            T = eltype(b)
            f = factorize(s, M)
            @test f.mtype == mtype
            @test residual_counts(f) == (0, 0) && isnan(f.last_residual)
            fwd = ext.forward_mode(mtype)
            held = Int(Pardiso.get_iparm(f.ps, 8))
            # The system a solve with iparm(12) = mode inverts. For the symmetric types both
            # are the matrix itself.
            systems = ((fwd, M), (0, copy(transpose(M))))

            # The backend's residual against an independent product, at a wrong answer.
            xw = randn(rng, T, n)
            for (mode, Op) in systems
                r = similar(b)
                want = b - Op * xw
                got = ext.relative_residual!(r, f, xw, b, mode)
                @test abs(got - norm(want) / norm(b)) <= 1e-12 * got
                @test norm(r - want) <= 1e-12 * norm(want)
            end
            @test ext.relative_residual!(similar(b), f, zeros(T, n), b, fwd) == 1.0   # an all-zero answer
            @test ext.relative_residual!(similar(b), f, zeros(T, n), zeros(T, n), fwd) == 0.0
            @test ext.relative_residual!(similar(b), f, xw, zeros(T, n), fwd) == Inf
            @test isnan(ext.relative_residual!(similar(b), f, fill(T(NaN), n), b, fwd))

            # Ordinary solves are checked, accepted and counted.
            x = DEO.solve!(s, f, b)
            @test residual_counts(f) == (1, 0)
            rf = norm(M * x - b) / norm(b)
            @test rf <= 1e-11 && abs(f.last_residual - rf) <= 1e-13
            y = DEO.solve_adjoint!(s, f, b)
            @test residual_counts(f) == (2, 0)
            ra = norm(M' * y - b) / norm(b)
            @test ra <= 1e-11 && abs(f.last_residual - ra) <= 1e-13
            @test Pardiso.get_iparm(f.ps, 8) == held && Pardiso.get_iparm(f.ps, 12) == fwd

            # The check only reads. With it off, the same factor gives the same bytes and counts
            # nothing; an accepted answer and the handle stay bit for bit as they were.
            x0 = DEO.solve!(off, f, b)
            y0 = DEO.solve_adjoint!(off, f, b)
            @test residual_counts(f) == (2, 0)
            @test isequal(x, x0) && isequal(y, y0)
            ip = residual_iparm(f)
            xc = copy(x0)
            @test ext.accept_or_refine!(f, x0, b, fwd, 1e-9) === x0
            @test isequal(x0, xc)
            @test residual_iparm(f) == ip
            @test residual_counts(f) == (3, 0)
            println("PARDISO residual $name: iparm(8)=$held forward=$rf adjoint=$ra; check off on the same factor: identical=$(isequal(x, x0) && isequal(y, y0))")

            # What a second solve should give: a solve with iparm(8) = resolve_steps(held) on a
            # factorization of its own, per system, with the steps it reports (iparm(7)).
            again = ext.resolve_steps(held)
            g = factorize(PardisoSolver(filter_mtype=s.filter_mtype, refinement_steps=again, residual_rtol=0), M)
            @test Pardiso.get_iparm(g.ps, 8) == again
            expected = Dict{Int,Any}()
            for (mode, _) in systems
                z = ext.raw_solve!(similar(b), g, b, mode)
                expected[mode] = (z, Int(Pardiso.get_iparm(g.ps, 7)))
            end
            DEO.release_factor!(s, g)

            # A wrong first answer fails the check and is solved again with refinement on: the
            # result is right and equals that reference, the steps are those the reference took,
            # and iparm(8) and iparm(12) are put back.
            for (mode, Op) in systems, (kind, wrong) in (("zero", zeros(T, n)), ("NaN", fill(T(NaN), n)))
                before = residual_counts(f)
                z = ext.accept_or_refine!(f, wrong, b, mode, 1e-9)
                done = Int(Pardiso.get_iparm(f.ps, 7))
                @test z === wrong
                @test norm(Op * z - b) / norm(b) <= 1e-11
                @test residual_counts(f) == before .+ (1, 1)
                @test f.last_residual <= 1e-11
                @test done == expected[mode][2]
                @test norm(z - expected[mode][1]) <= 1e-12 * norm(z)
                @test Pardiso.get_iparm(f.ps, 8) == held && Pardiso.get_iparm(f.ps, 12) == fwd
                println("PARDISO residual $name: $kind answer, iparm(12)=$mode: second solve with iparm(8)=$again took $done steps, residual=$(f.last_residual), identical to the reference=$(isequal(z, expected[mode][1]))")
            end

            # A limit no solve meets: second solve, then an error naming both residuals. The
            # handle is put back and the factor still serves.
            before = residual_counts(f)
            err = try
                DEO.solve!(tight, f, b)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            if err isa ErrorException
                @test occursin("residual_rtol=1.0e-30", err.msg)
                @test occursin("iparm(8)=$again", err.msg)
                @test occursin("after a second solve", err.msg)
            end
            @test_throws ErrorException DEO.solve_adjoint!(tight, f, b)
            @test residual_counts(f) == before .+ (2, 2)
            @test Pardiso.get_iparm(f.ps, 8) == held && Pardiso.get_iparm(f.ps, 12) == fwd
            @test !f.released
            @test isequal(DEO.solve!(s, f, b), x)

            # A zero right-hand side has the zero answer and a zero residual.
            before = residual_counts(f)
            z = DEO.solve!(s, f, zeros(T, n))
            @test all(iszero, z) && f.last_residual == 0.0
            @test residual_counts(f) == before .+ (1, 0)

            # Same pattern, new values: the check follows the matrix now factorized.
            @test factorize(s, f, 2M) === f
            x2 = DEO.solve!(s, f, b)
            @test norm(2x2 - x) <= 1e-10 * norm(x)
            @test residual_counts(f) == before .+ (2, 0) && f.last_residual <= 1e-11

            # An invalid tolerance is refused when a factor is built, reused or solved on.
            invalid(v) = PardisoSolver(filter_mtype=s.filter_mtype, residual_rtol=v)
            @test_throws ArgumentError factorize(invalid(-1.0), M)
            @test_throws ArgumentError factorize(invalid(Inf), f, M)
            @test_throws ArgumentError DEO.solve!(invalid(NaN), f, b)
            @test_throws ArgumentError DEO.solve_adjoint!(invalid(-1.0), f, b)
            @test !f.released && residual_counts(f) == before .+ (2, 0)
            DEO.release_factor!(s, f)
        end
    end
end

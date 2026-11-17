# PardisoSolver's `refinement_steps`: iparm(8), the iterative refinement of every solve on a
# factorization. With the default `nothing` the handle keeps what it had before the keyword
# existed: the iparm profile's value for a full-storage Maxwell factorization (mtype 13) and
# MKL's own default for the one-triangle types (Maxwell mtype 6, filter mtype -2), which the
# profile does not reach. A given value must arrive in the handle for each type, stay in
# force when the analysis is reused, give way to the next solver's setting on a reused
# handle, and leave forward and adjoint solutions equal to the default's within 1e-10.
# A solve through a solver with another setting must not change it, and a handle that
# replaces one (changed pattern, `reuse_symbolic = false`) must hold the value asked for.
# What iparm(8) = 0 means depends on the MKL release: from oneMKL 2025.0 it turns refinement
# off; before 2025.0 it ran two steps after perturbed pivots. The assertions hold for both,
# and the MKL release is printed. The residual check of every solve (`residual_rtol`, on by
# default) has its own file, pardiso_residual_tests.jl; here it runs and accepts every answer.
# The keyword and the defaults are checked whether or not Pardiso.jl is installed; the rest
# is skipped where Pardiso.jl or MKL is not available.
using DistributedEmitterOpt, LinearAlgebra, SparseArrays, Random, Test
const DEO = DistributedEmitterOpt
Base.find_package("Pardiso") !== nothing && @eval import Pardiso

# Worst relative residual of a forward and an adjoint solve, the two solutions, and the
# refinement steps each solve reports (iparm(7), read right after it).
function refinement_solves(s, f, M, b)
    x = DEO.solve!(s, f, b)
    forward_steps = Int(Pardiso.get_iparm(f.ps, 7))
    y = DEO.solve_adjoint!(s, f, b)
    adjoint_steps = Int(Pardiso.get_iparm(f.ps, 7))
    return max(norm(M * x - b), norm(M' * y - b)) / norm(b), x, y, (forward_steps, adjoint_steps)
end

# The steps the solves took stay within the limit iparm(8) set. An MKL before 2025.0 refines
# after perturbed pivots independently of iparm(8), so the limit is asked for only without them.
# Each limit that was really compared is counted, and the testset ends by asking that every
# limit was compared at least once: a run in which every factorization perturbed a pivot would
# otherwise pass without having checked a single bound.
const STEP_LIMITS_COMPARED = Dict{Int,Int}()
function steps_within(f, steps_done, limit)
    Pardiso.get_iparm(f.ps, 14) > 0 && return true
    STEP_LIMITS_COMPARED[limit] = get(STEP_LIMITS_COMPARED, limit, 0) + 1
    return all(k -> abs(k) <= abs(limit), steps_done)
end

# The MKL release in use, for the log only.
function mkl_version_string()
    try
        buf = zeros(UInt8, 200)
        Base.Libc.Libdl.dlopen(Pardiso.libmkl_rt) do lib
            ccall(Base.Libc.Libdl.dlsym(lib, :MKL_Get_Version_String), Cvoid, (Ptr{UInt8}, Cint), buf, length(buf))
        end
        return String(rstrip(String(buf), ['\0', ' ']))
    catch err
        return "not read ($(typeof(err)))"
    end
end

# iparm(8) of a handle that has seen `pardisoinit` only: MKL's default for that matrix type.
function mkl_default_refinement(mtype)
    ps = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(ps, mtype)
    Pardiso.pardisoinit(ps)
    return Int(Pardiso.get_iparm(ps, 8))
end

@testset "PardisoSolver refinement_steps" begin
    d = PardisoSolver()
    @test d.refinement_steps === nothing
    @test (d.mtype, d.mtype_symmetric, d.filter_mtype, d.assume_symmetric_maxwell,
           d.reuse_symbolic, d.msglvl, d.nprocs) == (13, 6, -2, nothing, true, 0, 0)
    @test d.iparm == DEO.legacy_pardiso_iparm()
    @test PardisoSolver(refinement_steps=-2).refinement_steps == -2

    if !(Base.get_extension(DEO, :PardisoExt) !== nothing && Pardiso.mkl_is_available())
        @info "Pardiso.jl with MKL is not available; skipping the PardisoSolver refinement tests"
    else
        println("PARDISO refinement: MKL = ", mkl_version_string())
        rng = MersenneTwister(1116)
        n = 400
        B = sprandn(rng, ComplexF64, n, n, 0.02) + 6I          # not symmetric: mtype 13
        S = B + transpose(B)                                   # complex symmetric: mtype 6
        R = sprandn(rng, n, n, 0.02)
        Rs = R + transpose(R) + 12I                            # real symmetric: filter mtype -2
        bc = randn(rng, ComplexF64, n)
        br = randn(rng, n)
        cases = (("Maxwell mtype 13", DEO.lu!, B, bc, 13),
                 ("Maxwell mtype 6", DEO.lu!, S, bc, 6),
                 ("filter mtype -2", DEO.filter_lu!, Rs, br, -2))

        for (name, factorize, M, b, mtype) in cases
            # Default: the handle holds what it held before the keyword existed.
            f0 = factorize(d, M)
            @test f0.mtype == mtype
            default = Int(Pardiso.get_iparm(f0.ps, 8))
            @test f0.refinement_default == default
            @test default == (mtype == 13 ? DEO.legacy_pardiso_iparm()[8] : mkl_default_refinement(mtype))
            @test Pardiso.get_iparm(f0.ps, 1) == 1             # MKL reads the array it is given
            res0, x0, y0, done0 = refinement_solves(d, f0, M, b)
            println("PARDISO refinement $name: default iparm(8)=$default residual=$res0 steps done=$done0 perturbed pivots=$(Pardiso.get_iparm(f0.ps, 14))")
            @test res0 < 1e-11
            @test steps_within(f0, done0, default)

            for steps in (0, 2, -2)
                s = PardisoSolver(refinement_steps=steps)
                f = factorize(s, M)
                @test f.mtype == mtype
                @test Pardiso.get_iparm(f.ps, 8) == steps
                @test Pardiso.get_iparm(f.ps, 1) == 1
                @test f.refinement_default == default
                res, x, y, done = refinement_solves(s, f, M, b)
                dx = norm(x - x0) / norm(x0)
                dy = norm(y - y0) / norm(y0)
                println("PARDISO refinement $name: refinement_steps=$steps residual=$res steps done=$done solution vs default: forward=$dx adjoint=$dy")
                @test res < 1e-11
                @test max(dx, dy) < 1e-10
                @test Pardiso.get_iparm(f.ps, 8) == steps      # the solves leave it in place
                # Never more steps than allowed, on either solve; with 0, none. An MKL before
                # 2025.0 is held to this only if the factorization perturbed no pivot.
                @test steps_within(f, done, steps)
                # Same pattern, new values: the analysis is reused and the setting stays.
                M2 = 2M
                @test factorize(s, f, M2) === f
                @test Pardiso.get_iparm(f.ps, 8) == steps
                @test refinement_solves(s, f, M2, b)[1] < 1e-11
                DEO.release_factor!(s, f)
            end

            # The default given by number makes the same handle settings as `nothing`.
            fd = factorize(PardisoSolver(refinement_steps=default), M)
            inputs = (1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 21, 24, 25, 27, 28, 35, 60)   # settings MKL reads
            @test [Pardiso.get_iparm(fd.ps, i) for i in inputs] == [Pardiso.get_iparm(f0.ps, i) for i in inputs]
            DEO.release_factor!(d, fd)

            # A reused handle follows the solver that factorizes: a value, back to the handle's
            # own default with `nothing`, and another value.
            f = factorize(PardisoSolver(refinement_steps=2), M)
            @test Pardiso.get_iparm(f.ps, 8) == 2
            @test factorize(d, f, 2M) === f
            @test Pardiso.get_iparm(f.ps, 8) == default
            _, x, y = refinement_solves(d, f, 2M, b)
            @test max(norm(2x - x0) / norm(x0), norm(2y - y0) / norm(y0)) < 1e-10
            @test factorize(PardisoSolver(refinement_steps=-2), f, M) === f
            @test Pardiso.get_iparm(f.ps, 8) == -2
            DEO.release_factor!(d, f)

            # Solving through a solver with another setting leaves the factor's in place.
            s1 = PardisoSolver(refinement_steps=1)
            f = factorize(s1, M)
            res1, _, _, done1 = refinement_solves(PardisoSolver(refinement_steps=(default == 0 ? 2 : 0)), f, M, b)
            @test res1 < 1e-11
            @test Pardiso.get_iparm(f.ps, 8) == 1
            @test steps_within(f, done1, 1)

            # A changed pattern, and `reuse_symbolic = false`, replace the handle: the old one
            # is released, and the new one holds the value asked for and the same default.
            c = eltype(M)(0.01)
            Mp = M + spdiagm(1 => fill(c, n - 1), -1 => fill(c, n - 1))
            g = factorize(s1, f, Mp)
            @test g !== f && f.released
            @test g.mtype == mtype
            @test Pardiso.get_iparm(g.ps, 8) == 1 && g.refinement_default == default
            @test refinement_solves(s1, g, Mp, b)[1] < 1e-11
            h = factorize(PardisoSolver(refinement_steps=-1, reuse_symbolic=false), g, Mp)
            @test h !== g && g.released
            @test Pardiso.get_iparm(h.ps, 8) == -1 && h.refinement_default == default
            @test refinement_solves(d, h, Mp, b)[1] < 1e-11
            DEO.release_factor!(d, h)
            DEO.release_factor!(d, f0)
        end

        # The full-storage default is the iparm profile, entry for entry, with the transposed
        # solve selected; `iparm = nothing` and a profile with another iparm(8) are respected.
        f = DEO.lu!(d, B)
        profile = DEO.legacy_pardiso_iparm()
        profile[12] = 2
        @test [Pardiso.get_iparm(f.ps, i) for i in (1, 2, 8, 10, 11, 12, 13)] == profile[[1, 2, 8, 10, 11, 12, 13]]
        DEO.release_factor!(d, f)
        custom = DEO.legacy_pardiso_iparm()
        custom[8] = 3
        f = DEO.lu!(PardisoSolver(iparm=custom), B)
        @test Pardiso.get_iparm(f.ps, 8) == 3 && f.refinement_default == 3
        @test DEO.lu!(PardisoSolver(iparm=custom, refinement_steps=1), f, B) === f
        @test Pardiso.get_iparm(f.ps, 8) == 1
        DEO.release_factor!(d, f)
        f = DEO.lu!(PardisoSolver(iparm=nothing), B)
        @test Pardiso.get_iparm(f.ps, 8) == mkl_default_refinement(13)
        DEO.release_factor!(d, f)

        # Every step limit asked for above was compared on a factorization without perturbed pivots.
        println("PARDISO refinement: step limits compared without perturbed pivots (limit => times): ", sort(collect(STEP_LIMITS_COMPARED)))
        @test all(limit -> get(STEP_LIMITS_COMPARED, limit, 0) > 0, (0, 1, 2, -2))
    end
end

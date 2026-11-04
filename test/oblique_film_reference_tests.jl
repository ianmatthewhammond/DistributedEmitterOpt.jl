# Uniform film on a substrate at oblique incidence: FEM against an analytic transfer-matrix
# reference (oblique_film_reference.jl).
#
# The normal wavenumber is FITTED from the field, not assumed: in each uniform zone the periodic
# envelope is fitted to a*exp(+iqz) + b*exp(-iqz) with q free. A wrong Bloch shift, a wrong
# boundary admittance or a wrong source normalization each shows up as its own wrong number
# (q, the boundary return, or the incident amplitude), not only as a reflectance error.
# The complex reflected tangential vector is compared, so a two-axis tilt exercises the
# off-diagonal boundary admittance and the cross-polarized reflection.
using DistributedEmitterOpt, Gridap, LinearAlgebra, SparseArrays, Printf, Test
const DEO = DistributedEmitterOpt
BLAS.set_num_threads(2)
include(joinpath(@__DIR__, "oblique_film_reference.jl")); using .ObliqueTMM
function fit2(z, U, qlo, qhi)   # U: npts x ncomp, common q
    res(q) = begin
        M = hcat(exp.(im * q .* z), exp.(-im * q .* z)); C = M \ U
        norm(M * C - U) / norm(U), C
    end
    qs = range(qlo, qhi; length=2001); best = argmin(first.(res.(qs)))
    lo, hi = qs[max(best - 1, 1)], qs[min(best + 1, end)]
    for _ in 1:60
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        res(m1)[1] < res(m2)[1] ? (hi = m2) : (lo = m1)
    end
    q = (lo + hi) / 2; r, C = res(q); q, C, r
end

function main(; λ=532.0, h=20.0, order=1, Lc=120.0, hd=80.0)
    geo = SymmetricGeometry(λ; L=Lc, W=Lc, hd=hd, hsub=220.0)
    geo.hair = 520.0; geo.hs = 330.0; geo.ht = 150.0; geo.l1 = h; geo.l2 = h; geo.l3 = h
    mesh = joinpath(mktempdir(), "film.msh")
    lo, hi = genmesh(geo, mesh; per_x=true, per_y=true)
    sim = build_simulation(mesh; order, dir_x=false, dir_y=false, foundry_mode=false, degree=4)
    zs, zt, zb = hi + geo.hs, hi + geo.hair, lo - geo.hsub
    k = 2π / λ
    dq = Measure(sim.Ω, 2); pts = get_cell_points(dq)
    zc = reduce(vcat, [[p[3] for p in c] for c in get_array(pts)])
    pt = interpolate(x -> 1.0, sim.Pf)
    println("nV=$(sim.nV) order=$order h=$h cell=$Lc")
    println("film   tilt_x tilt_y pol | q_air/k (exact) | q_sub/k (exact) | |E inc| | max|r_fem - r_ref| (tangential, both comps) | R fem / ref | topRet  botRet | resid")
    worst = 0.0
    @test ObliqueTMM.selfcheck(verbose=false)
    for (matname, mat) in (("n=2.0", 2.0), ("Ag", "Ag")), (tx, ty) in ((30.0, 0.0), (0.0, 30.0), (20.0, 15.0), (0.0, 0.0)), pol in (:x, :y)
        matname == "Ag" && !((tx, ty) in ((0.0, 30.0), (20.0, 15.0))) && continue
        fc = FieldConfig(λ; tilt_x=tx, tilt_y=ty, pol)
        env = Environment(mat_design=mat, mat_substrate=1.77, mat_fluid=1.0)
        n1 = DEO.resolve_index(mat, λ)
        phys = DEO.build_phys_params(fc, env, sim)
        A = assemble_maxwell(pt, sim, phys); b = assemble_source(sim, phys)
        uh = FEFunction(sim.U, A \ b)
        vals = uh(pts)
        U = hcat(reduce(vcat, [[v[1] for v in c] for c in vals]), reduce(vcat, [[v[2] for v in c] for c in vals]))
        zone(a, bb) = (m = (zc .> a + 12) .& (zc .< bb - 12); (zc[m], U[m, :]))
        s = (sind(tx), sind(ty)); kp = k * hypot(s...)
        q0 = real(ObliqueTMM.qz(k, 1.0, kp)); q2 = real(ObliqueTMM.qz(k, 1.77, kp))
        qm, Cm, rm = fit2(zone(hi, zs)..., 0.3k, 2.5k)
        qu, Cu, ru = fit2(zone(zs, zt)..., 0.3k, 2.5k)
        qb, Cb, rb = fit2(zone(zb, lo)..., 0.3k, 3.5k)
        # mid zone rows: 1 = exp(+iqz) incident (down), 2 = exp(-iqz) reflected. Refer both to z = hi.
        Ei = Cm[1, :] .* exp(im * qm * hi); Er = Cm[2, :] .* exp(-im * qm * hi)
        Er_ref, _ = ObliqueTMM.reflect_tangential((Ei[1], Ei[2]), k, s, 1.0, n1, hd, 1.77)
        dr = hypot(abs(Er[1] - Er_ref[1]), abs(Er[2] - Er_ref[2])) / norm(Ei)
        Rf = sum(abs2, Er) / sum(abs2, Ei); Rr = (abs2(Er_ref[1]) + abs2(Er_ref[2])) / sum(abs2, Ei)
        # full |E| of the incident wave from its tangential part: Ez = -(s.Et)/cos
        cz = sqrt(1 - s[1]^2 - s[2]^2)
        Einc = sqrt(sum(abs2, Ei) + abs2((s[1] * Ei[1] + s[2] * Ei[2]) / cz))
        topret = sum(abs2, Cu[1, :]) / sum(abs2, Cu[2, :]); botret = sum(abs2, Cb[2, :]) / sum(abs2, Cb[1, :])
        worst = max(worst, dr)
        @test abs(qm - q0) / k < 5e-4
        @test abs(qb - q2) / k < 5e-4
        @test abs(Einc - 0.5) < 1e-3          # half-unit incident amplitude, both polarizations
        @test dr < 1e-3
        @test topret < 1e-6 && botret < 1e-6  # specular order leaves through both boundaries
        @test max(rm, ru, rb) < 1e-2
        @printf("%-6s %5.1f %5.1f  %s  | %.4f (%.4f) | %.4f (%.4f) | %.4f | %.2e | %.5f / %.5f | %.1e %.1e | %.1e\n",
            matname, tx, ty, pol, qm / k, q0 / k, qb / k, q2 / k, Einc, dr, Rf, Rr, topret, botret, max(rm, ru, rb))
        flush(stdout)
    end
    println("WORST complex tangential reflection error (relative to incident): ", worst)
end
@testset "Oblique film against transfer-matrix reference" begin
    main()
end

# Air on a lossy metal substrate (ε = -15.1 - 2.43i, n - ik), at 0° and 45°, s and p. The
# substrate is only 30 nm thick, so the field at the bottom face is about a quarter of its
# surface value: the bottom specular admittance must take the decaying q branch of the lossy
# medium, or the reflection moves. The substrate field decays, so only the air zone is fitted.
# Keep at least six air elements across the periodic cell: a coarser periodic mesh can merge
# distinct edges and carry no propagating field.
function lossy_substrate(; λ=532.0, order=1, Lc=60.0)
    nsub = sqrt(complex(-15.1, -2.43)); real(nsub) < 0 && (nsub = -nsub)
    geo = SymmetricGeometry(λ; L=Lc, W=Lc, hd=20.0, hsub=30.0)
    geo.hair = 520.0; geo.hs = 330.0; geo.ht = 150.0; geo.l1 = 10.0; geo.l2 = 4.0; geo.l3 = 5.0
    mesh = joinpath(mktempdir(), "metal.msh")
    lo, hi = genmesh(geo, mesh; per_x=true, per_y=true)
    sim = build_simulation(mesh; order, dir_x=false, dir_y=false, foundry_mode=false, degree=4)
    zs, zt = hi + geo.hs, hi + geo.hair
    k = 2π / λ
    dq = Measure(sim.Ω, 2); pts = get_cell_points(dq)
    zc = reduce(vcat, [[p[3] for p in c] for c in get_array(pts)])
    pt = interpolate(x -> 0.0, sim.Pf)        # empty design region: air down to the metal
    env = Environment(mat_design=1.0, mat_substrate=1.0, mat_fluid=1.0)
    println("lossy substrate n=$nsub nV=$(sim.nV) order=$order cell=$Lc bottom-face field ≈ $(round(exp(imag(ObliqueTMM.qz(k, nsub, 0.0)) * geo.hsub); digits=2))")
    for tilt in (0.0, 45.0), pol in (:x, :y)
        fc = FieldConfig(λ; tilt_y=tilt, pol)
        p0 = DEO.build_phys_params(fc, env, sim)
        phys = DEO.PhysicalParams(p0.ω, p0.θ, p0.nf, p0.nm, nsub, p0.μ, p0.des_low, p0.des_high, p0.α,
            p0.bot_PEC, p0.pml, p0.tilt_x, p0.n_incident, p0.polarization, p0.formulation)
        A = assemble_maxwell(pt, sim, phys); b = assemble_source(sim, phys)
        uh = FEFunction(sim.U, A \ b)
        vals = uh(pts)
        U = hcat(reduce(vcat, [[v[1] for v in c] for c in vals]), reduce(vcat, [[v[2] for v in c] for c in vals]))
        zone(a, bb) = (m = (zc .> a + 12) .& (zc .< bb - 12); (zc[m], U[m, :]))
        s = (0.0, sind(tilt)); kp = k * hypot(s...)
        q0 = real(ObliqueTMM.qz(k, 1.0, kp))
        qm, Cm, rm = fit2(zone(hi, zs)..., 0.3k, 2.5k)
        qu, Cu, ru = fit2(zone(zs, zt)..., 0.3k, 2.5k)
        Ei = Cm[1, :] .* exp(im * qm * hi); Er = Cm[2, :] .* exp(-im * qm * hi)
        Er_ref, _ = ObliqueTMM.reflect_tangential((Ei[1], Ei[2]), k, s, 1.0, 1.0, geo.hd, nsub)
        dr = hypot(abs(Er[1] - Er_ref[1]), abs(Er[2] - Er_ref[2])) / norm(Ei)
        Rf = sum(abs2, Er) / sum(abs2, Ei); Rr = (abs2(Er_ref[1]) + abs2(Er_ref[2])) / sum(abs2, Ei)
        cz = sqrt(1 - s[1]^2 - s[2]^2)
        Einc = sqrt(sum(abs2, Ei) + abs2((s[1] * Ei[1] + s[2] * Ei[2]) / cz))
        topret = sum(abs2, Cu[1, :]) / sum(abs2, Cu[2, :])
        @test abs(qm - q0) / k < 5e-4
        @test abs(Einc - 0.5) < 1e-3
        @test dr < 1e-3
        @test topret < 1e-6
        @test max(rm, ru) < 1e-2
        @printf("metal  tilt_y %4.1f %s | q_air/k %.5f (%.5f) | |E inc| %.4f | %.2e | R %.5f / %.5f | top %.1e | resid %.1e\n",
            tilt, pol, qm / k, q0 / k, Einc, dr, Rf, Rr, topret, max(rm, ru))
        flush(stdout)
    end
end
@testset "Oblique incidence on a lossy metal substrate against transfer-matrix reference" begin
    lossy_substrate()
end

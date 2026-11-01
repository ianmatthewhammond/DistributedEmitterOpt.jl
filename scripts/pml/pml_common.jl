"""
Shared helpers for the ±z PML validation (scripts/pml/).

Lengths in nm. Time convention exp(+iωt): upward waves e^{-iqz}, downward e^{+iqz}.
The cell is the package's mirror quarter cell [-L/2, L/2] × [-W/2, W/2] with
PEC y-walls and PMC x-walls; for a y-polarized source Ey is even about every
wall, so Ey(x, y, z) = Σ c_mn(z) cos(mπ(x+L/2)/L) cos(nπ(y+W/2)/W) and
the lattice period is 2L × 2W.
"""

using DistributedEmitterOpt
using DistributedEmitterOpt: build_simulation, getdesignz, getpmlz, assemble_maxwell,
    assemble_source, PhysicalParams, ε_background_wf
using Gridap
using LinearAlgebra
using SparseArrays
using Printf

const DEO = DistributedEmitterOpt

# ─────────────────────────────────────────────────────────────────────────────
# Mesh + solve
# ─────────────────────────────────────────────────────────────────────────────

"""Geometry with explicit heights (all above the design top unless noted)."""
function test_geometry(; L, W, hd, hsub, ht, hs, hair, dpml, l1, l2, l3)
    SymmetricGeometry(L, W, hair, hs, ht, hd, hsub, dpml, l1, l2, l3, 0)
end

"""Boundary treatment for one solve."""
abstract type Boundary end
struct ABC <: Boundary end
struct NewPML <: Boundary
    spec::ZPML
end
struct LegacyPML <: Boundary   # Emitter3DTopOpt f1e1514^ form, see legacy_matrix
    R::Float64
    fix_sign::Bool
end

"""
    solve_case(meshfile, bc; λ, nf, nm, ns, pfun, order=1) -> (E, sim, phys)

Solve one field with the package assembly (ABC or NewPML) or the legacy
operator. `pfun(x)` is the design density, sampled at quadrature points.
"""
function solve_case(meshfile, bc::Boundary; λ, nf, nm, ns, pfun, order=1, degree=2 * order + 2)
    sim = build_simulation(meshfile; order, degree, dir_x=false, dir_y=true, foundry_mode=false)
    k0 = 2π / λ
    des_low, des_high = getdesignz(sim.labels, sim.Ω)
    pmlp = nothing
    if bc isa NewPML
        pmlp = ZPMLParams(bc.spec, k0, ns, nf, getpmlz(sim.labels, sim.Ω)...)
    end
    phys = PhysicalParams(k0, 0.0, complex(nf), complex(nm), complex(ns), 1.0,
        des_low, des_high, 0.0, false, pmlp)
    pt = CellField(pfun, sim.Ω_design)
    t = @elapsed begin
        A = bc isa LegacyPML ? legacy_matrix(pt, sim, phys, bc, meshfile) : assemble_maxwell(pt, sim, phys)
        b = assemble_source(sim, phys; source_y=true)
        x = lu(A) \ b
    end
    @printf("  solve: %d dofs, %.1f s (%s)\n", length(b), t, string(typeof(bc).name.name))
    return FEFunction(sim.U, x), sim, phys
end

"""
    mesh_and_solve(geo, meshfile, withpml, bc; kw...) -> (E, sim, phys, geo_used, retries)

genmesh + solve_case. Some genmesh outputs make Gridap's Nédélec space fail
while building it (gmsh leaves orphan interior nodes; DEO's mesh repair then
yields a model whose FE space construction throws a BoundsError in
`_generate_face_to_own_dofs_count_d!`). Only that failure is retried: the mesh
is regenerated with the air/substrate sizes shrunk by 3% per retry. Physical
dimensions are never changed. The geometry actually used is returned.
"""
function mesh_and_solve(geo, meshfile, withpml, bc; kw...)
    g = deepcopy(geo)
    for retry in 0:4
        genmesh(g, meshfile; pml=withpml)
        try
            E, sim, phys = solve_case(meshfile, bc; kw...)
            return E, sim, phys, g, retry
        catch e
            bt = sprint(Base.show_backtrace, catch_backtrace())
            (e isa BoundsError && occursin("_generate_face_to_own_dofs", bt)) || rethrow()
            @printf("  mesh retry %d: Gridap FE space construction failed on %s (l1=%.3f)\n", retry + 1, basename(meshfile), g.l1)
            g.l1 *= 0.97; g.l3 *= 0.97
        end
    end
    error("mesh_and_solve: 5 meshes failed for $meshfile")
end

# ─────────────────────────────────────────────────────────────────────────────
# Legacy operator (Emitter3DTopOpt, src/Physics.jl at f1e1514^), z-only
# ─────────────────────────────────────────────────────────────────────────────

"""
The 2023 3D PML, reproduced term for term for comparison:
  s = 1 + i σ/ω (u/d)²,  σ = -3/4 log(R)/d,  ψ = 1/s
  a = (∇x(ψ,∇ψ,v,∇v)) ⋅ (∇x(ψ,∇u)) - k² ε v⋅u    (no ABC, no Jacobian)
`fix_sign=true` flips only the sign of i σ (the exp(+iωt) choice).
"""
function legacy_matrix(pt, sim, phys, bc::LegacyPML, meshfile)
    k = phys.ω
    z_bot, d_bot, z_top, d_top = getpmlz(sim.labels, sim.Ω)
    d = d_bot
    σ = -3 / 4 * log(bc.R) / d
    sgn = bc.fix_sign ? -1 : 1
    uz(z) = z > z_top ? z - z_top : (z < z_bot ? z_bot - z : 0.0)
    dirz(z) = z > z_top ? 1.0 : (z < z_bot ? -1.0 : 0.0)
    s(z) = 1 + sgn * (1im * σ / k) * (uz(z) / d)^2
    ds(z) = sgn * (2im * σ / k) * (1 / d)^2 * uz(z) * dirz(z)
    ψ = CellField(x -> VectorValue{3,ComplexF64}(1, 1, 1 / s(x[3])), sim.Ω)
    dψ = CellField(x -> TensorValue{3,3,ComplexF64}(0, 0, 0, 0, 0, 0, 0, 0, -ds(x[3]) / s(x[3])^2), sim.Ω)
    curl_u(ψ, ∇u) = VectorValue(ψ[2] * ∇u[2, 3] - ψ[3] * ∇u[3, 2],
        ψ[3] * ∇u[3, 1] - ψ[1] * ∇u[1, 3],
        ψ[1] * ∇u[1, 2] - ψ[2] * ∇u[2, 1])
    curl_v(ψ, dψ, v, ∇v) = VectorValue(
        dψ[2, 2] * v[3] - dψ[3, 3] * v[2] + ψ[2] * ∇v[2, 3] - ψ[3] * ∇v[3, 2],
        dψ[3, 3] * v[1] - dψ[1, 1] * v[3] + ψ[3] * ∇v[3, 1] - ψ[1] * ∇v[1, 3],
        dψ[1, 1] * v[2] - dψ[2, 2] * v[1] + ψ[1] * ∇v[1, 2] - ψ[2] * ∇v[2, 1])
    ε₀ = x -> ε_background_wf(x, phys)
    εₘ = (p -> DEO.ε_design_wf(p, phys)) ∘ pt
    assemble_matrix(sim.U, sim.V) do u, v
        ∫((curl_v ∘ (ψ, dψ, v, ∇(v))) ⋅ (curl_u ∘ (ψ, ∇(u))) - (k^2) * (v ⋅ (ε₀ * u)))sim.dΩ -
        (k^2) * ∫(v ⋅ (εₘ * u))sim.dΩ_design
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Geometry bookkeeping (genmesh coordinates)
# ─────────────────────────────────────────────────────────────────────────────

"""z-planes of the genmesh stack: bottom/top of the physical region, design, source."""
function stack_z(geo)
    z0 = -(geo.hair + geo.hd + geo.hsub) / 2
    des_low = z0 + geo.hsub
    des_high = des_low + geo.hd
    (z_min=z0, des_low=des_low, des_high=des_high, z_src=des_high + geo.hs,
        z_max=des_high + geo.hair)
end

# ─────────────────────────────────────────────────────────────────────────────
# Analytic layered solution (flat stack, normal incidence, Ey only)
# ─────────────────────────────────────────────────────────────────────────────

"""
    flat_exact(z, zs; λ, ns, nslab, nf, z1, z2) -> Ey

Exact solution of -Ey'' - k₀² n(z)² Ey = iω δ(z - zs) (the package's sheet
source, ω = k₀) for substrate (z < z1) | slab (z1..z2) | fluid, outgoing at ±∞.
"""
function flat_exact_coeffs(zs; λ, ns, nslab, nf, z1, z2)
    k0 = 2π / λ
    ks, kd, kf = k0 * ns, k0 * nslab, k0 * nf
    f = im * k0
    e(k, z) = exp(-im * k * z)       # upward
    g(k, z) = exp(+im * k * z)       # downward
    # unknowns: D (sub, down), A, B (slab up/down), C1, C2 (fluid below source), U (above)
    M = zeros(ComplexF64, 6, 6); r = zeros(ComplexF64, 6)
    # z1: E and E' continuous between sub (D g) and slab (A e + B g)
    M[1, :] = [g(ks, z1), -e(kd, z1), -g(kd, z1), 0, 0, 0]
    M[2, :] = [im * ks * g(ks, z1), im * kd * e(kd, z1), -im * kd * g(kd, z1), 0, 0, 0]
    # z2: slab ↔ fluid below source
    M[3, :] = [0, e(kd, z2), g(kd, z2), -e(kf, z2), -g(kf, z2), 0]
    M[4, :] = [0, -im * kd * e(kd, z2), im * kd * g(kd, z2), im * kf * e(kf, z2), -im * kf * g(kf, z2), 0]
    # zs: E continuous, E'(zs+) - E'(zs-) = -f
    M[5, :] = [0, 0, 0, e(kf, zs), g(kf, zs), -e(kf, zs)]
    M[6, :] = [0, 0, 0, im * kf * e(kf, zs), -im * kf * g(kf, zs), -im * kf * e(kf, zs)]
    r[6] = -f
    c = M \ r
    function Ey(z)
        z < z1 && return c[1] * g(ks, z)
        z < z2 && return c[2] * e(kd, z) + c[3] * g(kd, z)
        z < zs && return c[4] * e(kf, z) + c[5] * g(kf, z)
        return c[6] * e(kf, z)
    end
    return Ey
end

# ─────────────────────────────────────────────────────────────────────────────
# Sampling and order decomposition
# ─────────────────────────────────────────────────────────────────────────────

"""
Point evaluation with an explicit bucketed tet locator. Gridap's KDTree
search misses points on these repaired gmsh models (verified: the point is
inside a cell by barycentric test, the search still fails), so cells are
found here directly and the field is evaluated in the physical domain.
"""
struct Locator
    cc::Vector{Vector{VectorValue{3,Float64}}}
    buckets::Dict{NTuple{3,Int},Vector{Int}}
    h::Float64
end

const LOCATORS = IdDict{Any,Locator}()

function Locator(Ω)
    cc = [collect(c) for c in Gridap.Geometry.get_cell_coordinates(Ω)]
    h = 1.5 * maximum(maximum(c[i][d] for i in 1:4) - minimum(c[i][d] for i in 1:4) for c in cc for d in 1:3)
    key(p) = (floor(Int, p[1] / h), floor(Int, p[2] / h), floor(Int, p[3] / h))
    buckets = Dict{NTuple{3,Int},Vector{Int}}()
    for (i, c) in enumerate(cc)
        lo = key(VectorValue(ntuple(d -> minimum(c[k][d] for k in 1:4), 3)))
        hi = key(VectorValue(ntuple(d -> maximum(c[k][d] for k in 1:4), 3)))
        for a in lo[1]:hi[1], b in lo[2]:hi[2], e in lo[3]:hi[3]
            push!(get!(buckets, (a, b, e), Int[]), i)
        end
    end
    Locator(cc, buckets, h)
end

function bary(c, p)
    M = [c[i][d] - c[4][d] for d in 1:3, i in 1:3]
    λ = M \ [p[d] - c[4][d] for d in 1:3]
    min(minimum(λ), 1 - sum(λ))
end

function locate(loc::Locator, p)
    k = (floor(Int, p[1] / loc.h), floor(Int, p[2] / loc.h), floor(Int, p[3] / loc.h))
    best, bestv = 0, -Inf
    for i in get(loc.buckets, k, Int[])
        v = bary(loc.cc[i], p)
        v > bestv && ((best, bestv) = (i, v))
        v >= 0 && return i
    end
    bestv > -1e-8 || error("locate: point $p outside the mesh (best barycentric $bestv)")
    best
end

"""Evaluate a CellField at points (vector of values)."""
function eval_at(E, pts::AbstractVector)
    Ω = get_triangulation(E)
    loc = get!(() -> Locator(Ω), LOCATORS, Ω)
    Ep = Gridap.CellData.change_domain(E, Gridap.CellData.PhysicalDomain())
    cf = Gridap.CellData.get_data(Ep)
    cache = Gridap.Arrays.array_cache(cf)
    map(pts) do p
        f = Gridap.Arrays.getindex!(cache, cf, locate(loc, p))
        Gridap.Arrays.evaluate(f, p)
    end
end

"""Ey on a cell-centred Nx×Ny grid at height z (Nx×Ny matrix)."""
function sample_plane(E, geo, z; Nx, Ny)
    xs = [-geo.L / 2 + (i - 0.5) * geo.L / Nx for i in 1:Nx]
    ys = [-geo.W / 2 + (j - 0.5) * geo.W / Ny for j in 1:Ny]
    pts = [Point(x, y, z) for x in xs, y in ys]
    vals = eval_at(E, vec(pts))
    reshape([v[2] for v in vals], Nx, Ny), xs, ys
end

"""Cosine-series coefficient c_mn of a sampled plane (exact for the midpoint grid)."""
function cos_coeff(F, xs, ys, geo, m, n)
    cx = cos.(m * π .* (xs .+ geo.L / 2) ./ geo.L)
    cy = cos.(n * π .* (ys .+ geo.W / 2) ./ geo.W)
    w = (m == 0 ? 1 : 2) * (n == 0 ? 1 : 2)
    w * sum(F .* (cx * cy')) / length(F)
end

"""Normal wavenumber of order (m,n) in index n_med, branch with Im q ≤ 0."""
function qz(m, n, geo, λ, n_med)
    k = 2π / λ * n_med
    kt2 = (m * π / geo.L)^2 + (n * π / geo.W)^2
    q = sqrt(complex(k^2 - kt2))
    imag(q) > 0 ? -q : q
end

propagating(m, n, geo, λ, n_med) = (m * π / geo.L)^2 + (n * π / geo.W)^2 < (2π / λ * real(n_med))^2

"""
    updown(cz, zs, q) -> (a_up, b_down, resid)

Least-squares fit c(z) = a e^{-iqz} + b e^{+iqz}; amplitudes are referred to
the first plane zs[1].
"""
function updown(cz, zs, q)
    z0 = zs[1]
    M = hcat(exp.(-im * q .* (zs .- z0)), exp.(im * q .* (zs .- z0)))
    ab = M \ cz
    resid = norm(M * ab - cz) / max(norm(cz), 1e-300)
    ab[1], ab[2], resid
end

"""
    order_table(E, geo, λ, nf, ns; Nx, Ny, mmax, nmax, nplanes, margin)

For every propagating order: fit up/down amplitudes in the substrate (below
the design, above the bottom PML/boundary) and in the fluid above the source.
Spurious return = |up|/|down| in the substrate, |down|/|up| above the source.
"""
function order_table(E, geo, λ, nf, ns; Nx, Ny, mmax, nmax, nplanes=12, margin=40.0)
    st = stack_z(geo)
    zsub = collect(range(st.z_min + margin, st.des_low - margin, length=nplanes))
    ztop = collect(range(st.z_src + margin, st.z_max - margin, length=nplanes))
    planes_sub = [sample_plane(E, geo, z; Nx, Ny) for z in zsub]
    planes_top = [sample_plane(E, geo, z; Nx, Ny) for z in ztop]
    rows = NamedTuple[]
    for m in 0:mmax, n in 0:nmax
        for (side, nmed, zs, planes) in (("sub", ns, zsub, planes_sub), ("top", nf, ztop, planes_top))
            propagating(m, n, geo, λ, nmed) || continue
            cz = [cos_coeff(F, xs, ys, geo, m, n) for (F, xs, ys) in planes]
            q = qz(m, n, geo, λ, nmed)
            a, b, res = updown(cz, zs, q)
            out, back = side == "sub" ? (b, a) : (a, b)
            θ = asind(sqrt((m * π / geo.L)^2 + (n * π / geo.W)^2) / (2π / λ * real(nmed)))
            push!(rows, (side=side, m=m, n=n, θ=θ, out=abs(out), back=abs(back),
                ratio=abs(back) / max(abs(out), 1e-300), resid=res,
                out_c=out, back_c=back, q=q))
        end
    end
    rows
end

"""First-order ABC reflection for a TE order at angle θ (prediction for the Ey part)."""
abc_te_reflection(θ) = abs((cosd(θ) - 1) / (cosd(θ) + 1))

"""∫ |E|⁴ over the Raman (target) region: a pump = emission SERS proxy."""
raman_metric(E, sim) = sum(∫(((x -> x^2) ∘ real(E ⋅ conj(E))))sim.dΩ_raman)

function print_orders(rows; label="")
    println("  orders ", label)
    @printf("    %-4s %3s %3s %7s %12s %12s %10s %9s\n", "side", "m", "n", "θ(deg)", "|out|", "|back|", "back/out", "fit-res")
    for r in rows
        @printf("    %-4s %3d %3d %7.2f %12.4e %12.4e %10.3e %9.1e\n", r.side, r.m, r.n, r.θ, r.out, r.back, r.ratio, r.resid)
    end
end

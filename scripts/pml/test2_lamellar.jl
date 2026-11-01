"""
Test 2 — lamellar grating (TE, normal incidence): FEM vs independent RCWA,
with and without higher diffraction orders.

Ridges (n=2.0, h=150) in water (1.33) on silica (1.45), λ = 532. In the
mirror quarter cell the ridge sits on the x = -L/2 wall, so the period is 2L.
  A: P = 780 — m=±1 propagate in water and silica, m=±2 in silica at 70.2°
  B: P = 600 — m=±1 propagate in both
  C: P = 300 — only m = 0 propagates (no diffraction)
For each order the FEM field is split into up/down waves in the substrate
and above the source; efficiencies are compared with RCWA, and the wrong-way
("back") wave in each region measures what the boundary reflects.

Run: julia --project scripts/pml/test2_lamellar.jl [outdir] [h] [nedelec order, default 1]
"""

include(joinpath(@__DIR__, "pml_common.jl"))
using JLD2

outdir = length(ARGS) >= 1 ? ARGS[1] : mktempdir()
hmesh = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 20.0
forder = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
mkpath(outdir)

λ = 532.0
nf, ns, nr = 1.33, 1.45, 2.0
k0 = 2π / λ

# RCWA, N = 641 harmonics, converged to ≤2e-9 (scripts/pml/rcwa/rcwa_v2.py → rcwa_results_v2.json).
# Per single order m (±m equal).
const RCWA = Dict(
    "A" => (P=780.0, R=Dict(0 => 5.68524e-3, 1 => 2.07865e-3), T=Dict(0 => 0.675388, 1 => 0.153256, 2 => 4.12868e-3)),
    "B" => (P=600.0, R=Dict(0 => 9.50397e-3, 1 => 3.73818e-4), T=Dict(0 => 0.715329, 1 => 0.137210)),
    "C" => (P=300.0, R=Dict(0 => 1.30389e-2), T=Dict(0 => 0.986961)),
)

geo_for(P; dpml) = test_geometry(L=P / 2, W=30.0, hd=150.0, hsub=600.0, ht=100.0, hs=600.0,
    hair=1000.0, dpml=dpml, l1=hmesh, l2=hmesh / 2, l3=hmesh)

"""FEM efficiencies per order, and the back-wave ratios."""
function analyse(E, geo; Nx=96)
    st = stack_z(geo)
    out = Dict{String,Any}()
    fit(z1, z2, m, nmed; tag="") = begin
        zs = collect(range(z1, z2, length=14))
        cz = [cos_coeff(sample_plane(E, geo, z; Nx, Ny=1)..., geo, m, 0) for z in zs]
        a, b, res = updown(cz, zs, qz(m, 0, geo, λ, nmed))
        isempty(tag) || (out["fit_$tag"] = Dict("up" => a, "down" => b, "resid" => res, "z" => [z1, z2]))
        a, b, res
    end
    mmax = floor(Int, 2geo.L / λ * ns)
    # incident = downward m=0 wave between grating and source
    zF = (st.des_high + 300.0, st.z_src - 40.0)
    zS = (st.z_min + 40.0, st.des_low - 40.0)
    zT = (st.z_src + 40.0, st.z_max - 40.0)
    up0, inc, _ = fit(zF..., 0, nf; tag="F0")
    for m in 0:mmax
        pair = m == 0 ? 1 : 4        # cos-series amplitude = 2 × single-order amplitude
        if propagating(m, 0, geo, λ, nf)
            a, b, res = fit(zF..., m, nf; tag="F$m")
            q = qz(m, 0, geo, λ, nf)
            out["R$m"] = real(q) / (k0 * nf) * abs2(a) / (pair * abs2(inc))
            m > 0 && (out["backF$m"] = abs(b) / abs(a))    # down-going m≠0 above grating: spurious
            at, bt, _ = fit(zT..., m, nf; tag="T$m")
            out["backT$m"] = abs(bt) / abs(at)
        end
        if propagating(m, 0, geo, λ, ns)
            a, b, res = fit(zS..., m, ns; tag="S$m")
            q = qz(m, 0, geo, λ, ns)
            out["T$m"] = real(q) / (k0 * nf) * abs2(b) / (pair * abs2(inc))
            out["backS$m"] = abs(a) / abs(b)
            out["θS$m"] = asind(m * λ / (2geo.L * ns))
        end
    end
    out["sum"] = sum(v * (k[end] == '0' ? 1 : 2) for (k, v) in out if startswith(k, "R") || startswith(k, "T"))
    out
end

results = Dict{String,Any}()
for name in ("C", "B", "A")
    ref = RCWA[name]
    println("== case $name: P = $(ref.P) nm, h = $hmesh, order $forder")
    res = Dict{String,Any}()
    pfun = x -> (x[1] < -ref.P / 4 + ref.P / 4 ? 1.0 : 0.0)  # ridge w = P/2 on the x = -L/2 wall
    runs = [("ABC", false, 400.0, ABC()),
        ("PML d=400", true, 400.0, NewPML(ZPML(R=1e-8, m=3.0))),
        ("PML d=800", true, 800.0, NewPML(ZPML(R=1e-8, m=3.0)))]
    for (label, withpml, dpml, bc) in runs
        geo0 = geo_for(ref.P; dpml)
        mesh = joinpath(outdir, "lam_$(name)_$(withpml ? "pml$(Int(dpml))" : "abc")_h$(Int(hmesh)).msh")
        E, sim, _, geo, retries = mesh_and_solve(geo0, mesh, withpml, bc; λ, nf, nm=complex(nr), ns, pfun, order=forder)
        a = analyse(E, geo)
        a["mesh_retries"] = retries
        a["geometry_used"] = Dict("L" => geo.L, "W" => geo.W, "l1" => geo.l1, "l2" => geo.l2, "l3" => geo.l3, "dpml" => geo.dpml)
        a["mesh_bytes"] = filesize(mesh)
        st = stack_z(geo)
        zlo, zhi = st.z_min - (withpml ? geo.dpml : 0.0), st.z_max + (withpml ? geo.dpml : 0.0)
        xs = [-geo.L / 2 + (i - 0.5) * geo.L / 64 for i in 1:64]
        zs = collect(range(zlo + 1, zhi - 1, length=360))
        a["slice_x"], a["slice_z"] = xs, zs
        a["slice_Ey"] = reshape([v[2] for v in eval_at(E, vec([Point(x, 0.0, z) for x in xs, z in zs]))], 64, 360)
        a["stack"] = Dict(pairs(st))
        a["raman"] = raman_metric(E, sim)
        res[label] = a
        @printf("  %-10s  ΣR+ΣT = %.6f   ∫|E|⁴ Raman = %.6e\n", label, a["sum"], a["raman"])
        for (kind, d) in (("R", ref.R), ("T", ref.T))
            for m in sort(collect(keys(d)))
                key = "$kind$m"
                haskey(a, key) || continue
                @printf("      %-3s FEM %.5e  RCWA %.5e  rel.err %+.2e\n", key, a[key], d[m], a[key] / d[m] - 1)
            end
        end
        for k in sort([k for k in keys(a) if startswith(k, "back")])
            @printf("      %-7s back/out = %.3e\n", k, a[k])
        end
    end
    results[name] = res
end

jldsave(joinpath(outdir, "test2_lamellar_h$(Int(hmesh))_o$(forder).jld2"); results)
println("saved")

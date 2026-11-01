"""
Test 5 — same mesh, stretch on vs off (no diffraction).

Case C (P = 300, no diffraction): only the normal order propagates, for which
the first-order ABC is exact in the continuum. On the SAME mesh (with PML
volumes), solve
  (a) stretch off: plain medium in the layers, ABC on the outer faces (reference)
  (b) stretch on: the PML
and report the field difference ‖E_b − E_a‖/‖E_a‖ over the physical region,
sweeping R, grading m and thickness. This is a field difference, not a fitted
reflection coefficient, and it does not by itself say what causes a
difference (layer, element dispersion, or the reference's own termination).

Run: julia --project scripts/pml/test5_pml_reflection.jl outdir h order
"""

include(joinpath(@__DIR__, "pml_common.jl"))
using JLD2

outdir = ARGS[1]; hmesh = parse(Float64, ARGS[2]); forder = parse(Int, ARGS[3])
mkpath(outdir)
λ = 532.0; nf, ns, nr = 1.33, 1.45, 2.0
P = 300.0
pfun = x -> (x[1] < 0.0 ? 1.0 : 0.0)
geo_for(dpml) = test_geometry(L=P / 2, W=30.0, hd=150.0, hsub=600.0, ht=100.0, hs=600.0,
    hair=1000.0, dpml=dpml, l1=hmesh, l2=hmesh / 2, l3=hmesh)

results = Dict{String,Any}()
for dpml in (400.0, 800.0)
    mesh = joinpath(outdir, "refl_d$(Int(dpml))_h$(Int(hmesh))_o$(forder).msh")
    Eref, sim, _, geo, retries = mesh_and_solve(geo_for(dpml), mesh, true, ABC(); λ, nf, nm=complex(nr), ns, pfun, order=forder)
    st = stack_z(geo)
    pts = [Point(x, 0.0, z) for x in range(-P / 4 + 3, P / 4 - 3, length=9) for z in range(st.z_min + 5, st.z_max - 5, length=300)]
    vref = [v[2] for v in eval_at(Eref, pts)]
    for R in (1e-3, 1e-5, 1e-8), m in (2.0, 3.0)
        # same mesh, same sim: only the stretch differs
        E, _, _ = solve_case(mesh, NewPML(ZPML(R=R, m=m)); λ, nf, nm=complex(nr), ns, pfun, order=forder)
        v = [u[2] for u in eval_at(E, pts)]
        d = norm(v - vref) / norm(vref)
        @printf("  d=%4d  R=%.0e  m=%.0f   ‖E_pml − E_ref‖/‖E_ref‖ = %.3e\n", dpml, R, m, d)
        results["d$(Int(dpml)) R$(R) m$(Int(m))"] = d
    end
end
jldsave(joinpath(outdir, "test5_refl_h$(Int(hmesh))_o$(forder).jld2"); results)
println("saved")

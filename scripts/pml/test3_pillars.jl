"""
Test 3 — 2D pillar array (full vector: TE, TM and diagonal orders).

Square pillars (n=2.0, side P/2, h=150) in water on silica, λ = 532, y-polarized
normal incidence; in the quarter cell the pillar sits in the (-L/2, -W/2) corner.
  P = 600: orders (1,0), (0,1), (1,1) propagate in both media; (1,1) at 70.5°
           in water and 59.9° in silica. (0,1) carries Ey as TM, (1,0) as TE.
  P = 300: no diffraction.
No RCWA here; the evidence is (a) the wrong-way wave per order, against the
first-order-ABC prediction |1-cosθ|/(1+cosθ) (same for TE and TM), (b) the
Raman near-field metric against PML thickness and R.

Run: julia --project scripts/pml/test3_pillars.jl outdir h P
"""

include(joinpath(@__DIR__, "pml_common.jl"))
using JLD2

outdir = ARGS[1]
hmesh = parse(Float64, ARGS[2])
P = parse(Float64, ARGS[3])
mkpath(outdir)

λ = 532.0
nf, ns, nr = 1.33, 1.45, 2.0

geo_for(; dpml) = test_geometry(L=P / 2, W=P / 2, hd=150.0, hsub=500.0, ht=100.0, hs=550.0,
    hair=900.0, dpml=dpml, l1=hmesh, l2=hmesh / 2, l3=hmesh)
pfun = x -> (x[1] < -P / 4 + P / 4 && x[2] < -P / 4 + P / 4) ? 1.0 : 0.0

runs = [("ABC", false, 400.0, ABC()),
    ("PML d=400 R=1e-8", true, 400.0, NewPML(ZPML(R=1e-8, m=3.0))),
    ("PML d=400 R=1e-5", true, 400.0, NewPML(ZPML(R=1e-5, m=3.0))),
    ("PML d=800 R=1e-8", true, 800.0, NewPML(ZPML(R=1e-8, m=3.0)))]

mmax = floor(Int, P / λ * ns)
results = Dict{String,Any}()
for (label, withpml, dpml, bc) in runs
    println("== P=$P h=$hmesh  $label")
    mesh = joinpath(outdir, "pil_P$(Int(P))_$(withpml ? "pml$(Int(dpml))" : "abc")_h$(Int(hmesh)).msh")
    t = @elapsed (E, sim, _, geo, retries) = mesh_and_solve(geo_for(; dpml), mesh, withpml, bc; λ, nf, nm=complex(nr), ns, pfun)
    rows = order_table(E, geo, λ, nf, ns; Nx=48, Ny=48, mmax, nmax=mmax, nplanes=12, margin=40.0)
    print_orders(rows; label)
    for r in rows
        (r.m, r.n) == (0, 0) && continue
        @printf("    ABC-prediction %-4s (%d,%d) %.3e\n", r.side, r.m, r.n, abc_te_reflection(r.θ))
    end
    raman = raman_metric(E, sim)
    @printf("  ∫|E|⁴ Raman = %.8e   (%.0f s incl. sampling, %d mesh retries)\n", raman, t, retries)
    results[label] = Dict("rows" => [Dict(pairs(r)) for r in rows], "raman" => raman,
        "ndofs" => sim.nV, "retries" => retries)
end
jldsave(joinpath(outdir, "test3_pillars_P$(Int(P))_h$(Int(hmesh)).jld2"); results)
println("saved")

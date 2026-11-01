"""
Test 1 — flat stack, no diffraction: FEM vs the exact layered solution.

A y-polarized sheet source above substrate | slab | fluid. Only the normal
order exists, so the ABC is exact here too; this test checks that the PML
(i) does not disturb the physical field and (ii) absorbs, against the exact
answer, and shows what the 2023 legacy form did.

Run: julia --project scripts/pml/test1_flat.jl [outdir]
"""

include(joinpath(@__DIR__, "pml_common.jl"))
using JLD2

outdir = length(ARGS) >= 1 ? ARGS[1] : mktempdir()
mkpath(outdir)

λ = 532.0
nf, ns = 1.33, 1.45

cases = [
    (name="dielectric slab n=2.0, 100 nm", nm=complex(2.0), hd=100.0, l2=20.0),
    (name="Ag film, 30 nm", nm=resolve_index("Ag", λ), hd=30.0, l2=6.0),
]

function geo_for(c; dpml)
    test_geometry(L=60.0, W=60.0, hd=c.hd, hsub=300.0, ht=100.0, hs=400.0, hair=700.0,
        dpml=dpml, l1=30.0, l2=c.l2, l3=30.0)
end

results = Dict{String,Any}()
for c in cases
    println("== ", c.name, "  (nm = ", c.nm, ")")
    st = stack_z(geo_for(c; dpml=400.0))
    exact = flat_exact_coeffs(st.z_src; λ, ns, nslab=c.nm, nf, z1=st.des_low, z2=st.des_high)
    zline = collect(range(st.z_min + 1, st.z_max - 1, length=400))
    Eex = exact.(zline)
    line = [Point(7.3, -4.1, z) for z in zline]

    runs = Dict{String,Any}()
    function record!(key, E)
        Ey = [v[2] for v in eval_at(E, line)]
        err = norm(Ey - Eex) / norm(Eex)
        maxerr = maximum(abs.(Ey - Eex)) / maximum(abs.(Eex))
        @printf("  %-28s rel L2 err %.3e   max err %.3e\n", key, err, maxerr)
        runs[key] = Dict("Ey" => Ey, "l2" => err, "max" => maxerr)
    end

    mesh_abc = joinpath(outdir, "flat_$(c.hd)_abc.msh")
    genmesh(geo_for(c; dpml=400.0), mesh_abc; pml=false)
    E, sim, _ = solve_case(mesh_abc, ABC(); λ, nf, nm=c.nm, ns, pfun=x -> 1.0)
    record!("ABC (no PML)", E)

    for dpml in (200.0, 400.0)
        mesh_pml = joinpath(outdir, "flat_$(c.hd)_pml$(Int(dpml)).msh")
        genmesh(geo_for(c; dpml), mesh_pml; pml=true)
        for R in (1e-4, 1e-8)
            E, sim, phys = solve_case(mesh_pml, NewPML(ZPML(R=R, m=3.0)); λ, nf, nm=c.nm, ns, pfun=x -> 1.0)
            record!("new PML d=$(Int(dpml)) R=$(R)", E)
        end
        if dpml == 400.0
            E, _, _ = solve_case(mesh_pml, LegacyPML(1e-10, false); λ, nf, nm=c.nm, ns, pfun=x -> 1.0)
            record!("legacy 2023 PML as written", E)
            E, _, _ = solve_case(mesh_pml, LegacyPML(1e-10, true); λ, nf, nm=c.nm, ns, pfun=x -> 1.0)
            record!("legacy, sign flipped", E)
        end
    end
    results[c.name] = Dict("z" => zline, "exact" => Eex, "runs" => runs, "stack" => Dict(pairs(st)))
end

jldsave(joinpath(outdir, "test1_flat.jld2"); results)
println("saved ", joinpath(outdir, "test1_flat.jld2"))

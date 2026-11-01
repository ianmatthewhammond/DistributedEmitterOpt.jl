#=
Test 1b — why does the sign-corrected 2023 form still miss? (flat dielectric slab)

Its z-reduction is the correct stretched scalar equation (as an independent review pointed out), so vary
what a discretisation/termination cause would respond to: mesh size and the
layer strength R. New tensor PML on the same meshes for comparison.

Run: julia --project scripts/pml/test1b_legacy.jl outdir
=#
include(joinpath(@__DIR__, "pml_common.jl"))
outdir = ARGS[1]; mkpath(outdir)
λ = 532.0; nf, ns = 1.33, 1.45; nm = complex(2.0)
for h in (30.0, 15.0)
    geo = test_geometry(L=60.0, W=60.0, hd=100.0, hsub=300.0, ht=100.0, hs=400.0, hair=700.0,
        dpml=400.0, l1=h, l2=min(20.0, h), l3=h)
    st = stack_z(geo)
    exact = flat_exact_coeffs(st.z_src; λ, ns, nslab=nm, nf, z1=st.des_low, z2=st.des_high)
    zl = collect(range(st.z_min + 1, st.z_max - 1, length=400)); Eex = exact.(zl)
    line = [Point(7.3, -4.1, z) for z in zl]
    mesh = joinpath(outdir, "legacy_h$(Int(h)).msh")
    genmesh(geo, mesh; pml=true)
    err(E) = (Ey = [v[2] for v in eval_at(E, line)]; norm(Ey - Eex) / norm(Eex))
    for R in (1e-10, 1e-4, 1e-2)
        E, _, _ = solve_case(mesh, LegacyPML(R, true); λ, nf, nm, ns, pfun=x -> 1.0)
        @printf("  h=%4.1f  legacy sign-fixed R=%.0e : rel L2 err %.3e\n", h, R, err(E))
    end
    E, _, _ = solve_case(mesh, NewPML(ZPML(R=1e-8)); λ, nf, nm, ns, pfun=x -> 1.0)
    @printf("  h=%4.1f  new tensor PML R=1e-8      : rel L2 err %.3e\n", h, err(E))
end

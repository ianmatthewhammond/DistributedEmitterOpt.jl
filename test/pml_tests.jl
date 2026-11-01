"""
±z PML tests: recipe validation, stretch sign and calibration, mesh tags,
and a flat stack against the exact layered solution.

The full validation suite (RCWA gratings, 2D pillars, gradients) is in
scripts/pml/; this file is the fast regression part.

Run with: julia --project test/pml_tests.jl
"""

include(joinpath(@__DIR__, "..", "scripts", "pml", "pml_common.jl"))
using Test

@testset "PML recipe" begin
    k = 2π / 532
    args = (1.45, 1.33, -100.0, 400.0, 100.0, 400.0)
    @test ZPMLParams(ZPML(), k, args...) isa ZPMLParams
    @test_throws ErrorException ZPMLParams(ZPML(m=-2.0), k, args...)
    @test_throws ErrorException ZPMLParams(ZPML(m=NaN), k, args...)
    @test_throws ErrorException ZPMLParams(ZPML(κmax=Inf), k, args...)
    @test_throws ErrorException ZPMLParams(ZPML(κmax=0.5), k, args...)
    @test_throws ErrorException ZPMLParams(ZPML(R=1.0), k, args...)
    @test_throws ErrorException ZPMLParams(ZPML(), 0.0, args...)
    @test_throws ErrorException ZPMLParams(ZPML(), k, 0.0, 1.33, -100.0, 400.0, 100.0, 400.0)
    # a side with zero thickness needs no medium
    @test ZPMLParams(ZPML(), k, 0.0, 1.33, -Inf, 0.0, 100.0, 400.0).d_bot == 0.0
    # pre-PML positional constructor still works
    @test isnothing(Environment("Ag", "Ag", 1.33, false, RunLogConfig()).pml)
end

@testset "PML stretch" begin
    k, n, R = 2π / 532, 1.45, 1e-8
    p = ZPMLParams(ZPML(R=R, m=3.0), k, n, n, -100.0, 400.0, 100.0, 400.0)
    @test pml_stretch(0.0, p) == 1
    @test pml_stretch(-100.0, p) == 1 && pml_stretch(100.0, p) == 1
    # exp(+iωt): absorbing layers have Im s < 0 on both sides
    @test imag(pml_stretch(-300.0, p)) < 0 && imag(pml_stretch(300.0, p)) < 0
    # round-trip amplitude attenuation of the continuous layer is R
    zs = range(100.0, 500.0, length=20001)
    f = [-imag(pml_stretch(z, p)) for z in zs]
    Σ = (sum(f) - (f[1] + f[end]) / 2) * step(zs)   # trapezoid
    @test isapprox(exp(-2k * n * Σ), R; rtol=1e-3)
    # tensors: ε diag(s,s,1/s), μ⁻¹ diag(1/s,1/s,s)
    x = VectorValue(0.0, 0.0, 300.0)
    s = pml_stretch(300.0, p)
    Λε, Λν = DEO.pml_Λε(x, p), DEO.pml_Λν(x, p)
    @test Λε[1, 1] ≈ s && Λε[3, 3] ≈ 1 / s && Λν[1, 1] ≈ 1 / s && Λν[3, 3] ≈ s
end

@testset "PML mesh + flat stack vs exact" begin
    λ, nf, ns, nm = 532.0, 1.33, 1.45, complex(2.0)
    geo = test_geometry(L=60.0, W=60.0, hd=100.0, hsub=300.0, ht=100.0, hs=400.0, hair=700.0,
        dpml=300.0, l1=40.0, l2=20.0, l3=40.0)
    st = stack_z(geo)
    mesh_pml, mesh_abc = tempname() * ".msh", tempname() * ".msh"
    genmesh(geo, mesh_pml; pml=true)
    genmesh(geo, mesh_abc; pml=false)

    E, sim, phys = solve_case(mesh_pml, NewPML(ZPML(R=1e-8)); λ, nf, nm, ns, pfun=x -> 1.0)
    z_bot, d_bot, z_top, d_top = getpmlz(sim.labels, sim.Ω)
    @test isapprox(z_bot, st.z_min; atol=1e-6) && isapprox(d_bot, geo.dpml; atol=1e-6)
    @test isapprox(z_top, st.z_max; atol=1e-6) && isapprox(d_top, geo.dpml; atol=1e-6)

    exact = flat_exact_coeffs(st.z_src; λ, ns, nslab=nm, nf, z1=st.des_low, z2=st.des_high)
    zl = collect(range(st.z_min + 1, st.z_max - 1, length=200))
    line = [Point(7.3, -4.1, z) for z in zl]
    Eex = exact.(zl)
    err(E) = (Ey = [v[2] for v in eval_at(E, line)]; norm(Ey - Eex) / norm(Eex))
    e_pml = err(E)
    E_abc, _, _ = solve_case(mesh_abc, ABC(); λ, nf, nm, ns, pfun=x -> 1.0)
    e_abc = err(E_abc)
    @info "flat stack vs exact" e_pml e_abc
    # At normal incidence the ABC is exact too: the PML must match it to mesh error.
    @test e_pml < 2e-2
    @test e_pml < 3 * e_abc + 1e-3
    rm(mesh_pml; force=true); rm(mesh_abc; force=true)
end

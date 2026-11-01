"""
Test 4 — adjoint gradient through the package path with the PML on.

Same small Ag problem as test/gradient_tests.jl ("3D Elastic Baseline"), but
meshed with genmesh(...; pml=true) and Environment(pml=ZPML()). This goes
through build_phys_params → getpmlz → assemble_maxwell_pml and the cached
adjoint solve. Finite difference vs adjoint along a random direction.

Run: julia --project scripts/pml/test4_gradient.jl
"""

using DistributedEmitterOpt
using LinearAlgebra
using Random
using Printf

function fd_check(prob, p0; δ=1e-8)
    grad = zeros(length(p0))
    Random.seed!(7)
    δp = randn(length(p0)) * δ
    g0 = objective_and_gradient!(grad, p0, prob)
    g1 = objective_and_gradient!(Float64[], p0 + δp, prob)
    fd, adj = g1 - g0, dot(grad, δp)
    (g0=g0, fd=fd, adj=adj, rel=abs(fd - adj) / abs(fd))
end

function build(; pml::Bool)
    λ = 532.0
    geo = SymmetricGeometry(λ; L=100.0, W=100.0, hd=80.0, hsub=40.0)
    geo.l1 = 50.0; geo.l2 = 30.0; geo.l3 = 50.0
    geo.hair = 200.0; geo.hs = 120.0; geo.ht = 80.0
    geo.dpml = 150.0
    meshfile = tempname() * ".msh"
    genmesh(geo, meshfile; pml)
    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33,
        pml=pml ? ZPML(R=1e-8, m=3.0) : nothing)
    objective = SERSObjective(αₚ=Matrix{ComplexF64}(I, 3, 3), volume=true, surface=false,
        use_damage_model=false, γ_damage=1.0, E_threshold=10.0)
    pde = MaxwellProblem(env=env, inputs=[FieldConfig(λ; θ=0.0, pol=:y)], outputs=FieldConfig[])
    control = Control(use_filter=true, R_filter=(20.0, 20.0, 20.0), use_dct=false,
        use_projection=true, β=8.0, η=0.5, use_ssp=true, flag_volume=true, flag_surface=false,
        use_damage=false, γ_damage=1.0, E_threshold=10.0)
    prob = OptimizationProblem(pde, objective, meshfile, UmfpackSolver();
        per_x=false, per_y=false, foundry_mode=false, control=control)
    Random.seed!(42)
    init_random!(prob)
    rm(meshfile; force=true)
    prob
end

for pml in (false, true)
    prob = build(; pml)
    Random.seed!(3)
    p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
    r = fd_check(prob, p0)
    @printf("%-4s  g0 = %.10e   FD = %.6e   adjoint = %.6e   rel.err = %.2e\n",
        pml ? "PML" : "ABC", r.g0, r.fd, r.adj, r.rel)
end

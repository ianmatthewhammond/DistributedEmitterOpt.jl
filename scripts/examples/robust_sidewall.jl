# Small generic example; all geometry coordinates use the same length unit.
using DistributedEmitterOpt
using Random

const DEO = DistributedEmitterOpt
mode = isempty(ARGS) ? :ks : Symbol(ARGS[1])
mode in (:ks, :epigraph) || error("usage: robust_sidewall.jl [ks|epigraph]")

geo = SymmetricGeometry(532.0; L=100.0, W=100.0, hd=80.0, hsub=40.0)
geo.l1 = 50.0; geo.l2 = 30.0; geo.l3 = 50.0
geo.hair = 200.0; geo.hs = 120.0; geo.ht = 80.0
out = mktempdir(cleanup=false)
meshfile = joinpath(out, "example.msh")
genmesh(geo, meshfile; per_x=false, per_y=false)

pde = MaxwellProblem(
    env=Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33),
    inputs=[FieldConfig(532.0; pol=:y)])
control = Control(R_filter=(20.0, 20.0, 20.0), use_dct=true,
                  foundry_projection_mode=:current)
prob = OptimizationProblem(pde, SERSObjective(), meshfile, UmfpackSolver();
    per_x=false, per_y=false, foundry_mode=true, control, root=out)
init_uniform!(prob, 0.5)
prob.p .= clamp.(0.5 .+ 0.15 .* randn(MersenneTwister(42), length(prob.p)), 0.0, 1.0)

sim = DEO.default_sim(prob.sim)
z_bottom, z_top = DEO.getdesignz(sim.labels, sim.Ω)
pivot = (z_bottom + z_top) / 2
samples = sidewall_samples(angles_deg=[85.0, 90.0], z0s=[pivot])
backend = mode == :ks ? :nlopt : :standalone_ccsaq
_, p = optimize!(prob; max_iter=4, β_schedule=[8.0], backend,
    robust_samples=samples, robust_mode=mode, robust_sharpness=50.0,
    backup=false, use_constraints=false)
values = robust_sidewall_samples!(nothing, p, prob, samples)
println("mode=$mode, samples=$samples, worst sample objective=$(minimum(values))")
println("Example files: $out")

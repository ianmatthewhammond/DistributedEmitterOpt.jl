# =============================================================================
# NEW (DistributedEmitterOpt) builders — metal-nominal config.
# Assumes the caller has already done:  import DistributedEmitterOpt as DEO
#                                       using LinearAlgebra
# and included config.jl.
# Faithfully mirrors scripts/experiments/paper/constrained_metal_nominal.jl
# (no foundry_projection_mode override -> DEO default, the config that produced
# the g~64 reproduction).  Constraints are OFF for the beta=8 epoch.
# =============================================================================

const ALPHA_P = Matrix{ComplexF64}(LinearAlgebra.I, 3, 3)

"Generate the ONE shared mesh via DEO genmesh (called once by meshgen job)."
function build_mesh!(meshfile::String)
    geo = DEO.SymmetricGeometry(
        GEOM.L, GEOM.W, GEOM.hair, GEOM.hs, GEOM.ht,
        GEOM.hd, GEOM.hsub, GEOM.dpml, GEOM.l1, GEOM.l2, GEOM.l3, 0,
    )
    DEO.genmesh(geo, meshfile; per_x=true, per_y=true)
    return geo
end

"Build the NEW OptimizationProblem from the shared mesh (UMFPACK, metal)."
function build_new_problem(root::String; meshfile::String=MESHFILE)
    sim = DEO.build_simulation(meshfile;
        foundry_mode=true, dir_x=false, dir_y=true, source_y=true, degree=6)

    env = DEO.Environment(
        mat_design=MAT_DESIGN, mat_substrate=MAT_SUBSTRATE, mat_fluid=N_FLUID)
    inputs = [DEO.FieldConfig(LAMBDA; θ=THETA, pol=:y)]
    outputs = DEO.FieldConfig[]
    pde = DEO.MaxwellProblem(env=env, inputs=inputs, outputs=outputs, α_loss=ALPHA_LOSS)

    objective = DEO.SERSObjective(
        αₚ=ALPHA_P, volume=true, surface=false,
        use_damage_model=false, E_threshold=E_THRESHOLD)

    control = DEO.Control(
        use_filter=true, R_filter=FILTER_RADIUS, use_dct=true, use_projection=true,
        β=BETA, η=ETA, use_ssp=true, R_ssp=R_SSP,
        use_constraints=false,                 # beta=8 epoch -> constraints OFF
        η_erosion=ETA_EROSION, η_dilation=ETA_DILATION, b1=B1, c0=C0,
        use_damage=false, γ_damage=1.0, E_threshold=E_THRESHOLD,
        flag_volume=true, flag_surface=false)

    solver = DEO.UmfpackSolver()
    prob = DEO.OptimizationProblem(pde, objective, sim, solver;
        foundry_mode=true, control=control, root=root)
    return prob
end

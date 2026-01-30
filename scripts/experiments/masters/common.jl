module MastersCommon

using DistributedEmitterOpt
using Gridap
using LinearAlgebra
using Random

export MastersConfig, setup_masters_problem, anisotropic_tensor

"""
Standard configuration for Masters thesis experiments.
"""
Base.@kwdef struct MastersConfig
    # Physics
    λ_pump::Float64 = 532.0
    λ_emission::Float64 = 532.0 # Different for splitting/inelastic cases
    pol_pump::Symbol = :y
    pol_emission::Symbol = :y # Usually :y, but sometimes mixed for output

    # Material
    mat_design::String = "Ag"
    mat_substrate::String = "Ag"
    mat_fluid::Float64 = 1.0 # Thesis often assumes vacuum/air

    # Geometry (nm)
    hd::Float64 = 100.0
    hsub::Float64 = 50.0
    l1::Float64 = 12.5 # Air Res
    l2::Float64 = 2.9  # Design Res
    l3::Float64 = 12.5 # PML Res
    L::Float64 = 100.0 # Not always explicit, but often implied or calc'd
    W::Float64 = 100.0

    # Control
    R_filter::Float64 = 20.0
    foundry_mode::Bool = true

    # Objectives
    anisotropic::Bool = false
    bidirectional::Bool = false
    nonlinear::Bool = false
    E_threshold::Float64 = 10000.0 # Default High (effectively linear)
end

"""
Generate anisotropic polarizability tensor.
"""
function anisotropic_tensor()
    # Diagonal [1 0 0; 0 0 0; 0 0 0] - x-only response
    α = zeros(ComplexF64, 3, 3)
    α[1, 1] = 1.0 + 0.0im
    return α
end


"""
Setup a masters thesis problem instance.
"""
function setup_masters_problem(cfg::MastersConfig)
    # 1. Geometry
    # Thesis typically uses SymmetricGeometry
    geo = SymmetricGeometry()
    # Approximate mapping of L/W if not explicit
    geo.L = cfg.L * 3 # Usually includes PML/Air buffer
    geo.W = cfg.W * 3
    # ... (Actual geometry generation requires careful mapping of l1/l2/l3)
    # For now, we reuse the robust build_simulation with foundry_mode

    # Temporary: Use a generated mesh name
    meshfile = tempname() * ".msh"

    # We will assume a standard periodic box for now using genmesh
    # In reality, we might need to be more precise with l1/l2/l3
    # but `genmesh` lets us pass `lc_design` etc if we extend it.
    # For this reproduction, let's trust `build_simulation`'s default scaling or
    # explicitly pass parameters if available.

    # HACK: Using a standard periodic generation for now.
    # ideally we wrap `gmsh` directly matching the thesis generation.
    geo = SymmetricGeometry()
    # Thesis specific overrides if any:
    # geo.hd = cfg.hd ... (SymmetricGeometry might not have these fields accessible directly in older versions)

    # Generate Mesh
    genmesh(geo, meshfile; per_x=true, per_y=cfg.foundry_mode)

    # 2. Simulation
    sim = build_simulation(meshfile;
        foundry_mode=cfg.foundry_mode,
        dir_x=false,
        dir_y=!cfg.foundry_mode # If foundry (2D grid) -> periodic Y. Else -> PEC Y (Half cell)?
        # Thesis config: full_x=false, full_y=false => Symmetric/HalfCell?
        # Resume.yaml says `dir_x=false, dir_y=false` in foundry 2d tests? 
        # Let's stick to : dir_x=false (periodic), dir_y=false (periodic) for foundry mode
    )

    # 3. Physics
    env = Environment(mat_design=cfg.mat_design, mat_substrate=cfg.mat_substrate, mat_fluid=cfg.mat_fluid)

    pump = FieldConfig(cfg.λ_pump; θ=0.0, pol=cfg.pol_pump)

    outputs = FieldConfig[]
    if cfg.bidirectional
        # Bidirectional: Maximize Pump * Emission
        # We model this as a weighted sum or product in the objective?
        # The new code separates inputs/outputs.
        # "Bidirectional" usually means optimizing E_pump * E_emission
        # effectively handling them as separate fields.
        push!(outputs, FieldConfig(cfg.λ_emission; θ=0.0, pol=cfg.pol_emission))
    elseif cfg.λ_emission != cfg.λ_pump
        # Stokes shifted emission
        push!(outputs, FieldConfig(cfg.λ_emission; θ=0.0, pol=cfg.pol_emission))
    else
        # Elastic
        # Empty outputs list often implies elastic optimization in some contexts,
        # but explicit is better.
    end

    pde = MaxwellProblem(env=env, inputs=[pump], outputs=outputs)

    # 4. Objective
    αₚ = cfg.anisotropic ? anisotropic_tensor() : Matrix{ComplexF64}(I, 3, 3)

    objective = SERSObjective(
        αₚ=αₚ,
        volume=true,
        surface=false,
        use_damage_model=cfg.nonlinear,
        E_threshold=cfg.E_threshold
    )

    # 5. Control
    control = Control(
        use_filter=true,
        R_filter=(cfg.R_filter, cfg.R_filter, cfg.R_filter),
        use_dct=cfg.foundry_mode,
        use_projection=true,
        β=8.0,
        η=0.5,
        use_ssp=true,
        flag_volume=true
    )

    solver = UmfpackSolver()

    prob = OptimizationProblem(pde, objective, sim, solver;
        foundry_mode=cfg.foundry_mode,
        control=control
    )

    return prob, meshfile
end

end

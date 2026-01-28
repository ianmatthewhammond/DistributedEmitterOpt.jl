"""
    Simulation

Container for all FEM infrastructure: discrete model, function spaces,
integration measures, and FoundryGrid for 2D DOF mode.

## FE Spaces
- `V`, `U` — Nédélec spaces for E-field
- `P` — Piecewise constant (raw design on mesh)
- `Pf` — Continuous Lagrange (filtered design)

## Integration Measures
- `dΩ` — Full domain
- `dΩ_design` — Design region
- `dΩ_raman` — Raman/target region
- `dS_top`, `dS_bottom` — ABC boundaries
- `dΓ_source` — Source plane
"""
mutable struct Simulation
    # Discrete model
    model::Any  # DiscreteModel

    # FE spaces (Nédélec for E-field)
    V::Any  # FESpace - test
    U::Any  # FESpace - trial

    # FE spaces (Lagrange for density) — 3D DOF mode
    P::Any   # Piecewise constant
    Pf::Any  # Continuous Lagrange

    # Integration domains and measures
    Ω::Any       # Full domain triangulation
    dΩ::Any      # Measure
    Ω_design::Any
    dΩ_design::Any
    Ω_raman::Any
    dΩ_raman::Any
    dS_top::Any
    dS_bottom::Any
    dΓ_source::Any
    dS_target::Any   # Target surface for surface integrals

    # 2D DOF grid (foundry mode)
    grid::FoundryGrid

    # Metadata
    np::Int              # Number of design DOFs
    nV::Int              # Number of field DOFs
    dir_x::Bool          # Dirichlet (PEC) in x
    dir_y::Bool          # Dirichlet (PEC) in y
    source_y::Bool       # Source polarization direction
    labels::Any          # Face labeling

    # Inner constructor for partial initialization
    Simulation() = new()
end

Base.show(io::IO, s::Simulation) = print(io, "Simulation(np=$(s.np), nV=$(s.nV))")

# ═══════════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_simulation(meshfile; kwargs...) -> Simulation

Build Simulation from mesh file.

## Arguments
- `meshfile` — Path to .msh file
- `order` — Nédélec element order (default 0)
- `degree` — Quadrature degree (default 4)
- `dir_x`, `dir_y` — Dirichlet boundaries for PEC symmetry
- `source_y` — Source polarization (true=y, false=x)
- `foundry_mode` — Use 2D DOF grid
- `sizes` — Manual (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) for grid
"""
function build_simulation(meshfile::String;
    order::Int=0,
    degree::Int=4,
    dir_x::Bool=false,
    dir_y::Bool=true,
    source_y::Bool=true,
    foundry_mode::Bool=true,
    sizes=nothing)

    sim = Simulation()

    # Load mesh
    sim.model = GmshDiscreteModel(meshfile)

    # Dirichlet tags
    dirichlet_tags = String[]
    dir_x && push!(dirichlet_tags, "FacesX")
    dir_y && push!(dirichlet_tags, "FacesY")

    # Nédélec space for E-field
    reffe = ReferenceFE(nedelec, Float64, order)
    if isempty(dirichlet_tags)
        sim.V = TestFESpace(sim.model, reffe, vector_type=Vector{ComplexF64})
    else
        sim.V = TestFESpace(sim.model, reffe, dirichlet_tags=dirichlet_tags, vector_type=Vector{ComplexF64})
    end
    sim.U = sim.V
    sim.nV = num_free_dofs(sim.V)

    # Lagrange spaces for density (3D DOF mode)
    p_reffe = ReferenceFE(lagrangian, Float64, 0)
    sim.P = TestFESpace(sim.model, p_reffe, vector_type=Vector{Float64})
    pf_reffe = ReferenceFE(lagrangian, Float64, 1)
    sim.Pf = TestFESpace(sim.model, pf_reffe, vector_type=Vector{Float64})

    # Labels
    sim.labels = get_face_labeling(sim.model)
    dimension = num_cell_dims(sim.model)

    # Integration domains
    sim.Ω = Triangulation(sim.model)
    sim.dΩ = Measure(sim.Ω, degree)

    # Design region
    cellmask_d = get_face_mask(sim.labels, "Design", dimension)
    sim.Ω_design = Triangulation(sim.model, cellmask_d)
    sim.dΩ_design = Measure(sim.Ω_design, degree)

    # Raman region
    cellmask_r = get_face_mask(sim.labels, "Raman", dimension)
    sim.Ω_raman = Triangulation(sim.model, cellmask_r)
    sim.dΩ_raman = Measure(sim.Ω_raman, degree)

    # Boundaries
    S_top = BoundaryTriangulation(sim.model, tags=["TopZ"])
    sim.dS_top = Measure(S_top, degree)
    S_bottom = BoundaryTriangulation(sim.model, tags=["BottomZ"])
    sim.dS_bottom = Measure(S_bottom, degree)
    Γ_source = BoundaryTriangulation(sim.model, tags=["Source"])
    sim.dΓ_source = Measure(Γ_source, degree)
    S_target = BoundaryTriangulation(sim.model, tags=["Target"])
    sim.dS_target = Measure(S_target, degree)

    # Grid for 2D DOF mode
    nodes = collect(Gridap.Geometry.get_cell_coordinates(sim.Ω))
    np_mesh = num_free_dofs(sim.P)
    sim.grid = getgrid(sim.labels, sim.Ω, np_mesh, nodes; sizes)

    # DOF count based on mode
    if foundry_mode
        sim.np = length(sim.grid.x) * length(sim.grid.y)
    else
        sim.np = np_mesh
    end

    sim.dir_x = dir_x
    sim.dir_y = dir_y
    sim.source_y = source_y

    return sim
end

# ═══════════════════════════════════════════════════════════════════════════════
# Design region helpers
# ═══════════════════════════════════════════════════════════════════════════════

"""Get z-bounds of design region from mesh Design volume cells."""
function getdesignz(labels, Ω)
    dimension = num_cell_dims(Ω.model)
    cellmask_d = get_face_mask(labels, "Design", dimension)
    design_cell_ids = findall(cellmask_d)

    if isempty(design_cell_ids)
        error("No Design cells found in mesh")
    end

    cell_coords = Gridap.Geometry.get_cell_coordinates(Ω)

    all_z = Float64[]
    for cell_id in design_cell_ids
        cell_nodes = cell_coords[cell_id]
        for node in cell_nodes
            push!(all_z, node[3])
        end
    end

    return minimum(all_z), maximum(all_z)
end


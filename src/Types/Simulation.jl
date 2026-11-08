import GridapGmsh

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
- `degree` — Quadrature degree (default 6)
- `dir_x`, `dir_y` — Dirichlet boundaries for PEC symmetry
- `source_y` — Source polarization (true=y, false=x)
- `foundry_mode` — Use 2D DOF grid
- `sizes` — Manual (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) for grid
"""
function build_simulation(meshfile::String;
    order::Int=0,
    degree::Int=6,
    dir_x::Bool=false,
    dir_y::Bool=true,
    source_y::Bool=true,
    foundry_mode::Bool=true,
    sizes=nothing,
    model=nothing,
    bot_PEC::Bool=false)

    sim = Simulation()

    # Load mesh (or reuse model)
    if isnothing(model)
        sim.model = GmshDiscreteModel(meshfile)
        sim.model = repair_gmsh_model(meshfile, sim.model)
    else
        sim.model = model
    end

    # Dirichlet tags
    dirichlet_tags = String[]
    dir_x && push!(dirichlet_tags, "FacesX")
    dir_y && push!(dirichlet_tags, "FacesY")
    bot_PEC && push!(dirichlet_tags, "BottomZ")

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

    # Grid for 2D DOF mode (use Pf DOF-ordered vertex coordinates)
    topo = Gridap.Geometry.get_grid_topology(sim.model)
    vertex_coords = collect(Gridap.Geometry.get_vertex_coordinates(topo))
    nodes = vertex_coords
    begin
        space = sim.Pf
        if hasproperty(space, :space)
            space = getproperty(space, :space)
        end
        if hasproperty(space, :glue)
            glue = getproperty(space, :glue)
            if hasproperty(glue, :free_dof_to_node)
                nodes = vertex_coords[glue.free_dof_to_node]
            end
        end
    end
    np_mesh = num_free_dofs(sim.P)
    # FoundryGrid sizing uses design-restricted cell count so nx/ny track
    # FEM resolution inside the design region, independent of air/substrate/PML
    # mesh refinement. `sim.P` is Lagrange P0, so cells = DOFs on that subset.
    np_design = count(cellmask_d)
    sim.grid = getgrid(sim.labels, sim.Ω, np_design, nodes; sizes)

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
# Mesh repair (ported from old repo)
# ═══════════════════════════════════════════════════════════════════════════════

"""Repair a Gmsh model if it contains bad vertices."""
function repair_gmsh_model(meshfile::String, model)
    grid_topology = Gridap.Geometry.get_grid_topology(model)
    vertex_to_cells = Gridap.Geometry.get_faces(grid_topology, 0, num_cell_dims(grid_topology))
    bad_vertices = findall(i -> i == 0, map(length, vertex_to_cells))
    @show bad_vertices

    if !isempty(bad_vertices)
        model = fix_gmsh_model(meshfile, bad_vertices)
        grid_topology = Gridap.Geometry.get_grid_topology(model)
        vertex_to_cells = Gridap.Geometry.get_faces(grid_topology, 0, num_cell_dims(grid_topology))
        bad_vertices = findall(i -> i == 0, map(length, vertex_to_cells))
        @show bad_vertices
    elseif needs_renumber(model)
        # Some meshes pass the bad-vertex check but still have invalid node ids.
        model = fix_gmsh_model(meshfile, Int[]; renumber=true)
    end

    return model
end

"""Detect if node ids are out of bounds (renumber needed)."""
function needs_renumber(model)::Bool
    try
        grid = if hasproperty(model, :grid)
            getproperty(model, :grid)
        else
            Gridap.Geometry.get_grid(model)
        end
        node_coords = getproperty(grid, :node_coordinates)
        cell_node_ids = getproperty(grid, :cell_node_ids)
        node_ids = Gridap.Geometry.to_dict(cell_node_ids)[:data]
        if isempty(node_ids)
            return false
        end
        max_id = maximum(node_ids)
        min_id = minimum(node_ids)
        return max_id > length(node_coords) || min_id < 1
    catch
        return false
    end
end

"""
Fix a Gmsh model whose mesh has orphan nodes (nodes in no top-dimensional cell).

gmsh's 3D Delaunay occasionally leaves interior nodes that no tetrahedron uses.
The orphans are removed from the gmsh model itself, then GridapGmsh builds grid,
topology and labeling through its normal path, so every table stays consistent.
(The ported version edited GridapGmsh's tables by hand after the fact; it shifted
indices in ascending order and corrupted the vertex→face tables, which made
Nédélec spaces fail with a `BoundsError`.)

`bad_vertices` are the vertices `repair_gmsh_model` found with no cells; it is
used only as a cross-check on the orphans found here.
"""
function fix_gmsh_model(mshfile::String, bad_vertices::AbstractArray; renumber::Bool=true)
    GridapGmsh.@check_if_loaded
    if !isfile(mshfile)
        error("Msh file not found: $mshfile")
    end

    gmsh = GridapGmsh.gmsh
    gmsh.initialize()
    model = try
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("Mesh.MedImportGroupsOfNodes", 1)
        gmsh.open(mshfile)
        n_orphans = remove_orphan_nodes!(gmsh)
        if n_orphans != length(bad_vertices)
            @warn "fix_gmsh_model: removed $n_orphans orphan nodes, expected $(length(bad_vertices))"
        end
        renumber && gmsh.model.mesh.renumberNodes()
        renumber && gmsh.model.mesh.renumberElements()
        GridapGmsh.GmshDiscreteModel(gmsh)
    finally
        gmsh.finalize()
    end
    return model
end

"""
Remove nodes of the loaded gmsh model that no top-dimensional element uses.

Each entity of the cell dimension holding orphans is cleared and re-meshed with
the same nodes (minus the orphans) and the same elements. Returns the number of
nodes removed.
"""
function remove_orphan_nodes!(gmsh)
    Dc = GridapGmsh._setup_cell_dim(gmsh)
    _, _, cell_nodes = gmsh.model.mesh.getElements(Dc)
    used = Set{UInt64}()
    for t in cell_nodes
        union!(used, t)
    end

    n_removed = 0
    for d in 0:Dc, (dim, tag) in gmsh.model.getEntities(d)
        node_tags, coords, _ = gmsh.model.mesh.getNodes(dim, tag, false, false)
        keep = findall(t -> t in used, node_tags)
        length(keep) == length(node_tags) && continue
        if dim != Dc
            error("Orphan mesh nodes on entity (dim=$dim, tag=$tag); only orphans " *
                  "interior to $Dc-dimensional entities can be repaired")
        end
        etypes, etags, enodes = gmsh.model.mesh.getElements(dim, tag)
        gmsh.model.mesh.clear([(dim, tag)])
        xyz = reshape(coords, 3, :)[:, keep]
        gmsh.model.mesh.addNodes(dim, tag, node_tags[keep], vec(xyz))
        for (etype, et, en) in zip(etypes, etags, enodes)
            gmsh.model.mesh.addElements(dim, tag, [etype], [et], [en])
        end
        n_removed += length(node_tags) - length(keep)
    end
    return n_removed
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

"""
    getpmlz(labels, Ω) -> (z_bot, d_bot, z_top, d_top)

Inner faces and thicknesses of the `PMLBottom` / `PMLTop` volumes written by
`genmesh(...; pml=true)`. A missing layer has thickness 0. Errors if the mesh
has neither.
"""
function getpmlz(labels, Ω)
    dimension = num_cell_dims(Ω.model)
    cell_coords = Gridap.Geometry.get_cell_coordinates(Ω)
    names = Gridap.Geometry.get_tag_name(labels)
    function zrange(tag)
        tag in names || return nothing
        ids = findall(get_face_mask(labels, tag, dimension))
        isempty(ids) && return nothing
        zs = [node[3] for id in ids for node in cell_coords[id]]
        return minimum(zs), maximum(zs)
    end
    bot = zrange("PMLBottom")
    top = zrange("PMLTop")
    isnothing(bot) && isnothing(top) &&
        error("Environment has a PML but the mesh has no PMLBottom/PMLTop volume; mesh it with genmesh(...; pml=true)")
    z_bot, d_bot = isnothing(bot) ? (-Inf, 0.0) : (bot[2], bot[2] - bot[1])
    z_top, d_top = isnothing(top) ? (Inf, 0.0) : (top[1], top[2] - top[1])
    return z_bot, d_bot, z_top, d_top
end

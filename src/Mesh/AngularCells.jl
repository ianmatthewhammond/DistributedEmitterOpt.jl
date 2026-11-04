"""
    build_angular_simulation(geo, configs; meshfile, cell=:auto, design_shape=(17,17), ...)

Choose a shared field domain for all configurations: quarter cell at normal
incidence, half cell for one tilt axis, full periodic cell when both axes occur.
`geo.L, geo.W` are the independent mirror-patch widths, which are HALF the lattice
period: the lattice repeats every 2L × 2W, and a periodic axis doubles its
field-domain width. Bloch field configurations refuse a lattice on which a
nonspecular diffraction order propagates (`check_diffraction_orders`). The 2D design is reflected into that domain without
adding design DOFs. No diagonal symmetry is imposed. `cell=:full` is useful for
independent half/full comparisons. Mixed x/y collections use the union domain;
this builder does not claim separate half-cell factorizations for each angle.

Objectives/gradients from OptimizationProblem are normalized to the independent
patch area, so their scale does not depend on the selected field domain.

The CAD curves and points on the mirror walls are tagged with the wall (see
`genmesh`), so a PEC mirror constrains every tangential degree of freedom on
it, for structured and unstructured (`structured=false`) meshes alike.
`reuse_mesh=true` keeps an existing `meshfile` (a resumed run) instead of
generating it again. Every generated mesh gets a recipe file (`meshfile * ".recipe"`)
holding the genmesh inputs, and a reused mesh must match the recipe this call would
write. A mesh without a recipe (generated before recipes existed) must show the
expected extents and periodic links; a missing recipe and missing mirror-wall curve
tags are warnings.
"""
function build_angular_simulation(geo::SymmetricGeometry, configs;
        meshfile::String, cell::Symbol=:auto, design_shape::Tuple{Int,Int}=(17,17),
        order::Int=0, degree::Int=4, bot_PEC::Bool=false, structured::Bool=true,
        reuse_mesh::Bool=false)
    isempty(configs) && throw(ArgumentError("build_angular_simulation: configs must not be empty"))
    cell in (:auto,:quarter,:half_x,:half_y,:full) || throw(ArgumentError("build_angular_simulation: unknown cell $cell"))
    all(>=(2), design_shape) || throw(ArgumentError("build_angular_simulation: design_shape entries must be at least 2"))
    need_x = any(fc -> fc.tilt_x != 0, configs)
    need_y = any(fc -> fc.θ != 0, configs)
    px, py = cell == :auto ? (need_x, need_y) :
        (cell in (:half_x,:full), cell in (:half_y,:full))
    need_x && !px && throw(ArgumentError("build_angular_simulation: tilt_x crosses mirror wall FacesX"))
    need_y && !py && throw(ArgumentError("build_angular_simulation: tilt_y crosses mirror wall FacesY"))
    field_geo = deepcopy(geo)
    field_geo.L *= px ? 2 : 1
    field_geo.W *= py ? 2 : 1
    spacing=min(geo.l1,geo.l2,geo.l3)
    shape=structured ? (max(2,ceil(Int,geo.L/spacing))*(px ? 2 : 1),
                        max(2,ceil(Int,geo.W/spacing))*(py ? 2 : 1)) : nothing
    center = (px ? geo.L/2 : 0.0, py ? geo.W/2 : 0.0)
    recipe = angular_mesh_recipe(field_geo, px, py, shape, center)
    if reuse_mesh && isfile(meshfile)
        check_reused_mesh(meshfile, recipe, field_geo, center, px, py)
    else
        genmesh(field_geo, meshfile; per_x=px, per_y=py,structured_xy=shape,
                center_xy=center, mirror_curves=true)
        write_mesh_recipe(meshfile, recipe)
    end
    bundle = build_simulation_bundle(meshfile; per_x=px,per_y=py,foundry_mode=true,order,degree,bot_PEC)
    for sim in values(bundle.by_pol)
        gx = collect(range(-geo.L/2,geo.L/2; length=design_shape[1]))
        gy = collect(range(-geo.W/2,geo.W/2; length=design_shape[2]))
        grid = FoundryGrid(gx,gy,zeros(design_shape),sim.grid.nodes)
        grid.reflection_bounds = (gx[1],gx[end],gy[1],gy[end])
        sim.grid = grid
        sim.np = prod(design_shape)
        sim.objective_normalization = 1 / ((px ? 2 : 1)*(py ? 2 : 1))
    end
    # Both polarizations share scalar geometry and material spaces exactly.
    base = default_sim(bundle)
    for sim in values(bundle.by_pol)
        sim.P = base.P; sim.Pf = base.Pf; sim.grid = base.grid
        for fc in configs
            fc.polarization == (sim.source_y ? :y : :x) && validate_field_boundaries(sim,fc)
        end
    end
    # Full periodic spaces have identical boundary constraints for both pols.
    # Sharing the object also removes unnecessary field/material transfers.
    if px && py
        bundle.by_pol[:x] = base
    end
    bundle
end

# ---------------------------------------------------------------------------
# Mesh recipes for reuse_mesh
# ---------------------------------------------------------------------------

mesh_recipe_path(meshfile) = meshfile * ".recipe"

"The genmesh inputs that define an angular-cell mesh, as strings."
function angular_mesh_recipe(field_geo::SymmetricGeometry, px, py, shape, center)
    recipe = Dict{String,String}(
        "recipe" => "angular-cell-1",
        "per_x" => string(px), "per_y" => string(py), "mirror_curves" => "true",
        "structured" => shape === nothing ? "none" : "$(shape[1])x$(shape[2])",
        "center_x" => repr(Float64(center[1])), "center_y" => repr(Float64(center[2])))
    for f in (:L, :W, :hair, :hs, :ht, :hd, :hsub, :l1, :l2, :l3)
        recipe[string(f)] = repr(Float64(getfield(field_geo, f)))
    end
    return recipe
end

function write_mesh_recipe(meshfile, recipe)
    open(mesh_recipe_path(meshfile), "w") do io
        println(io, "# genmesh inputs written by build_angular_simulation; checked by reuse_mesh=true")
        for key in sort!(collect(keys(recipe)))
            println(io, key, " = ", recipe[key])
        end
    end
end

function read_mesh_recipe(path)
    recipe = Dict{String,String}()
    for line in eachline(path)
        entry = strip(line)
        (isempty(entry) || startswith(entry, "#")) && continue
        key, value = split(entry, "="; limit=2)
        recipe[String(strip(key))] = String(strip(value))
    end
    return recipe
end

"Node extents of a mesh file and whether it tags curves on the mirror walls (genmesh mirror_curves)."
function mesh_extents_and_wall_curves(meshfile)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(meshfile)
        _, coord, _ = gmsh.model.mesh.getNodes()
        xyz = reshape(coord, 3, :)
        lo, hi = vec(minimum(xyz; dims=2)), vec(maximum(xyz; dims=2))
        names = [gmsh.model.getPhysicalName(1, tag) for (_, tag) in gmsh.model.getPhysicalGroups(1)]
        return lo, hi, any(in(("FacesX", "FacesY")), names)
    finally
        gmsh.finalize()
    end
end

function check_reused_mesh(meshfile, expected, field_geo, center, px, py)
    path = mesh_recipe_path(meshfile)
    if isfile(path)
        found = read_mesh_recipe(path)
        bad = String[]
        for key in sort!(collect(keys(expected)))
            want, have = expected[key], get(found, key, "missing")
            a, b = tryparse(Float64, want), tryparse(Float64, have)
            same = (a !== nothing && b !== nothing) ? isapprox(a, b; atol=1e-9) : want == have
            same || push!(bad, "$key = $have (expected $want)")
        end
        isempty(bad) || throw(ArgumentError("build_angular_simulation: reuse_mesh recipe $path does not match this geometry and cell: " * join(bad, ", ")))
        return nothing
    end
    # No recipe: a mesh from before recipes existed. Check what the mesh itself shows.
    height = field_geo.hair + field_geo.hd + field_geo.hsub
    want_lo = [center[1] - field_geo.L / 2, center[2] - field_geo.W / 2, -height / 2]
    want_hi = [center[1] + field_geo.L / 2, center[2] + field_geo.W / 2, height / 2]
    lo, hi, has_curves = mesh_extents_and_wall_curves(meshfile)
    (isapprox(lo, want_lo; atol=1e-6) && isapprox(hi, want_hi; atol=1e-6)) ||
        throw(ArgumentError("build_angular_simulation: reused mesh $meshfile spans $lo to $hi, expected $want_lo to $want_hi"))
    mesh_periodic_axes(meshfile) == (px, py) ||
        throw(ArgumentError("build_angular_simulation: reused mesh $meshfile has periodic axes $(mesh_periodic_axes(meshfile)), expected ($px, $py)"))
    @warn "build_angular_simulation: reusing $meshfile without a recipe; checked its extents and periodic links only"
    has_curves || @warn "build_angular_simulation: $meshfile has no mirror-wall curve tags (generated before mirror_curves); a PEC mirror leaves the tangential degrees of freedom on the wall curves free"
    return nothing
end

"""
    collection_configs(λ, samples; n_incident=1)

Expand `(tilt_x, tilt_y, weight)` samples into both transverse polarizations.
Weights apply per polarization. Keep signed samples explicit unless a validated
symmetry of the complete objective permits reducing them and adding weights.
"""
function collection_configs(λ::Real, samples; n_incident::Real=1)
    [FieldConfig(λ; tilt_x=tx,tilt_y=ty,weight=w,pol,n_incident)
        for (tx,ty,w) in samples for pol in (:x,:y)]
end

"Normal plus four rim points; weights are quadrature inputs, not an accuracy claim."
function rim_collection_samples(na::Real; n_incident::Real=1,
                                normal_weight::Real=1, rim_weight::Real=1)
    0 <= na < n_incident || throw(ArgumentError("rim_collection_samples: require 0 ≤ NA < n_incident"))
    t = asind(na/n_incident)
    [(0.0,0.0,Float64(normal_weight)),(t,0.0,Float64(rim_weight)),
     (-t,0.0,Float64(rim_weight)),(0.0,t,Float64(rim_weight)),(0.0,-t,Float64(rim_weight))]
end

"""
    reduce_mirror_samples(samples; mirror_x=false, mirror_y=false)

Combine reflected directions by adding weights. The caller asserts symmetry of
both design and complete pump/objective (not just the collection cone). A fixed
normal linear pump with the isotropically averaged SERS tensor and intensity-only
damage satisfies x/y mirror symmetry on a reflected foundry design. Diagonal
exchange is not assumed: a fixed polarized pump generally breaks it.
"""
function reduce_mirror_samples(samples; mirror_x::Bool=false, mirror_y::Bool=false)
    merged = Dict{Tuple{Float64,Float64},Float64}()
    order = Tuple{Float64,Float64}[]
    for (tx,ty,w) in samples
        key = (Float64(mirror_x ? abs(tx) : tx),Float64(mirror_y ? abs(ty) : ty))
        haskey(merged,key) || push!(order,key)
        merged[key] = get(merged,key,0.0) + w
    end
    [(tx,ty,merged[(tx,ty)]) for (tx,ty) in order]
end

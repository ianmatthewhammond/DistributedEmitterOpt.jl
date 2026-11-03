"""
    build_angular_simulation(geo, configs; meshfile, cell=:auto, design_shape=(17,17), ...)

Choose a shared field domain for all configurations: quarter cell at normal
incidence, half cell for one tilt axis, full periodic cell when both axes occur.
`geo.L, geo.W` are the independent mirror-patch widths; a periodic axis doubles
its field-domain width. The 2D design is reflected into that domain without
adding design DOFs. No diagonal symmetry is imposed. `cell=:full` is useful for
independent half/full comparisons. Mixed x/y collections use the union domain;
this builder does not claim separate half-cell factorizations for each angle.

Objectives/gradients from OptimizationProblem are normalized to the independent
patch area, so their scale does not depend on the selected field domain.

The CAD curves and points on the mirror walls are tagged with the wall (see
`genmesh`), so a PEC mirror constrains every tangential degree of freedom on
it, for structured and unstructured (`structured=false`) meshes alike.
`reuse_mesh=true` keeps an existing `meshfile` (a resumed run) instead of
generating it again; the periodic axes are still read from the file.
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
    (reuse_mesh && isfile(meshfile)) ||
        genmesh(field_geo, meshfile; per_x=px, per_y=py,structured_xy=shape,
                center_xy=(px ? geo.L/2 : 0.0,py ? geo.W/2 : 0.0), mirror_curves=true)
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

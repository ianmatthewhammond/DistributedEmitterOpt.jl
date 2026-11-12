# Regressions from the final review of the oblique-incidence PR:
# - a PEC bottom constrains every tangential DOF on its plane, including the perimeter
#   curves and points Gmsh keeps as separate entities (they must carry the BottomZ tag);
# - the θ convenience constructors build tilted configs that measure their tilt in the
#   environment's fluid (n_incident = its lossless index), and θ = 0 stays legacy;
# - the public adjoint/sensitivity API still accepts concrete legacy-key dictionaries.
using DistributedEmitterOpt, Gridap, Test
const DEO = DistributedEmitterOpt

@testset "Oblique API and PEC closure" begin
    λ = 325.0
    geo = SymmetricGeometry(λ; L=85.0, W=85.0, hd=30.0, hsub=60.0)
    geo.hair = 300; geo.hs = 200; geo.ht = 80
    geo.l1 = 30.0; geo.l2 = 15.0; geo.l3 = 30.0
    fc = FieldConfig(λ; tilt_y=30, pol=:x)
    sim = build_angular_simulation(geo, [fc]; meshfile=joinpath(mktempdir(), "pec_full.msh"),
        design_shape=(5, 5), structured=false, bot_PEC=true, cell=:full)
    s = DEO.sim_for(sim, fc)
    model = s.model
    topo = Gridap.Geometry.get_grid_topology(model)
    X = Gridap.Geometry.get_vertex_coordinates(topo)
    zb = minimum(x[3] for x in X)
    labels = Gridap.Geometry.get_face_labeling(model)
    for dim in (0, 1)
        mask = Gridap.Geometry.get_face_mask(labels, "BottomZ", dim)
        f2v = Gridap.Geometry.get_faces(topo, dim, 0)
        onbottom = [all(abs(X[v][3] - zb) < 1e-6 for v in f2v[f]) for f in 1:length(f2v)]
        println("PEC CLOSURE dim=$dim faces in the bottom plane=$(count(onbottom)) tagged=$(count(mask[onbottom]))")
        @test count(onbottom) > 0
        @test all(mask[onbottom])
    end

    nf = sqrt(1.77)
    p = MaxwellProblem(532.0; θ=20.0)
    @test p.inputs[1].formulation == :bloch
    @test p.inputs[1].n_incident ≈ nf
    q = MaxwellProblem(325.0, 342.8; θ=20.0)
    @test q.inputs[1].n_incident ≈ nf && q.outputs[1].n_incident ≈ nf
    @test MaxwellProblem(532.0).inputs[1].formulation == :legacy
    @test MaxwellProblem(532.0, 550.0).outputs[1].formulation == :legacy
    # The tilted config passes build_phys_params on a periodic half cell in the default fluid.
    hsim = build_angular_simulation(geo, [FieldConfig(λ; tilt_y=20, pol=:y, n_incident=nf)];
        meshfile=joinpath(mktempdir(), "half.msh"), design_shape=(5, 5), structured=false)
    tfc = DEO.incident_config(p.env, λ; θ=20.0, pol=:y)
    @test DEO.build_phys_params(tfc, p.env, DEO.sim_for(hsim, tfc)) isa DEO.PhysicalParams

    legacy = Dict{Tuple{Float64,Float64,Symbol},Vector{ComplexF64}}()
    @test hasmethod(DEO.solve_adjoint!, Tuple{MaxwellProblem,typeof(legacy),Any,Any})
end

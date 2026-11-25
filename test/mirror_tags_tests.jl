# A non-periodic axis has mirror walls that are PEC (Dirichlet FacesX or FacesY) for one
# polarization. The tangential field must vanish on every edge of such a wall (and, from Nédélec
# order 1, on every face of it), including the edges on the CAD curves where the wall meets the
# other wall, a layer interface, or the top and bottom faces. genmesh(...; mirror_curves=true)
# tags those curves; plain genmesh tags the wall surfaces only and leaves the curve edges free.
#
# The check below is independent of the package's own labelling and of the mesh topology's
# vertex coordinates: an edge or face is on a wall when all of its vertices, taken from the
# physical node coordinates of the cell itself, lie in one wall plane; and a degree of freedom is
# Dirichlet when the FE space numbers it negatively. (On a mesh periodic in one direction the
# topology merges the vertices of the two periodic faces, so a vertex's topological coordinates
# can lie on the other side of the cell; the cell's own coordinates do not.)
using DistributedEmitterOpt, Gridap, Test
const GG = Gridap.Geometry
const RF = Gridap.ReferenceFEs

"""
For the field space of `sim` (Nédélec order `order`), the global ids of the DOFs owned by edges and
faces that lie in a wall plane normal to one of `pec_axes`, split into Dirichlet and free, and the
same for every other edge and face.
"""
function wall_dof_status(sim, pec_axes, order)
    model = sim.model
    cc = GG.get_cell_coordinates(GG.get_grid(model))
    lo = [minimum(p[a] for c in cc for p in c) for a in 1:3]
    hi = [maximum(p[a] for c in cc for p in c) for a in 1:3]
    tol = 1e-9 * maximum(hi .- lo)
    poly = only(GG.get_polytopes(model))
    own = RF.get_face_own_dofs(RF.NedelecRefFE(Float64, poly, order))
    cell_dofs = get_cell_dof_ids(sim.V)
    wall_dir, wall_free, off_dir, off_free = Set{Int}(), Set{Int}(), Set{Int}(), Set{Int}()
    for cell in 1:length(cc)
        xyz = cc[cell]
        dofs = cell_dofs[cell]
        for d in 1:2
            offset = RF.get_offset(poly, d)
            for (j, fv) in enumerate(RF.get_faces(poly, d, 0))
                ldofs = own[offset+j]
                isempty(ldofs) && continue
                wall = any(a -> all(v -> abs(xyz[v][a] - lo[a]) <= tol, fv) ||
                                all(v -> abs(xyz[v][a] - hi[a]) <= tol, fv), pec_axes)
                for g in dofs[ldofs]
                    push!(wall ? (g < 0 ? wall_dir : wall_free) : (g < 0 ? off_dir : off_free), g)
                end
            end
        end
    end
    return (; wall_dir, wall_free, off_dir, off_free)
end

"Collect `genmesh(` snippets without a literal `...` from the listed files."
function genmesh_calls(text)
    calls = String[]
    for m in eachmatch(r"genmesh\(", text)
        j, depth = m.offset + ncodeunits("genmesh("), 1
        while depth > 0 && j <= ncodeunits(text)
            c = text[j]
            c == '(' && (depth += 1)
            c == ')' && (depth -= 1)
            j = nextind(text, j)
        end
        call = text[m.offset:prevind(text, j)]
        occursin("...", call) || push!(calls, call)
    end
    return calls
end

"Check the Dirichlet status of the enumerated edge and face DOFs."
closed_and_only(st) = isempty(st.wall_free) && isempty(st.off_dir) && !isempty(st.wall_dir) &&
                      !isempty(st.off_free) && isempty(intersect(union(st.wall_dir, st.wall_free),
                                                                 union(st.off_dir, st.off_free)))

@testset "Mirror wall tags" begin
    QUICKSTART = "genmesh(geo, meshfile; per_x=false, per_y=false, mirror_curves=true)"
    root = joinpath(@__DIR__, "..")

    geo = SymmetricGeometry(532.0; L=100.0, W=100.0, hd=80.0, hsub=40.0)
    geo.l1 = 50.0; geo.l2 = 30.0; geo.l3 = 50.0
    geo.hair = 200.0; geo.hs = 120.0; geo.ht = 80.0
    dir = mktempdir()
    mesh(name; kw...) = (f = joinpath(dir, name); genmesh(geo, f; kw...); f)
    tagged = mesh("tagged.msh"; per_x=false, per_y=false, mirror_curves=true)
    per_x = mesh("per_x.msh"; per_x=true, per_y=false, mirror_curves=true)
    per_y = mesh("per_y.msh"; per_x=false, per_y=true, mirror_curves=true)
    legacy = mesh("legacy.msh"; per_x=false, per_y=false)        # surface-only wall tags

    @testset "README quick start" begin
        @test occursin(QUICKSTART, read(joinpath(root, "README.md"), String))
        @test occursin(QUICKSTART, read(joinpath(root, "docs", "src", "index.md"), String))
    end

    # regression: the quick start's mesh left the wall-curve edges free. On meshes the package
    # generates with mirror_curves=true, every wall DOF is Dirichlet and no other DOF is, for both
    # polarizations, at orders 0, 1 and 2, without periodicity and with periodicity in the other
    # direction (where only one polarization has a PEC wall).
    cases = (("non-periodic, y", tagged, (; dir_x=false, dir_y=true, source_y=true), [2]),
             ("non-periodic, x", tagged, (; dir_x=true, dir_y=false, source_y=false), [1]),
             ("periodic in x, y", per_x, (; dir_x=false, dir_y=true, source_y=true), [2]),
             ("periodic in y, x", per_y, (; dir_x=true, dir_y=false, source_y=false), [1]))
    @testset "Generated mesh, $label, order $order" for (label, file, kw, axes) in cases, order in 0:2
        sim = build_simulation(file; foundry_mode=false, order, kw...)
        st = wall_dof_status(sim, axes, order)
        @test closed_and_only(st)
        # The whole space: the Dirichlet DOFs are exactly the wall DOFs counted above (none in a cell's interior).
        @test length(st.wall_dir) == Gridap.FESpaces.num_dirichlet_dofs(sim.V)
        @test isempty(st.wall_free)          # wall DOFs are all Dirichlet
        @test isempty(st.off_dir)            # no DOF off the PEC wall planes is Dirichlet
    end

    @testset "Control: surface-only wall tags leave wall DOFs free" begin
        # The check above can fail: on the plain genmesh mesh some wall DOFs are free.
        for order in 0:1
            sim = build_simulation(legacy; foundry_mode=false, order, dir_x=false, dir_y=true, source_y=true)
            st = wall_dof_status(sim, [2], order)
            @test !isempty(st.wall_free)
            @test isempty(st.off_dir)
        end
        # Same volume cells, fewer free DOFs with the curve tags.
        a = build_simulation(tagged; foundry_mode=false, dir_x=false, dir_y=true, source_y=true)
        b = build_simulation(legacy; foundry_mode=false, dir_x=false, dir_y=true, source_y=true)
        @test num_cells(a.model) == num_cells(b.model)
        @test a.nV < b.nV
    end

    @testset "Test fixture mesh (gradient_common.jl)" begin
        # The shared gradient fixture's mesh closes its walls too, for both polarizations of its bundle.
        isdefined(@__MODULE__, :build_test_problem) || include(joinpath(@__DIR__, "gradient_common.jl"))
        prob = build_test_problem()
        for (pol, axes) in ((:y, [2]), (:x, [1]))
            @test closed_and_only(wall_dof_status(prob.sim.by_pol[pol], axes, 0))
        end
    end

    @testset "Every genmesh call of the examples, scripts and test fixtures passes mirror_curves=true" begin
        # Each of these calls builds a mesh with a non-periodic axis (or the PML stack, whose side walls
        # are mirrors): the call must carry mirror_curves=true.
        files = ["scripts/examples/robust_sidewall.jl", "scripts/pml/pml_common.jl", "scripts/pml/test1_flat.jl",
                 "scripts/pml/test1b_legacy.jl", "scripts/pml/test4_gradient.jl",
                 "docs/src/examples/anisotropic_3d_inelastic_optimization.jl",
                 "docs/src/examples/anisotropic_3d_inelastic_optimization.md",
                 "test/constrained_optimization_tests.jl", "test/constraint_tests.jl", "test/gradient_common.jl",
                 "test/logger_tests.jl", "test/oblique_normal_equivalence_tests.jl", "test/pml_tests.jl"]
        for f in files
            text = read(joinpath(root, f), String)
            calls = genmesh_calls(text)
            @test !isempty(calls)
            for c in calls
                @test occursin("mirror_curves=true", c) || (println("missing in $f: $c"); false)
            end
        end
    end
end

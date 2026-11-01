"""
Mesh repair tests for DistributedEmitterOpt.jl

gmsh's 3D Delaunay sometimes leaves orphan interior nodes (nodes in no
tetrahedron). `repair_gmsh_model` must then return a model that is the raw mesh
minus those nodes, on which Nédélec spaces build and Maxwell solves.

The geometry below reproduces it: `genmesh` leaves two orphan nodes in the
design volume. Before the fix, `fix_gmsh_model` returned a model whose Nédélec
space failed with a `BoundsError` at orders 0 and 1.
"""

using DistributedEmitterOpt
using DistributedEmitterOpt: build_simulation, getdesignz, assemble_maxwell, assemble_source,
    PhysicalParams, repair_gmsh_model
using Gridap
using GridapGmsh
using LinearAlgebra
using Test

const GG = Gridap.Geometry

orphan_geometry() = SymmetricGeometry(300.0, 30.0, 1000.0, 600.0, 100.0, 150.0, 600.0, 400.0,
    20.0, 10.0, 20.0, 0)

cellless_vertices(model) =
    findall(isempty, GG.get_faces(GG.get_grid_topology(model), 0, num_cell_dims(model)))

mktempdir() do tmp
    meshfile = joinpath(tmp, "orphan_nodes.msh")
    genmesh(orphan_geometry(), meshfile)

    raw = GmshDiscreteModel(meshfile)
    @testset "Reproducer has orphan nodes" begin
        @test length(cellless_vertices(raw)) == 2
    end

    model = repair_gmsh_model(meshfile, raw)

    @testset "Repaired model is the raw mesh minus orphans" begin
        @test isempty(cellless_vertices(model))
        @test num_vertices(model) == num_vertices(raw) - 2
        @test num_cells(model) == num_cells(raw)
        cr = GG.get_cell_coordinates(Triangulation(raw))
        cm = GG.get_cell_coordinates(Triangulation(model))
        @test all(i -> Set(cr[i]) == Set(cm[i]), 1:num_cells(raw))
        lr, lm = get_face_labeling(raw), get_face_labeling(model)
        @test GG.get_tag_name(lr) == GG.get_tag_name(lm)
        for tag in GG.get_tag_name(lr), d in 1:3
            @test count(GG.get_face_mask(lr, tag, d)) == count(GG.get_face_mask(lm, tag, d))
        end
    end

    @testset "Nédélec space builds and solves (order $order)" for order in (0, 1)
        sim = build_simulation(meshfile; order, degree=2order + 2, dir_x=false, dir_y=true,
            foundry_mode=false)
        @test sim.nV > 0
        order == 1 && continue  # order-1 LU is ~140k dofs; the space build is the regression
        lo, hi = getdesignz(sim.labels, sim.Ω)
        phys = PhysicalParams(2π / 532, 0.0, complex(1.33), complex(0.054, 3.429), complex(1.45),
            1.0, lo, hi, 0.0, false)
        A = assemble_maxwell(CellField(x -> 0.0, sim.Ω_design), sim, phys)
        b = assemble_source(sim, phys; source_y=true)
        x = lu(A) \ b
        @test norm(A * x - b) / norm(b) < 1e-8
        @test isfinite(norm(x)) && norm(x) > 0
    end
end

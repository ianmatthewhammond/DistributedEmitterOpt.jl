# reuse_mesh=true must not accept a mesh generated for another geometry or cell. Generated
# meshes carry a recipe of their genmesh inputs and the SHA-256 of the mesh file, and a reused
# mesh is also checked against the extents, periodic links and wall tags it shows; an older
# mesh without a recipe gets those checks, with warnings for what cannot be checked.
include("oblique_common.jl")

@testset "Angular mesh reuse checks the recipe" begin
    geo = angular_test_geometry(; resolution=20.0)
    fc = FieldConfig(532; tilt_y=20, pol=:x)
    dir = mktempdir()
    mesh = joinpath(dir, "cell.msh")
    build(g, cfgs, file) = build_angular_simulation(g, cfgs; meshfile=file, design_shape=(5, 5),
        structured=false, reuse_mesh=true)
    build(geo, [fc], mesh)                     # no mesh yet: generated, recipe written
    @test isfile(mesh * ".recipe")
    @test DEO.default_sim(build(geo, [fc], mesh)).per_y
    wrong = deepcopy(geo)
    wrong.L = 70.0
    @test_throws ArgumentError build(wrong, [fc], mesh)
    finer = deepcopy(geo)
    finer.l2 = 10.0
    @test_throws ArgumentError build(finer, [fc], mesh)
    @test_throws ArgumentError build(geo, [FieldConfig(532; tilt_x=20, pol=:y)], mesh)

    # The recipe is bound to its mesh file. A copy with its recipe is accepted; a replaced mesh
    # file is refused although its recipe still matches the call.
    refusal(f) = try f(); "" catch err; err isa ArgumentError ? err.msg : rethrow() end
    copied = joinpath(dir, "copied.msh")
    cp(mesh, copied); cp(mesh * ".recipe", copied * ".recipe")
    @test DEO.default_sim(build(geo, [fc], copied)).per_y
    other = joinpath(dir, "other.msh")
    build(wrong, [fc], other)                  # a mesh of another geometry, with its own recipe
    cp(other, copied; force=true)
    @test occursin("SHA-256", refusal(() -> build(geo, [fc], copied)))
    # A recipe that matches the call and the file's digest but was written for another mesh:
    # the mesh's own extents are checked as well.
    forged = joinpath(dir, "forged.msh")
    cp(other, forged)
    open(forged * ".recipe", "w") do io
        for line in eachline(mesh * ".recipe")
            println(io, startswith(line, "mesh_sha256") ? "mesh_sha256 = $(DEO.mesh_digest(forged))" : line)
        end
    end
    @test occursin("spans", refusal(() -> build(geo, [fc], forged)))

    # An older mesh: no recipe and surface-only wall tags.
    old = joinpath(dir, "old.msh")
    field = deepcopy(geo)
    field.W *= 2
    genmesh(field, old; per_x=false, per_y=true, center_xy=(0.0, geo.W / 2))
    @test !isfile(old * ".recipe")
    @test_logs (:warn, r"without a recipe") (:warn, r"no mirror-wall curve tags") match_mode = :any build(geo, [fc], old)
    @test_throws ArgumentError build(wrong, [fc], old)
end

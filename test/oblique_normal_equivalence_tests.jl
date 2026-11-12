# At normal incidence in air, the corrected formulation (explicit tilt keyword) and the legacy
# θ = 0 path assemble the same source and the same matrix, except for one term: the specular
# admittance acts on the tangential boundary trace only, the legacy absorbing condition on the
# full trace. The difference is assembled here and must account for A_legacy − A_corrected.
using DistributedEmitterOpt, Gridap, LinearAlgebra, Test
const DEO = DistributedEmitterOpt

@testset "Corrected normal incidence against legacy θ = 0" begin
    geo = SymmetricGeometry(532.0; L=60.0, W=60.0, hd=30.0, hsub=60.0)
    geo.hair = 160; geo.hs = 100; geo.ht = 40
    geo.l1 = 15.0; geo.l2 = 15.0; geo.l3 = 15.0
    mesh = joinpath(mktempdir(), "normal.msh")
    genmesh(geo, mesh)
    env = Environment(mat_design="Ag", mat_substrate=1.45, mat_fluid=1.0)
    ẑ = VectorValue(0.0, 0.0, 1.0)
    for pol in (:x, :y)
        sim = build_simulation(mesh; dir_x=pol == :x, dir_y=pol == :y, source_y=pol == :y, foundry_mode=false)
        legacy = DEO.build_phys_params(FieldConfig(532.0; θ=0.0, pol), env, sim)
        corrected = DEO.build_phys_params(FieldConfig(532.0; tilt_y=0.0, pol), env, sim)
        @test legacy.formulation == :legacy && corrected.formulation == :bloch
        pt = interpolate(x -> 0.4 + 0.1 * sin(x[1] / 7 + x[2] / 11), sim.Pf)
        A_legacy = assemble_maxwell(pt, sim, legacy)
        A_corrected = assemble_maxwell(pt, sim, corrected)
        b_legacy = assemble_source(sim, legacy; source_y=pol == :y)
        b_corrected = assemble_source(sim, corrected; source_y=pol == :y)
        k, ns = legacy.ω, legacy.ns
        D = assemble_matrix(sim.U, sim.V) do u, v
            im * k * ∫((v ⋅ ẑ) * (u ⋅ ẑ))sim.dS_top + im * k * ns * ∫((v ⋅ ẑ) * (u ⋅ ẑ))sim.dS_bottom
        end
        source = norm(b_legacy - b_corrected) / norm(b_legacy)
        matrix = norm(A_legacy - A_corrected - D) / norm(A_legacy)
        normal = norm(D) / norm(A_legacy)
        println("NORMAL pol=$pol source=$source matrix_minus_normal_trace=$matrix normal_trace=$normal")
        @test source < 1e-14
        @test matrix < 1e-13
        @test normal > 1e-8
    end
end

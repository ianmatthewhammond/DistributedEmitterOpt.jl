
using Test
using Random
using LinearAlgebra
using DistributedEmitterOpt

struct CaseSpec
    name::String
    foundry_mode::Bool
    flag_volume::Bool
    flag_surface::Bool
    use_damage::Bool
    isotropic::Bool
    λ_pump::Float64
    λ_emission::Float64
    complex_config::Bool
end

const CASES = [
    CaseSpec("3D Elastic Baseline", false, true, false, false, true, 532.0, 532.0, false),
    CaseSpec("3D Inelastic Scattering", false, true, false, false, true, 532.0, 600.0, false),
    CaseSpec("3D Complex Configuration", false, true, false, false, true, 532.0, 532.0, true),
    CaseSpec("3D Surface Objective", false, false, true, false, true, 532.0, 532.0, false),
    CaseSpec("3D With Damage Model", false, true, false, true, true, 532.0, 532.0, false),
    CaseSpec("3D Anisotropic (Elastic)", false, true, false, false, false, 532.0, 532.0, false),
    CaseSpec("3D Anisotropic + Multi Output", false, true, false, false, false, 532.0, 532.0, true),
    CaseSpec("2D Foundry Mode", true, true, false, false, true, 532.0, 532.0, false),
    CaseSpec("2D Foundry Inelastic", true, true, false, false, true, 532.0, 600.0, false),
    CaseSpec("2D Foundry Anisotropic", true, true, false, false, false, 532.0, 532.0, false),
    CaseSpec("2D Foundry Anisotropic + Multi Output", true, true, false, false, false, 532.0, 532.0, true),
]

const OBJECTIVE_BASELINES = Dict(
    "3D Anisotropic + Multi Output" => (len=813, norm=14.345340287591288, obj=116460.59121184202),
    "3D Inelastic Scattering" => (len=813, norm=14.345340287591288, obj=106857.88184860666),
    "2D Foundry Anisotropic + Multi Output" => (len=10000, norm=50.37032830942858, obj=995260.7610887402),
    "2D Foundry Anisotropic" => (len=10000, norm=50.37032830942858, obj=986770.7457187152),
    "3D Surface Objective" => (len=813, norm=14.345340287591288, obj=1447.3803882181296),
    "3D With Damage Model" => (len=813, norm=14.345340287591288, obj=106625.11391929301),
    "3D Elastic Baseline" => (len=813, norm=14.345340287591288, obj=106625.11391929301),
    "3D Anisotropic (Elastic)" => (len=813, norm=14.345340287591288, obj=114082.55721479755),
    "2D Foundry Inelastic" => (len=10000, norm=50.37032830942858, obj=896830.5675490539),
    "3D Complex Configuration" => (len=813, norm=14.345340287591288, obj=108785.70661678203),
    "2D Foundry Mode" => (len=10000, norm=50.37032830942858, obj=922266.9934199577),
)

function outputs_for_case(case::CaseSpec)
    if case.complex_config
        return FieldConfig[
            FieldConfig(case.λ_emission; θ=0.0, pol=:y, weight=1.0),
            FieldConfig(case.λ_emission + 10.0; θ=0.0, pol=:x, weight=0.5),
        ]
    elseif case.λ_emission == case.λ_pump
        return FieldConfig[]
    else
        return FieldConfig[FieldConfig(case.λ_emission; θ=0.0, pol=:y, weight=1.0)]
    end
end

function anisotropic_tensor()
    α = ComplexF64[
        1.10+0.00im 0.05+0.02im 0.01-0.03im
        0.02-0.01im 0.95+0.00im 0.03+0.04im
        0.01+0.00im 0.02-0.02im 1.05+0.00im
    ]
    return (α + transpose(α)) / 2
end

function alpha_for_case(case::CaseSpec)
    if case.isotropic
        return ComplexF64[
            1.0+0.0im 0.0+0.0im 0.0+0.0im
            0.0+0.0im 1.0+0.0im 0.0+0.0im
            0.0+0.0im 0.0+0.0im (1.0+1e-9)+0.0im
        ]
    end
    return anisotropic_tensor()
end

function geom_params_for_case(case::CaseSpec)
    if case.foundry_mode
        g = SymmetricGeometry()
        g.L = 200.0
        g.W = 200.0
        g.l1 = 40.0
        g.l2 = 20.0
        g.l3 = 40.0
        return g
    else
        g = SymmetricGeometry(case.λ_pump; L=100.0, W=100.0, hd=80.0, hsub=40.0)
        g.l1 = 50.0
        g.l2 = 30.0
        g.l3 = 50.0
        g.hair = 200.0
        g.hs = 120.0
        g.ht = 80.0
        return g
    end
end

function gen_p0(n::Int; seed::Int=42)
    Random.seed!(seed)
    return 0.4 .+ 0.2 .* rand(n)
end

function build_new_problem(case::CaseSpec, meshfile::String)
    geo = geom_params_for_case(case)
    genmesh(geo, meshfile; per_x=false, per_y=false)
    nf = case.flag_volume ? sqrt(1.77) : 1.0
    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=nf)
    inputs = [FieldConfig(case.λ_pump; θ=0.0, pol=:y)]
    outputs = outputs_for_case(case)
    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)
    objective = SERSObjective(
        αₚ=alpha_for_case(case),
        volume=case.flag_volume,
        surface=case.flag_surface,
        use_damage_model=case.use_damage,
        γ_damage=1.0,
        E_threshold=10.0
    )
    control = Control(
        use_filter=true,
        R_filter=(20.0, 20.0, 20.0),
        use_dct=true,
        use_projection=true,
        β=8.0,
        R_ssp=2.0,
        η=0.5,
        use_ssp=true,
        flag_volume=case.flag_volume,
        flag_surface=case.flag_surface,
        use_damage=case.use_damage,
        γ_damage=1.0,
        E_threshold=10.0
    )
    solver = UmfpackSolver()
    return OptimizationProblem(pde, objective, meshfile, solver;
        per_x=false,
        per_y=false,
        foundry_mode=case.foundry_mode,
        control=control
    )
end

@testset "Objective Baselines (New Code)" begin
    mktempdir() do tmp
        for case in CASES
            base = OBJECTIVE_BASELINES[case.name]
            meshfile = joinpath(tmp, replace(case.name, " " => "_") * ".msh")
            prob = build_new_problem(case, meshfile)
            p = gen_p0(length(prob.p))
            g = objective_and_gradient!(zeros(length(p)), p, prob)

            @test length(p) == base.len
            @test isapprox(norm(p), base.norm; rtol=1e-10, atol=0.0)

            if case.foundry_mode
                # 2D foundry objective is allowed to drift slightly with filtering/projection changes.
                @test isapprox(g, base.obj; rtol=1e-10, atol=0.0)
            else
                # 3D should match baseline to machine precision.
                @test isapprox(g, base.obj; rtol=1e-10, atol=0.0)
            end

            rm(meshfile; force=true)
        end
    end
end

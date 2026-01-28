"""
    New-code debug script (inelastic): isotropic tensor, 3D DOFs, volume objective, inelastic (different pump/emission types).

Run from REPL:
    julia> include("scripts/experiments/isotropic_3d_volume_inelastic/debug_isotropic_3d_volume_inelastic.jl")
"""

using Pkg;
const OLD_ROOT = "/Users/ianhammond/GitHub/Emitter3DTopOpt"
Pkg.activate(OLD_ROOT)
using Revise

using DistributedEmitterOpt
using Gridap
using LinearAlgebra
using Random
using PyCall

const OUTDIR = dirname(@__FILE__)
const PERTURBATION = 1e-8
const TOL_RELATIVE = 1e-4

"""Build a problem for inelastic scattering (pump != emission)."""
function build_inelastic_problem(; outdir::String=OUTDIR)
    mkpath(outdir)

    # Geometry (reuse existing mesh for consistency)
    meshfile = "/Users/ianhammond/GitHub/DistributedEmitterOpt.jl/scripts/experiments/isotropic_3d_volume_old_sandbox/mesh.msh"

    # 3D DOF mode, normal incidence (default source_y=true)
    sim = build_simulation(meshfile; foundry_mode=false, dir_x=false, dir_y=true)

    # Isotropic polarizability tensor (identity)
    αₚ = Matrix{ComplexF64}(I, 3, 3)

    # SERS Objective 
    # Logic in standard SERSObjective picks first sorted key as pump, last as emission.
    objective = SERSObjective(
        αₚ=αₚ,
        volume=true,
        surface=false,
        use_damage_model=false,
        E_threshold=10.0
    )

    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)

    # Inelastic Configuration
    # Pump at 532nm
    pump_config = FieldConfig(532.0; θ=0.0, pol=:y)

    # Emission at 560nm (Raman shifted)
    emission_config = FieldConfig(560.0; θ=0.0, pol=:y)

    inputs = [pump_config]
    outputs = [emission_config]

    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

    control = Control(
        use_filter=true,
        R_filter=(20.0, 20.0, 20.0),
        use_dct=false,         # Helmholtz filter for 3D DOF mode
        use_projection=true,
        β=Inf,
        η=0.5,
        use_ssp=true,
        R_ssp=2.0,
        flag_volume=true,
        flag_surface=false,
        use_damage=false,
        E_threshold=10.0
    )

    solver = UmfpackSolver()
    prob = OptimizationProblem(pde, objective, sim, solver;
        foundry_mode=false,
        control=control
    )

    Random.seed!(2)
    init_random!(prob)

    return prob
end

"""Run a finite-difference directional derivative check."""
function test_gradient(prob, p0; δ=PERTURBATION, verbose=true)
    np = length(p0)
    grad = zeros(np)
    δp = randn(np) * δ

    g0 = objective_and_gradient!(grad, p0, prob)
    g1 = objective_and_gradient!(Float64[], p0 + δp, prob)

    fd = g1 - g0
    adj = dot(grad, δp)
    rel_error = abs(fd - adj) / (abs(fd) + 1e-12)

    if verbose
        println("g0 = $g0")
        println("g1 = $g1")
        println("FD  = $fd")
        println("Adj = $adj")
        println("Relative error = $(round(rel_error * 100, digits=2))%")
    end

    return rel_error
end

println("\n=== New-code Debug (Inelastic): isotropic, 3D DOF, volume, Pump 532nm / Emit 560nm ===")
println("Output directory: $(OUTDIR)")

prob = build_inelastic_problem()
p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
rel_err = test_gradient(prob, p0)
println("PASS = $(rel_err < TOL_RELATIVE)  (rel_err = $rel_err, tol = $TOL_RELATIVE)")

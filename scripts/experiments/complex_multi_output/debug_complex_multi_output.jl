"""
    New-code debug script (complex multi-output): 
    - 4 distinct emission outputs (mixed pol, different weights).
    - Pump at 532nm (pol=:x).
    - Verifies gradient correctness for weighted sum of objectives.
    - Uses finite β=8.0 to ensure smooth gradients.

Run from REPL:
    julia> include("scripts/experiments/complex_multi_output/debug_complex_multi_output.jl")
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

"""Build a problem with complex multi-output configuration."""
function build_complex_problem(; outdir::String=OUTDIR)
    mkpath(outdir)

    # Geometry (reuse existing mesh for consistency)
    meshfile = "/Users/ianhammond/GitHub/DistributedEmitterOpt.jl/scripts/experiments/isotropic_3d_volume_old_sandbox/mesh.msh"

    # 3D DOF mode
    # Open boundaries (Periodic or ABC usually required, but for debug we disable Dirichlet triggers)
    # dir_x/dir_y=false implies natural or periodic depending on mesh/space construction.
    # Given we are not calling genperiodic, boundaries might be Natural (Neumann/ABC?).
    # This is safer for mixed polarization than enforcing PEC.
    sim = build_simulation(meshfile; foundry_mode=false, dir_x=false, dir_y=false)

    # Isotropic polarizability tensor (identity)
    αₚ = Matrix{ComplexF64}(I, 3, 3)

    # SERS Objective 
    objective = SERSObjective(
        αₚ=αₚ,
        volume=true,
        surface=false,
        use_damage_model=false,
        E_threshold=10.0
    )

    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)

    # COMPLEX CONFIGURATION

    # Pump: 532nm, Polarization X
    pump_config = FieldConfig(532.0; θ=0.0, pol=:x)

    # Emission Outputs:
    # 1. 540nm, Pol X, Weight 1.0
    # 2. 550nm, Pol Y, Weight 0.5
    # 3. 560nm, Pol X, Weight 2.0
    # 4. 570nm, Pol Y, Weight 0.1

    outputs = [
        FieldConfig(540.0; θ=0.0, pol=:y, weight=1.0),
        # FieldConfig(550.0; θ=0.0, pol=:y, weight=0.5),
        # FieldConfig(560.0; θ=0.0, pol=:x, weight=2.0),
        # FieldConfig(570.0; θ=0.0, pol=:y, weight=0.1)
    ]

    inputs = [pump_config]

    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

    control = Control(
        use_filter=true,
        R_filter=(20.0, 20.0, 20.0),
        use_dct=false,         # Helmholtz filter for 3D DOF mode
        use_projection=true,
        β=8.0,                 # Finite beta for gradients
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

    Random.seed!(42) # Different seed
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

println("\n=== New-code Debug (Complex Multi-Output): 4 Emissions, Mixed Pol/Weights ===")
println("Output directory: $(OUTDIR)")
println("Configuration:")
println("  Pump: 532nm (Pol X)")
println("  Emit 1: 540nm (Pol X, w=1.0)")
println("  Emit 2: 550nm (Pol Y, w=0.5)")
println("  Emit 3: 560nm (Pol X, w=2.0)")
println("  Emit 4: 570nm (Pol Y, w=0.1)")

prob = build_complex_problem()
p0 = 0.5 .+ 0.1 .* rand(length(prob.p))
rel_err = test_gradient(prob, p0)
println("PASS = $(rel_err < TOL_RELATIVE)  (rel_err = $rel_err, tol = $TOL_RELATIVE)")

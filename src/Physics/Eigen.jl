"""
    Eigen

Assembly for the generalized eigenproblem A(p) u = λ E(p) u.
Uses the same weak-form blocks as the legacy code for resonance extraction.
"""

"""Physical parameters for eigen assembly."""
struct EigenPhysicalParams
    θ::Float64
    nf::ComplexF64
    nm::ComplexF64
    ns::ComplexF64
    μ::Float64
    des_low::Float64
    des_high::Float64
    α::Float64
end

"""
    build_eigen_phys_params(pde, sim) -> EigenPhysicalParams

Build physical parameters for eigen assembly from EigenProblem + Simulation.
"""
function build_eigen_phys_params(pde::EigenProblem, sim::Simulation)
    nf = resolve_index(pde.env.mat_fluid, pde.λ_ref)
    nm = resolve_index(pde.env.mat_design, pde.λ_ref)
    ns = resolve_index(pde.env.mat_substrate, pde.λ_ref)
    des_low, des_high = getdesignz(sim.labels, sim.Ω)
    return EigenPhysicalParams(pde.θ, nf, nm, ns, 1.0, des_low, des_high, pde.α_loss)
end

"""
    assemble_eigen_matrices(pt, sim, phys) -> (A, E, ε_design, ε_bg)

Construct the block matrices for the generalized eigenproblem:
    A = [0  -K; M  -C]
    E = [N   0; 0   M]
"""
function assemble_eigen_matrices(pt, sim, phys::EigenPhysicalParams)
    ε₀ = x -> ε_background_wf(x, phys)
    √ε₀ = sqrt ∘ ε₀
    εₘ = (p -> ε_design_wf(p, phys)) ∘ pt
    k = 1.0

    K = assemble_matrix(sim.U, sim.V) do u, v
        ∫(shifted_curl_op(v, k, phys.θ) ⋅ shifted_curl_op(u, k, phys.θ))sim.dΩ
    end
    C = assemble_matrix(sim.U, sim.V) do u, v
        im * cosd(phys.θ) * ∫(v ⋅ (u * √ε₀))sim.dS_top +
        im * cosd(phys.θ) * ∫(v ⋅ (u * √ε₀))sim.dS_bottom
    end
    N = assemble_matrix(sim.U, sim.V) do u, v
        -1 * ∫(v ⋅ (ε₀ * u))sim.dΩ -
        ∫(v ⋅ (εₘ * u))sim.dΩ_design
    end
    M = I(size(N, 1))

    A = [spzeros(size(M)) -K; M -C]
    E = [N spzeros(size(M)); spzeros(size(K)) M]

    return A, E, εₘ, ε₀
end

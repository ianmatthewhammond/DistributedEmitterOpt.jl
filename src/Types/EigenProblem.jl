"""
    EigenProblem

PDE definition for the generalized eigenproblem A(p) u = λ E(p) u.
This describes WHAT to solve (environment + solver settings), not HOW.
"""
Base.@kwdef struct EigenProblem
    env::Environment
    θ::Float64 = 0.0                 # Incidence angle for ABC terms
    α_loss::Float64 = 0.0            # Artificial absorption
    shift::Float64 = 2π / 532.0      # Shift σ for shift-invert (k or ω)
    num_modes::Int = 6               # Number of eigenpairs
    krylovdim::Int = 110             # Krylov subspace dimension
    which::Symbol = :LM              # KrylovKit selector (e.g., :LM)
    λ_ref::Float64 = 532.0           # Reference wavelength for dispersion
end

"""Abstract type for eigen objectives."""
abstract type AbstractEigenObjective end

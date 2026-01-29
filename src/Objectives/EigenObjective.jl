"""
    EigenObjective

Objective for eigenvalue-based optimization.
"""

Base.@kwdef struct EigenObjective <: AbstractEigenObjective
    target::Union{Nothing,ComplexF64} = nothing
    mode_index::Int = 1
    weight::Float64 = 1.0
    kind::Symbol = :target  # :target, :maximize_real, :maximize_Q
end

"""Select the eigenvalue of interest from an EigenSolveResult."""
function eigen_value_of_interest(obj::EigenObjective, result::EigenSolveResult)
    return result.vals[obj.mode_index]
end

"""Compute objective value from an EigenSolveResult."""
function compute_eigen_objective(obj::EigenObjective, result::EigenSolveResult)
    λ = eigen_value_of_interest(obj, result)
    if obj.kind == :target
        if isnothing(obj.target)
            error("EigenObjective target is required for kind=:target")
        end
        return obj.weight * abs2(λ - obj.target)
    elseif obj.kind == :maximize_real
        return obj.weight * (-real(λ))
    elseif obj.kind == :maximize_Q
        Q = -real(λ) / (2 * imag(λ))
        return obj.weight * (-Q)
    else
        error("Unknown EigenObjective kind: $(obj.kind)")
    end
end

"""
    eigen_sensitivity(obj, result, pf, pt, sim, control; space=sim.Pf)

TODO: Implement eigenvalue sensitivity for generalized eigenproblems.
Should return ∂g/∂pf in the provided FE space.
"""
function eigen_sensitivity(::AbstractEigenObjective, ::EigenSolveResult,
    pf, pt, sim, control; space=sim.Pf)
    error("TODO: eigenvalue sensitivity not implemented")
end

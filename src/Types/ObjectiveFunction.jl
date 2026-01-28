"""
    ObjectiveFunction

Abstract type for optimization objectives. Implementations define how to
compute the objective value and gradients from field solutions.

## Required interface
- `compute_objective(obj, pde, fields, pt, sim)` — Objective value g
- `compute_adjoint_sources(obj, pde, fields, pt, sim)` — ∂g/∂E for each field
- `explicit_sensitivity(obj, pde, fields, pf, pt, sim, control)` — Explicit ∂g/∂pf

Fields is a Dict{CacheKey, CellField} mapping (λ, θ, pol) → E.
`pde` provides the input/output FieldConfig lists and weights.
"""
abstract type ObjectiveFunction end

# Interface stubs (implemented by concrete types)
function compute_objective end
function compute_adjoint_sources end
function explicit_sensitivity end

# ═══════════════════════════════════════════════════════════════════════════════
# Default implementations
# ═══════════════════════════════════════════════════════════════════════════════

"""Default: no explicit sensitivity term."""
explicit_sensitivity(::ObjectiveFunction, pde, fields, pf, pt, sim, control) =
    zeros(Float64, num_free_dofs(sim.Pf))

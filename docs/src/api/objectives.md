# Objectives

Objective function implementations for different optimization targets.

## SERS Objective

Surface-Enhanced Raman Scattering enhancement using SO(3)-averaged trace formula.

```@docs
SERSObjective
```

## SERS Utilities

Helper functions for SERS calculations:

- `α_invariants(αₚ)` — Compute SO(3) invariants for polarizability tensor
- `α_cellfields(αₚ, Ω)` — Build CellField constants for integration
- `α̂ₚ²` — The trace formula integrand
- `sumabs2` — Vector field magnitude squared
- `damage_factor` — Molecular quenching model
- `∂damage_∂E` — Derivative of damage factor

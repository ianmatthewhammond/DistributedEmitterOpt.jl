# Handoff: Gradient Debugging & Refactoring

**Date:** 2026-01-26
**Topic:** Fixing gradient computation and porting 2D/3D configuration patterns from `Emitter3DTopOpt` to `DistributedEmitterOpt.jl`.

## Context
We are debugging `test/gradient_tests.jl` failures. The core issues stem from differences in how `pt` (projected topology) and integration were handled in the old codebase vs the new one, specifically regarding Gridap's lazy evaluation of ComposedFunctions and `CellField` wrapping.

## Status Summary
- **Current State:** Tests are failing. Fundamental issues with `abs2` on vector fields and dimension mismatches in 2D backpropagation have been fixed, but high-level logic for `pt` creation and SERS formulation needs refactoring.
- **Goal:** Simplify SERS physics to always use the Anisotropic formulation and fix `pt` creation to match the 4 standard configurations (Foundry/Mesh × Smoothing/NoSmoothing).

## Changes Made So Far
1.  **Fixed `abs2` usage in `SERS.jl`**: Replaced `abs2(Ep)` with `Ep ⋅ conj(Ep)` because `abs2` does not define a gradient for Gridap `VectorValue` types.
2.  **Fixed 2D Sensitivity Dimensions**: Updated `assemble_material_sensitivity_2d` in `Gradients2D.jl` to use `sim.P` (piecewise constant per cell) instead of `sim.Pf` (Lagrange nodes). This ensures the output vector matches the dimensions required by `sim.grid.jacobian`.
3.  **Fixed 2D Explicit Sensitivity**: Updated the call to `explicit_pt_sensitivity` in 2D mode to add zeros of the correct size (`sim.P` dimensions).

## Immediate Next Steps

### 1. Refactor SERS.jl for Anisotropic-Only
**File:** `src/Physics/SERS.jl`
The user requested we remove isotropic special handling.
-   **Action:** Modify `compute_sers_objective` and `compute_sers_adjoint_sources` to **always** use the Anisotropic `α̂ₚ²` formulation.
-   **Reference:** See `get_gy` in `Emitter3DTopOpt/src/Objectives.jl` (lines ~212+).
-   **Details:** Even if `αₚ` is Identity, use the trace formula. Remove the `is_isotropic` branches.

### 2. Fix `pt` Creation Patterns (Critical)
The new code prematurely wraps `pt` in a `CellField`, breaking Gridap's lazy composition chain. We need to restore the 4 configuration patterns found in `Emitter3DTopOpt/src/Objectives.jl` (lines ~88-120).

**File:** `src/Optimization/Gradients2D.jl` (AmFoundry Mode)
-   **NoSmoothing:**
    -   Do **not** project to array first.
    -   Store `pf_vec` (filter only) in grid.
    -   Define `pf` as plain function: `r -> pf_grid(r...)`.
    -   Define `pt` as composition: `(pf -> Threshold(pf)) ∘ pf`.
-   **Smoothing (SSP):**
    -   Project array-wise `pf_vec` -> `pt_vec` using `smoothed_projection`.
    -   Store `pt_vec` in grid.
    -   Define `pt` as plain function interpolation of `pt_vec`.

**File:** `src/Optimization/Gradients3D.jl` (NotFoundry/Mesh Mode)
-   **NoSmoothing:**
    -   `pf = FEFunction(...)`
    -   `pt = (pf -> Threshold(pf)) ∘ pf`
-   **Smoothing (SSP):**
    -   `pt = ((pf,∇pf) -> smooth_project(...)) ∘ (pf,∇pf)` (Requires `∇pf` for gradient-aware smoothing).

### 3. Add Missing Helpers
-   Ensure `Threshold` (tanh projection function) is available for composition.
-   Ensure `pf_grid` handles `VectorValue` inputs if not already fixed.

## References
-   **Old Objectives Code:** `Emitter3DTopOpt/src/Objectives.jl` (Look at `getpft!` methods).
-   **Old Gradients:** `Emitter3DTopOpt/src/Gradients.jl` (Look at `get_adjoint` for Anisotropic).

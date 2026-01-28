# Handoff: Continue Debugging DistributedEmitterOpt.jl Gradient Tests

## Context Summary

You are continuing work on **DistributedEmitterOpt.jl** — a Julia package for SERS (Surface-Enhanced Raman Spectroscopy) topology optimization. This package was created by restructuring an existing working codebase called **Emitter3DTopOpt** into a cleaner module structure.

**THE CRITICAL ISSUE**: The new package compiles but gradient tests fail. The failures are likely due to code I (the previous AI assistant) wrote using my own creativity rather than carefully copying patterns from the user's working codebase. The user explicitly warned:

> "If there is a lot of code you wrote without inspiration from MY code, it's going to be buggy. This type of code is too specific to get wrong. Deep dive all the errors that might've stemmed from your own creativity and not adopting MY way of doing things."

---

## Repository Locations

- **New package**: `/Users/ianhammond/GitHub/DistributedEmitterOpt.jl`
- **Old working code**: `/Users/ianhammond/GitHub/Emitter3DTopOpt`

---

## Files You MUST Read First

### From the old working codebase (Emitter3DTopOpt):

1. **`/Users/ianhammond/GitHub/Emitter3DTopOpt/src/Physics.jl`** — Contains:
   - `∇x` curl operator (lines 18-23)
   - `ϵ0wf`, `ϵdwf` permittivity functions
   - `_MatrixA` weak form assembly (lines 98-111)
   - `MatA` matrix assembly function
   - How `εₘ = ((p -> ϵdwf(p; phys)) ∘ p)` is composed

2. **`/Users/ianhammond/GitHub/Emitter3DTopOpt/src/Objectives.jl`** — Contains:
   - `getpft!` functions (lines 103-122) — **CRITICAL**: Shows exactly how `pf` and `pt` CellFields are created:
     - **Foundry mode**: `pf = (r -> pf_grid(r; grid=sim.grid))` — plain function, not FEFunction
     - **3D mode**: `pf = FEFunction(sim.Pf, pf_vec)` — FEFunction
     - Then `pt = (pf -> Threshold(pf; control)) ∘ pf`

3. **`/Users/ianhammond/GitHub/Emitter3DTopOpt/src/Gradients.jl`** — Full adjoint gradient computation

4. **`/Users/ianhammond/GitHub/Emitter3DTopOpt/src/Simulations.jl`** — How simulation is built

5. **`/Users/ianhammond/GitHub/Emitter3DTopOpt/src/Controls.jl`** — Filtering and projection

### From the new codebase (DistributedEmitterOpt.jl):

1. **`/Users/ianhammond/GitHub/DistributedEmitterOpt.jl/src/Physics/Maxwell.jl`** — My Maxwell assembly (likely buggy)
2. **`/Users/ianhammond/GitHub/DistributedEmitterOpt.jl/src/Optimization/Gradients2D.jl`** — My 2D gradient code  
3. **`/Users/ianhammond/GitHub/DistributedEmitterOpt.jl/src/Optimization/Gradients3D.jl`** — My 3D gradient code
4. **`/Users/ianhammond/GitHub/DistributedEmitterOpt.jl/test/gradient_tests.jl`** — The failing tests

---

## What Was Fixed Already

1. ✅ Package compiles with Julia 1.11
2. ✅ Removed duplicate method definitions (`tanh_projection`, `plasmon_period`)
3. ✅ Fixed `@set` macro dependency → replaced with direct struct reconstruction
4. ✅ Fixed `getgrid` to use `Design` volume cells instead of broken `DesignNodes` physical group
5. ✅ Fixed `getdesignz` with same pattern
6. ✅ Fixed Materials.jl parser to handle `#` comments
7. ✅ Fixed 2D filter kernel dimension mismatch (conic_filter now takes explicit nx, ny)
8. ✅ Fixed Helmholtz filter to integrate over `dΩ` (full domain) instead of `dΩ_design`
9. ✅ Changed curl operator from `∇ × u` to `∇x ∘ ∇(u)` pattern

---

## Current Error

All 7 tests still fail with a Gridap integration error. The stack trace shows the error occurs in `assemble_maxwell` at line 108 during the integration of the curl-curl term.

The error appears to be in how the composed permittivity function `εₘ` is being used with Gridap's integration machinery.

---

## Key Differences I've Identified (Not Yet Fixed)

### 1. How `pt` CellField is Created (Line 34 in my Gradients2D.jl)

**My code:**
```julia
pt = CellField(x -> pf_grid(x, sim.grid), sim.Ω_design)
```

**Old code (Objectives.jl:105-106):**
```julia
pf = (r -> pf_grid(r; grid=sim.grid))  # Just a function, not CellField
pt = (pf -> Threshold(pf; control)) ∘ pf  # Composed function
```

The old code does NOT wrap in CellField — it passes the composed function directly to the permittivity composition `εₘ = ((p -> ϵdwf(p; phys)) ∘ p)` which is later evaluated during integration.

### 2. Domain Used in Design Terms

**Old code uses `sim.dΩ_d`** for design region integration.
**My code uses `sim.dΩ_design`** — this is the same but need to verify naming consistency.

### 3. Integration Pattern

**Old code (Physics.jl:105-107):**
```julia
∫((∇x ∘ (∇ₛ̃(v))) ⋅ (∇x ∘ (∇ₛ(u))))sim.dΩ -
(k^2) * ∫(v ⋅ (ε₀ * u))sim.dΩ -
(k^2) * ∫(v ⋅ (ϵₘ * u))sim.dΩ_d +
```

Note: The old code uses `ε₀ * u` not `ε₀(x) * u` — the multiplication is with the composed function directly.

---

## How to Run Tests

```bash
cd /Users/ianhammond/GitHub/DistributedEmitterOpt.jl
/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia --project=. test/gradient_tests.jl
```

---

## Your Task

1. **Read the old working code** thoroughly — especially Physics.jl, Objectives.jl (getpft! functions), and how `pt` is created/used
2. **Compare each function in my new code** against the old pattern
3. **Find where I deviated** and introduced bugs  
4. **Fix the issues** by adopting the exact patterns from the old code
5. **DO NOT HACK** — the goal is to make the tests pass by doing the same thing as the old code, not by working around errors

---

## Critical Reminders

1. The old code *works*. If something is different in my code, my code is probably wrong.
2. Gridap's FE assembly is very particular about how functions are composed and evaluated
3. The `∘` operator creates ComposedFunctions that Gridap evaluates lazily during integration
4. Using `CellField(...)` prematurely can break the evaluation chain
5. The order of arguments matters: `v ⋅ (ε₀ * u)` is different from `ε₀ * (v ⋅ u)`

# Mixed‑Output Gradient Bug (Cross‑Polarization Mapping)

**Status:** Active investigation / ready to implement fix

This document explains why gradients degrade (>0.1% relative error) for **mixed‑output** cases (multi‑output with mixed polarizations) and how to fix it. The short version: **we are using a forward projection in both forward and adjoint paths, but the adjoint requires the transpose of the forward operator**. This only appears when we cross‑map fields between x‑ and y‑polarized simulations.

---

## 1) Symptom Summary

### Failing gradient cases
- **3D Complex (Mixed Pol/Weights)**
- **3D Anisotropic + Multi Output**
- **2D Foundry Anisotropic + Multi Output**

### Passing cases
- Elastic / single‑output cases (one polarization)
- Anisotropic elastic (single polarization)

### Why this pattern matters
All failing cases require **cross‑polarization mapping** (x<->y) using `SimulationBundle`. Passing cases stay in a **single sim** (no mapping), so the bug path is not exercised.

---

## 2) Where the bug lives

### A) Mapping between sims currently uses `interpolate`

```julia
# src/Types/SimulationBundle.jl
function map_field(field, sim::Simulation, space::Symbol)
    if space == :U
        return interpolate(field, sim.U)
    elseif space == :Pf
        return interpolate(field, sim.Pf)
    else
        error("map_field: unsupported space $space")
    end
end
```

This creates a **projection** into the target U space, which implicitly applies that sim’s boundary constraints (dirichlet tags differ for x vs y). This is a **linear operator T**.

### B) SERSObjective uses this mapping in *both* forward and adjoint

```julia
# src/Objectives/SERSObjective.jl (forward)
Ep_out = sim_out === sim_in ? Ep : map_field(Ep, sim_out, :U)
```

```julia
# src/Objectives/SERSObjective.jl (adjoint)
Ee_in  = sim_out === sim_in ? Ee : map_field(Ee, sim_in, :U)
```

The adjoint path **reuses the forward map**, but mathematically it should use the **transpose** (adjoint) operator **T'**. Using **T** in both directions breaks adjoint consistency.

---

## 3) Why this only shows up in mixed outputs

- If `sim_out == sim_in`, mapping is skipped and the adjoint is correct.
- Mixed outputs require `sim_out != sim_in` (x vs y), so `map_field` is used.
- The error does not appear in single‑pol scenarios.

This explains why anisotropy alone is not the cause — the issue is **cross‑sim mapping**, not the tensor itself.

---

## 4) Correct Fix (Adjoint‑Consistent Transfer)

### Goal
Replace `map_field` for **U fields** with a **pair** of operators:

- **Forward transfer:** `T : U_src -> U_dst`
- **Adjoint transfer:** `T' : U_dst -> U_src`

### What we need to implement
1. Build (or assemble) a transfer operator `T` between U spaces in `SimulationBundle`.
2. Add `map_field_forward` and `map_field_adjoint` helpers.
3. Use **forward map** in `compute_objective` and **adjoint map** in `compute_adjoint_sources`.

---

## 5) Proposed Implementation Sketch

### A) Extend SimulationBundle with transfer matrices

```julia
# src/Types/SimulationBundle.jl
struct SimulationBundle
    default::Simulation
    by_pol::Dict{Symbol,Simulation}
    transfer_U::Dict{Tuple{Symbol,Symbol}, SparseMatrixCSC{ComplexF64,Int}}
end
```

### B) Build transfer matrix once

```julia
# build_simulation_bundle(...)
T_yx = build_transfer(sim_y.U, sim_x.U)
T_xy = build_transfer(sim_x.U, sim_y.U)
transfer = Dict((:y,:x)=>T_yx, (:x,:y)=>T_xy)
SimulationBundle(sim_y, by_pol, transfer)
```

### C) Forward/Adjoint map helpers

```julia
function map_field_forward(field, sim_src::Simulation, sim_dst::Simulation, T)
    u_src = Gridap.FESpaces.get_free_values(field)
    u_dst = T * u_src
    return FEFunction(sim_dst.U, u_dst)
end

function map_field_adjoint(field, sim_src::Simulation, sim_dst::Simulation, T)
    u_dst = Gridap.FESpaces.get_free_values(field)
    u_src = T' * u_dst
    return FEFunction(sim_src.U, u_src)
end
```

### D) Update SERSObjective

```julia
# Forward objective
Ep_out = sim_out === sim_in ? Ep : map_field_forward(Ep, sim_in, sim_out, T_in_to_out)

# Adjoint sources (pull output contribution back)
Ee_in  = sim_out === sim_in ? Ee : map_field_adjoint(Ee, sim_in, sim_out, T_in_to_out)
```

---

## 6) How to build the transfer matrix T

### Option A: Gridap interpolation operator (preferred)
If Gridap provides a direct interpolation matrix between U spaces:

```julia
T = interpolation_matrix(sim_src.U, sim_dst.U)
```

### Option B: L2 projection operator (robust)
Assemble a mixed mass matrix and project:

- `M_dst = ∫ v·u dΩ` (dst space mass)
- `B = ∫ v·u_src dΩ` (mixed mass)

Then:

```
T = M_dst \ B
```

The adjoint is then `T'`.

---

## 7) Why this should fix gradients

The adjoint uses **T'**, the transpose of the forward map. This restores mathematical consistency:

```
Forward:  u_out = T u_in
Adjoint:  v_in  = T' v_out
```

With that change, the mixed‑output gradient should return to the same accuracy as single‑output cases.

---

## 8) Tests to confirm

### Required regression checks
- `test/gradient_tests.jl`:
  - **3D Complex (Mixed Pol/Weights)**
  - **3D Anisotropic + Multi Output**
  - **2D Foundry Anisotropic + Multi Output**

### Quick debugging checks
1. Force all outputs to `pol=:y` and confirm gradient error collapses.
2. Re‑enable mixed outputs and confirm error is gone.

---

## 9) Additional notes

- Scalar fields (`Pf`, `pt`) can still use the existing `map_field` (interpolate), because those do not introduce cross‑sim U‑space issues.
- This issue is confined to **U‑field mapping** (vector field with boundary constraints).
- If we ever move to a **single simulation object** for both polarizations, the mapping can be eliminated entirely. That’s a larger architectural refactor but would simplify gradients.

---

## 10) Code References

- **Mapping helper:**
  - `src/Types/SimulationBundle.jl` → `map_field`

- **Objective / adjoint:**
  - `src/Objectives/SERSObjective.jl`
    - `compute_objective`
    - `compute_adjoint_sources`

- **Simulation builder:**
  - `src/Types/SimulationBundle.jl` → `build_simulation_bundle`

---

## 11) Next Steps Checklist

- [ ] Implement transfer operator `T` between U spaces.
- [ ] Add `map_field_forward` and `map_field_adjoint`.
- [ ] Replace cross‑sim `map_field(..., :U)` in SERSObjective.
- [ ] Re‑run mixed‑output gradient tests.
- [ ] If still failing, inspect explicit sensitivity path for any remaining cross‑sim U mapping.

---

If you want, I can implement the above and update the gradient tests to confirm the fix.

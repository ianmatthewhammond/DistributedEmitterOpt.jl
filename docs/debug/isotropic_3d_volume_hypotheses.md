# Isotropic 3D Volume Debug — Hypotheses & Experiments

Context: This compares `scripts/experiments/isotropic_3d_volume/debug_isotropic_3d_volume.jl` (new repo) against `scripts/experiments/isotropic_3d_volume_old/debug_isotropic_3d_volume_old.jl` (legacy repo). The new repo’s objective and gradients appear wrong. Below are the likely discrepancy points and brief experiments to validate each.

---

## Highest‑confidence issues (likely breaking objective *and* gradients)

### 1) Maxwell LU cache not invalidated when design changes
**Why suspicious**
- New `solve_forward!` only assembles/factorizes if `!has_maxwell_factor(cache)`.
- There is no call to `clear_maxwell_factors!` or `invalidate_maxwell_cache!` inside `objective_and_gradient!` or its callers.
- The gradient check calls `objective_and_gradient!` twice with different `p` but (likely) the same cached LU from the first call.

**Expected symptom**
- Forward solves are effectively frozen after the first evaluation (stale fields), so FD check fails and objective is wrong during optimization.

**Experiment**
- In the debug script, before *each* call to `objective_and_gradient!`, add:
  - `clear_maxwell_factors!(prob.pool)`
- Re-run the directional derivative test. If it passes or greatly improves, this is the root cause.

**TODO**
- Replace “always clear” with design-aware invalidation (hash or version) so LU reuse is safe across unchanged designs.

---

## Medium‑confidence discrepancies (affect objective comparisons vs legacy)

### 2) SSP radius mismatch (new vs old defaults)
**Why suspicious**
- Old: `R_s = 0.55 * R_f[1]` by default when `subpixel=true`.
- New: `Control.R_ssp` defaults to `11.0` and SSP is always used.

**Expected symptom**
- Objective values differ significantly between repos even if gradients are internally consistent.

**Experiment**
- In the new debug script, set `control.R_ssp = control.R_filter[1] * 0.55` and compare objective values against old.

### 3) Complex index sign convention (n + i k vs n − i k)
**Why suspicious**
- Old uses `n - i k` from material data.
- New uses `n + i k`.

**Expected symptom**
- Objective differs between repos, possibly large changes in field magnitude.

**Experiment**
- Force a purely real constant material index (e.g., `mat_design = 0.2`) in *both* repos and compare objective/gradients.

---

## Likely issues for inelastic or multi‑config cases

### 4) Emission adjoint RHS ordering mismatch
**Why suspicious**
- New emission RHS uses `α̂ₚ²(v, Ee', Ep, Ep, ...)`.
- Old uses `α̂ₚ²(Ee, v, Ep, Ep, ...)`.

**Expected symptom**
- Inelastic gradients fail or are inconsistent when λ_emission ≠ λ_pump.

**Experiment**
- Run an inelastic test (λ_emission ≠ λ_pump). If it fails, swap the argument order in new code to match old and re‑check.

### 5) Pump/emission selection by sorting keys
**Why suspicious**
- New code sorts dict keys and assumes first = pump, last = emission.
- This can break when λ_emission < λ_pump or with multiple configs.

**Expected symptom**
- Objective/adjoint mapping swaps pump and emission in some setups.

**Experiment**
- Set λ_emission < λ_pump and compare objective vs old. If it flips, map explicitly via `pde.inputs`/`pde.outputs`.

### 6) FieldConfig weights not applied consistently
**Why suspicious**
- Weights are applied in `pde_sensitivity` but not in `compute_objective` or `compute_adjoint_sources`.

**Expected symptom**
- Weighted cases fail FD checks (objective and gradient inconsistent).

**Experiment**
- Set a non‑unit weight and run FD check; if it breaks only then, apply weights consistently in objective + adjoint sources.

---

## Behavioral mismatches (less likely to break gradients by themselves)

### 7) SSP always applied in new code
**Why suspicious**
- `project_ssp` is always used, ignoring `use_ssp/use_projection` flags.
- Old code could disable SSP when `subpixel=false`.

**Expected symptom**
- New and old objective/gradient differ even when controls are intended to be “no SSP.”

**Experiment**
- Match old settings with `subpixel=false` and check if new still applies SSP (likely yes). If so, add a conditional path to bypass SSP.

---

## If I had to bet (for this specific test)
The Maxwell LU cache invalidation is the most likely root cause for the **incorrect objective and gradient** in the new repo. It directly breaks FD checks and would corrupt the optimization loop.

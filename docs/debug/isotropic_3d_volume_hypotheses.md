# Isotropic 3D volume debug -- hypotheses

Comparing `scripts/experiments/isotropic_3d_volume/debug_isotropic_3d_volume.jl` (new repo) against the legacy repo version. The new repo's objective and gradients look wrong. Here's what might be going on and how to check each one.

---

## Most likely (breaks both objective and gradients)

### 1. Maxwell LU cache not invalidated when design changes

`solve_forward!` only assembles/factorizes when `!has_maxwell_factor(cache)`, but nothing clears those factors between calls to `objective_and_gradient!`. So the gradient check calls it twice with different `p` but gets the same stale fields from the first call.

To test: call `clear_maxwell_factors!(prob.pool)` before each `objective_and_gradient!` call and re-run the FD check. If it passes, this is the bug.

Follow-up: replace the "always clear" workaround with design-aware invalidation (hash or version counter) so LU reuse still works when the design hasn't changed.

---

## Likely (affects objective values vs. legacy, not necessarily gradients)

### 2. SSP radius mismatch

Old code: `R_s = 0.55 * R_f[1]` when `subpixel=true`. New code: `Control.R_ssp` defaults to `11.0` and SSP is always on.

To test: set `control.R_ssp = control.R_filter[1] * 0.55` in the new script and compare.

### 3. Complex index sign convention

Old code uses `n - ik`, new uses `n + ik`. Could cause large field-magnitude differences.

To test: force a purely real material index (e.g. `mat_design = 0.2`) in both repos and compare.

---

## Relevant for inelastic / multi-config cases

### 4. Emission adjoint RHS argument order

New code passes `alpha_p2(v, Ee', Ep, Ep, ...)`, old code passes `alpha_p2(Ee, v, Ep, Ep, ...)`.

To test: run an inelastic case (lambda_emission != lambda_pump). If it fails, swap the argument order.

### 5. Pump/emission selection by sorting keys

New code sorts dict keys and assumes first = pump, last = emission. Breaks when lambda_emission < lambda_pump.

To test: set lambda_emission < lambda_pump and compare.

### 6. FieldConfig weights applied inconsistently

Weights show up in `pde_sensitivity` but not in `compute_objective` or `compute_adjoint_sources`.

To test: use a non-unit weight and run the FD check; if it only breaks then, apply weights in all three places.

---

## Minor (probably not the root cause on its own)

### 7. SSP always applied even when flags say otherwise

`project_ssp` runs unconditionally; old code could skip it with `subpixel=false`.

To test: match old settings with `subpixel=false` and check whether new code still applies SSP.

---

## Best guess for this specific test

The LU cache invalidation issue (#1) is the most likely root cause. It directly breaks the FD check and would corrupt the optimization loop.

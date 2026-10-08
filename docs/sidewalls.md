# Linear sidewalls and robust optimization

A foundry design has two-dimensional density parameters. In `foundry_projection_mode=:current`,
`Control(use_sidewall=true, sidewall_slope=s, sidewall_z0=z0)` makes its FEM projection
vary with physical height:

```text
rho_sw = rho_filtered - |grad rho_filtered| * s * (z - z0)
```

`sidewall_slope` is lateral displacement per unit height. For an angle `alpha` from vertical,
`s = tan(alpha)`; `sidewall_samples(angles_deg=..., z0s=...)` instead accepts angles from
horizontal, so a vertical wall is 90 degrees. A positive slope tapers solid inward as `z`
increases. `sidewall_z0` is the physical pivot height, where the nominal design is preserved.
Obtain the design-region bounds with `getdesignz(sim.labels, sim.Ω)` and choose the bottom,
top, or midpoint deliberately. Zero slope uses the original vertical projection exactly.
The sidewall path requires a triangulation and the current FEM foundry projection; legacy
grid projection does not carry the physical-height information.

The `R_ssp` projection smoothing radius and `R_filter` density-filter radius encode different
operations. The filter and `eta_erosion`/`eta_dilation` thresholds define the grid linewidth
constraints; they do not automatically certify every slice of the fabricated body. For
positive slope about the mid-height of a layer of height `H`, the normal-distance model
reduces solid width at the top and void gap at the base by `s*H` relative to the pivot slice.
A target width or gap `ell` therefore requires pivot dimensions of at least `ell + s_max*H`
for the selected slope range. Calibrate the filter-radius/threshold relationship for the
chosen filter and grid, then verify the binary top solid and base gap. Finite-beta results
are training geometries; inspect the beta-infinity projection before exporting a design.

## Worst-case aggregation

Pass `robust_samples=[(slope, pivot), ...]` to `optimize!`. Every sample receives a full
objective and adjoint evaluation. The chosen set bounds only those sampled profiles;
it does not prove a bound for all angles between them.

- `robust_mode=:ks` uses a smooth lower approximation to the minimum sample objective.
  `robust_sharpness` controls its sharpness, divided by the fixed `g_norm` normalization.
  This scale must stay independent of the sample values for the analytic gradient to be valid.
  KS works with the NLopt and standalone backends.
- `robust_mode=:epigraph` introduces `tau` and one inequality per sample, and minimizes
  `-tau`. It requires `backend=:standalone_mma` or `:standalone_ccsaq`. The seed scale `T`
  stays fixed within an epoch. `epigraph_sigma` sets the initial step scale of `tau`, and
  `epigraph_tol` sets the sample-constraint tolerance.
- `use_constraints=true` enables the existing solid/void grid linewidth constraints.
  In epigraph mode use `constraint_scale=1/control.b1`; `epigraph_linewidth_tol` then admits
  `M <= b1*(1+epigraph_linewidth_tol)`. Grid feasibility still needs a body-slice check.

The aggregate uses the selected objective, including band, volume, or `surface_mode=:tv_void`
terms. The surface measure is independently configurable; see [surface objectives](surface_objectives.md).

## Standalone library and checkpoints

Build [NLopt-MMA-CCSA](https://github.com/ianmatthewhammond/NLopt-MMA-CCSA) with support for
`NLOPT_LD_MMA` in its dual solver and the committed-checkpoint callback arguments of
`mma_minimize`/`ccsa_quadratic_minimize`. Set `JULIA_NLOPT_MMACCSA_LIB` to the resulting shared
library. Merely finding a library file does not establish ABI compatibility. The wrapper
probes dual-solver support and refuses older builds that reject the default `dual_algorithm=:mma`.
`dual_algorithm=:lbfgs` explicitly selects the older dual solver and still requires the callback ABI.
The NLopt backend does not need this shared library.

```sh
git clone --branch public/portable-parity-paths https://github.com/ianmatthewhammond/NLopt-MMA-CCSA.git
cmake -S NLopt-MMA-CCSA -B NLopt-MMA-CCSA/build
cmake --build NLopt-MMA-CCSA/build
export JULIA_NLOPT_MMACCSA_LIB="$PWD/NLopt-MMA-CCSA/build/libmmaccsa.dylib"
# On Linux, use libmmaccsa.so instead.
```

The build commands select the parity-portable branch, which includes the MMA dual and
committed callback implementation. A release or main branch containing the same features
can also be used.

An explicit library path is authoritative: a missing file produces a build/setup error
instead of falling back to another checkout. Without an explicit path, the wrapper also
checks the conventional `~/GitHub` and `~/github` checkout locations. If none exists,
standalone optimization stops with instructions to build the library or set the variable.

Checkpoint metadata records the continuation schedule, backend, robust mode and sample set,
KS sharpness or epigraph tolerances, and epigraph `T`/`tau`. A mid-epoch resume also needs the
associated serialized C state. Save standalone state inside the committed checkpoint
callback, paired with its design and evaluation count; an evaluation-limit exit can
interrupt an uncommitted inner trial, which a resume must replay. Mismatched configurations or variable/constraint counts are
refused. Legacy design-only checkpoints retain their older resume heuristic. Checkpoints
store the committed design as `p`; `p_best` tracks objective alone and can be infeasible
when linewidth constraints are active.

## Example and checks

Run the small generic example with `julia --project scripts/examples/robust_sidewall.jl`.
Pass `epigraph` as its argument to select the standalone backend.

```sh
julia --project test/runtests.jl
julia --project test/test_mmaccsa_backend.jl
julia --project test/test_backend_parity.jl
```

The sidewall suite includes finite-difference gradients, a vertical-wall control, KS weights,
epigraph cache controls, committed checkpoint/resume comparisons, and named mismatch refusals.
The full suite needs the compatible standalone library for its epigraph tests. Set
`DEO_TEST_LEGACY_MMACCSA_LIB` to a pre-MMA-dual build to exercise the optional refusal test.

# General-angle illumination and collection

`FieldConfig` accepts `tilt_x` and `tilt_y` in degrees. They specify direction
cosines sx=sin(tilt_x), sy=sin(tilt_y), not successive Euler rotations. The
physical downward ray is (sx,sy,-sqrt(1-sx²-sy²)); grazing/nonpropagating pairs
are rejected. `n_incident` is the lossless fluid index (air by default) and must
match the environment. The time convention is exp(+iωt), material index n−ik,
and physical E=envelope exp(−i k_parallel·r). The trial curl has the negative
Bloch shift and the bilinear test curl has the positive shift. Oblique matrices
are generally not complex symmetric (A(κ)ᵀ = A(−κ)), so adjoint solves must be true
Aᴴ solves. UMFPACK does this, and MUMPS and Pardiso solve Aᵀ conj(x) = conj(b) on
the same factorization (see [Solver backends](#solver-backends)).
Symmetric storage (Pardiso symmetric mtypes, MUMPS `symmetric=true`) checks the
matrix before discarding a triangle and refuses incompatible operators.

The MUMPS extension (MUMPS.jl 1.x) runs the sequential MUMPS build, which needs no
MPI, when MUMPS.jl has a `Sequential` backend (1.7.1 does), and MUMPS on one MPI
rank (initializing MPI) otherwise, as with 1.6.1. It drives the MUMPS job itself,
because MUMPS.jl's solve call returns without solving on an instance that has solved
already: every right-hand side runs a solve, and new values on an unchanged sparsity
pattern reuse the analysis. Each phase checks INFOG(1), and a factorization that
runs out of workspace is retried with ICNTL(14) doubled. MUMPS memory lives outside
Julia's GC, so factors are released explicitly (and before MPI is finalized at
exit): a solver pool dropped without `release_factor!` or `empty!` keeps its MUMPS
factors until the process exits. Output is limited to errors unless `MUMPSSolver(verbose=true)`.

Polarization `:x` projects global x onto the transverse plane and normalizes it;
`:y` completes the orthonormal basis. Thus x-plane tilt maps :x/:y to p/s, and
y-plane tilt maps them to s/p. Two-axis tilts have the documented Cartesian
transverse basis rather than a silently selected s/p basis. Legacy `θ` still
means y tilt. Old normal calls retain the legacy matrix/source path exactly;
nonzero angles and explicit tilt keywords use the corrected formulation.
Previous nonzero-θ numbers are deliberately corrected: the old operator had
incorrect dispersion and boundary admittance. The low-level legacy
`PhysicalParams` constructor refuses nonzero angles instead of silently using it.
Mixed-polarization objective values can change even at normal incidence: the
shared-model vector/material mapping correction removes projections that altered
the forward fields and material while using inconsistent derivatives.

## Cells and design degrees of freedom

`build_angular_simulation(geo, configs; meshfile, design_shape=(17,17))` chooses
periodic axes from all field configurations. Normal fields use two mirror axes
(quarter cell). One tilt axis uses a half cell, periodic along that axis and
PEC/PMC across the other according to polarization. Both tilt axes require a
full periodic cell. Any nonzero tilt across a mirror errors with the axis and
boundary name. Mesh periodic links are inspected rather than trusting flags.

Geometry L/W are widths of the independent mirror patch, which is half the
lattice period: the lattice repeats every 2L × 2W, and a periodic direction has
twice the patch width as its field width. All fields share one model. A weighted collection
containing x and y tilts therefore uses the full union domain, including its
normal pump. It does not yet give each tilt its own half-cell factorization.
The grid still contains only design_shape independent 2D DOFs; material values
are reflected into the enlarged field domain, and their adjoints scatter back
through the same reflection. There is no diagonal x/y exchange constraint.
Objectives and gradients are divided by the domain multiplicity (2 or 4), so
quarter/half/full values have the same independent-patch normalization.

The new builder defaults to a structured hexahedral mesh with matched mirror
intervals. `structured=false` keeps independent unstructured meshing; its
finite-mesh symmetry error must then be checked by refinement. The builder
always tags the CAD curves and points on the mirror walls (`genmesh(...;
mirror_curves=true)`), so a PEC wall constrains every tangential degree of
freedom on it for either kind of mesh. Existing `genmesh` calls retain their
original tetrahedral behavior. Each generated mesh gets a recipe file
(`meshfile * ".recipe"`) with its genmesh inputs and the SHA-256 of the mesh file;
`reuse_mesh=true` refuses a mesh whose recipe differs from the one this call would
write or whose file no longer has the recorded digest, and checks the extents,
periodic links and mirror-wall tags the mesh itself shows. An older mesh without a
recipe gets the extents and periodic-link checks, with warnings.

## Collection sampling and solve counts

`rim_collection_samples(na; normal_weight=1, rim_weight=1)` gives normal and
four signed rim directions at asin(NA/n_incident). `collection_configs(λ,samples)`
expands every (tilt_x,tilt_y,weight) tuple into both polarizations. Weights are
inputs per polarization; this five-point rule is not a demonstrated quadrature
error bound for a continuous cone. A pupil annulus or another quadrature only
changes the input list.

At equal pump and collection wavelengths, one normal linear pump plus both
collection polarizations at normal and four signed rim points gives 10 unique
field solves. `reduce_mirror_samples(...;mirror_x=true,mirror_y=true)` combines
signed weights, giving 6 factorizations: two normal and four tilted. This
reduction requires symmetry of the complete pump/objective, not just geometry.
A reflected design, fixed normal linear pump, orientationally averaged SERS
tensor, and intensity-only damage satisfy those mirrors. A fixed polarized
pump generally breaks diagonal exchange even if the design has that symmetry.
Without normal collection the count is 5. Cache keys include both tilts, index,
polarization and formulation, and exclude objective weights.
These counts use the corrected formulation for the pump too:
`FieldConfig(λ; tilt_x=0, pol=:y)`. An old-style normal pump defaults to the
legacy formulation and retains a distinct cache entry; mixing it with corrected
normal collection adds one solve. Use explicit tilt keywords consistently for
a new collection problem.

## Absorbing boundaries

For the specular order in medium ε, q=sqrt(k0²ε−|k_parallel|²), with the outgoing
branch for exp(+iωt). Tangential admittance is q for TE and k0²ε/q for TM. The
implemented tensor is Y=q I_t+k_parallel k_parallelᵀ/q; normal electric field
is excluded from this surface term. A sheet source iY·E_t produces a downward
incident electric-field amplitude of 1/2. In air that matches the old normal convention;
in a fluid of index nf the old normal source gives 1/(2nf), so absolute fields differ by nf
between the two formulations there (ratios to a flat reference under the same formulation
do not).

The old scalar k0 n cosθ is incorrect for refracted substrate waves and for TM
even in air. For n=1.8 and θ=30°, q/k0=1.7292 versus 1.5588 from the old scalar;
TM admittance/k0=1.8737. In air TE/k0=.8660, TM/k0=1.1547. Comparing tangential
admittances alone gives 2.04% reflected power for an air TM wave against the old
scalar boundary. This is not the complete error of the old operator/source.

This local boundary is exact for the specular plane wave. A diffraction order
with reciprocal vector G has a different q and tensor; those orders are not
absorbed exactly by the local specular tensor. Evanescent orders need sufficient
homogeneous buffer to decay. Propagating nonspecular orders need a modal DtN or
PML treatment for quantitatively reliable power. Uniform-film agreement does
not establish that error for patterned structures. Bloch field configurations
therefore check the lattice when their parameters are built
(`check_diffraction_orders`): an order (m,n) ≠ (0,0) that propagates in the
incidence medium, or in a lossless or low-loss substrate, is an error that
names the order and the periods below which every order is evanescent. An order
at or near cutoff (|q| ≤ 10⁻³ k|n|, a decay length above about 160 wavelengths)
counts as propagating. A PEC bottom has no admittance term, so substrate orders
are not checked there. A warning is issued when the slowest evanescent order
still has amplitude above 1e-2 at a checked face.

## Validation

Run `test/oblique_fresnel_tests.jl` for independent transfer-matrix R/T comparisons
(both axes and polarizations, 10° and 30°, mesh refinement), including the signed
normal electric field fitted inside the substrate. H(curl) elements control the
tangential boundary trace; normal boundary traces are unsuitable for that check.
`test/oblique_film_reference_tests.jl` adds a free-wavenumber fit and complex
reflection-vector gate on dielectric and lossy films, including two-axis tilts,
source normalization and specular boundary returns.

`test/oblique_gradient_tests.jl` checks foundry tv_void step ladders for x/y,
both polarizations, two-axis full cells and a weighted signed collection, with
damage off/on. `test/oblique_negative_control_tests.jl` deliberately uses an
unconjugated adjoint and must fail the gradient criterion.
`test/oblique_symmetric_adjoint_control_tests.jl` also rejects the old shortcut
that assumes a complex symmetric operator. `test/oblique_solver_guard_tests.jl`
checks matrix symmetry and refusal of incompatible Pardiso storage modes.
`test/oblique_symmetry_tests.jl` checks transpose, signed angles, and half/full
cells. `test/backend_adjoint_tests.jl` checks forward and Aᴴ residuals of every
solver backend that loads on non-symmetric complex matrices, in forward, adjoint,
forward sequences before and after a refactorization; `test/mumps_gradient_tests.jl`
(slow, skipped without MUMPS.jl) compares MUMPS objectives and gradients with UMFPACK.
`test/oblique_order_guard_tests.jl` covers the diffraction-order guard,
`test/oblique_mesh_reuse_tests.jl` the mesh recipe, and
`test/oblique_normal_equivalence_tests.jl` shows that corrected normal incidence in
air differs from legacy θ = 0 only by the normal-trace boundary term. The film
reference also covers a lossy metal substrate. `test/runtests.jl` runs these files;
`DEO_SLOW_TESTS=1` adds the gradient, symmetry and MUMPS suites. `test/mixed_output_gradient_tests.jl` covers the corrected shared-model
vector/material mapping. `scripts/benchmark_angular_solves.jl` compares local
sequential/concurrent factorization; no distributed scheduling layer is added.

Primary background: [Bloch envelopes](https://mpb.readthedocs.io/en/latest/Developer_Information/),
[Maxwell weak curl](https://mfem.org/fem_weak_form/),
[vector mirror symmetry](https://meep.readthedocs.io/en/latest/Exploiting_Symmetry/),
and [oblique Fresnel validation](https://doc.comsol.com/6.3/doc/com.comsol.help.models.rf.fresnel_equations/fresnel_equations.html).

The eigenmode problem is not extended here. Nonzero eigenmode angles now refuse
explicitly; the old generalized eigenproblem cannot represent this corrected
angle-dependent frequency-domain operator by silently reusing its old shift.

## Solver backends

`UmfpackSolver()` runs everywhere and is the default. `MUMPSSolver()` needs MUMPS.jl
(and MPI), `PardisoSolver()` needs Pardiso.jl and MKL, and `CUDSSSolver()` needs
CUDSS.jl and an NVIDIA GPU. Each solves A x = b and the adjoint Aᴴ x = b on one
factorization, so all four are valid for oblique fields.

`PardisoSolver` drives MKL Pardiso through Pardiso.jl; `import Pardiso` loads the
extension. Pardiso.jl exports its own `PardisoSolver`, so after `using Pardiso`
write `DistributedEmitterOpt.PardisoSolver`. MKL is not available on every platform
(not on Apple Silicon, for example); there the first factorization reports it, and
`UmfpackSolver()` is the fallback. Pardiso.jl uses MKL_jll by default. A Pardiso.jl
built against a local MKL (`MKLROOT` set at build time) needs `MKLROOT` and MKL's
library directory on `LD_LIBRARY_PATH` at run time, and `MKL_THREADING_LAYER=GNU`
when other libraries in the process use GNU OpenMP. `nprocs > 0` sets MKL Pardiso's
thread count, a process-wide MKL setting; the default 0 keeps MKL's own
(`MKL_NUM_THREADS`, or all cores).

Pardiso reads Julia's CSC arrays as CSR, that is, it sees Aᵀ. The Maxwell type is
chosen for each factorization, on the full matrix. A matrix verified complex
symmetric (‖A − Aᵀ‖_F ≤ 10⁻¹² ‖A‖_F, which holds at normal incidence, also for the
normal emission of a tilted pump) gets mtype 6, which keeps one triangle. Any other,
such as an oblique Bloch operator, gets mtype 13 in full storage. With mtype 13 the
forward solve is Pardiso's transposed solve and the adjoint is its plain solve on
conj(b), conjugated; with mtype 6 both are plain solves, and the conjugated one is
the exact adjoint of the stored symmetric operator. That is the matrix itself to within the
10⁻¹² symmetry tolerance it was accepted under. `assume_symmetric_maxwell` keeps its meaning when it is given:
`true` forces mtype 6 (refused for a matrix that is not complex symmetric) and
`false` forces mtype 13. Its default is now `nothing`, the per-factorization choice;
it used to be `false`. The Helmholtz filter uses mtype -2 by default; 2 and 11 are
also accepted, and the symmetric types check the matrix first. The analysis is
reused while the sparsity pattern matches the one it ran on, so a design update
costs one numerical factorization. The legacy iparm profile applies to
full-storage Maxwell factorizations, and mtype 6 keeps MKL's defaults; a custom
`iparm` must keep iparm(1) = 1, iparm(6) = 0, iparm(28) = 0 and iparm(35) = 0,
and iparm(12) is set by the backend for each solve.

`CUDSSSolver` factors on an NVIDIA GPU with cuDSS through CUDSS.jl 1.0 or later;
`using CUDSS` loads the extension. Matrices are built on the host as before and copied
to the device in CSR form for each factorization. As for Pardiso, the structure is
chosen per factorization on the full matrix: a complex symmetric matrix (the same
10⁻¹² test) is factored as LDLᵀ from its lower triangle (cuDSS structure "S", which
for complex values is the transpose, not the Hermitian, factorization: checked with a
matrix that is transpose symmetric but not Hermitian), and any other matrix as a
general LDU ("G"). The adjoint of "S" is the plain solve on conj(b), conjugated; the
adjoint of "G" uses cuDSS's conjugate-transpose solve mode on the same factor.
`assume_symmetric_maxwell` has the same three settings as for Pardiso. The analysis is
reused while the stored pattern is unchanged. After each factorization cuDSS's `info`
is checked, and after each solve the residual of the intended operator, A or Aᴴ built
from the full matrix, is computed on the device in double precision; up to
`refine_steps` (3) refinement steps run if it exceeds `residual_rtol` (10⁻⁹), and a
solve that still misses it throws. Every factor is built, in the CUDA context current at
its creation, on a cuDSS handle owned by the extension; all cuDSS calls are serialized by one
lock and run, with the uploads, residuals and frees around them, on one CUDA stream the
extension owns per context (cuDSS keeps the stream a factor's analysis ran on), so a factor may
be used from any task. Each factor's cuDSS objects are held until they are destroyed under that lock:
`release_factor!` destroys them and frees the factor's device arrays at once, and a factor
dropped without release is destroyed at a later call after the GC collects it. Options are
checked when a factor is built and at every solve (`residual_rtol` finite and ≥ 0,
`refine_steps` ≥ 0), and a right-hand side must match the factor's size.

On a 435,606-unknown lowest-order Nédélec quarter cell (6.9 M nonzeros) a numerical
refactorization with reused analysis took 0.34 s ("S") and 0.49 s ("G") on an NVIDIA
B200 and 4.1 s / 7.4 s on a V100, against 5.6–6.5 s (mtype 6) and 8.9–11.2 s
(mtype 13) for MKL Pardiso on 32 cores of the B200 node's host; solutions agreed with
Pardiso to 10⁻¹². GPUs of compute capability 7.0 (V100) are not supported by CUDA 13:
pin the CUDA 12 runtime there with `CUDA.set_runtime_version!(v"12.9")`. On clusters
whose GPU nodes have no internet access, install and precompile CUDA.jl and CUDSS.jl
where there is, since CUDSS_jll downloads its library on first use.
`CUDSSSolver(precision=:single)` stores the values of "S" factors in single precision (the
factor values take half the memory; the full double matrix kept for residuals, the indices and
cuDSS's workspace do not shrink). Every solve starts from that factor and is refined in double
precision against the full matrix toward the tighter of `refine_rtol` (default 10⁻¹¹) and
`residual_rtol`, at most `refine_steps` (default 20) times, each correction computed on the
residual scaled to unit norm. Every returned answer meets `residual_rtol` (with
`residual_rtol = 0`, `refine_rtol`); otherwise the solve throws. Plain refinement with a
single-precision factor converges only when the matrix is well enough conditioned for it
(roughly κ(A)·u_single < 1), so 10⁻¹¹ is an empirical target, set just above the double-precision
floor some right-hand sides reach. "G" factors stay in double precision: cuDSS's single-precision
general factor diverged under refinement on these matrices. In a standalone benchmark of the same
factorizations on the quarter cell above, a single-precision "S" refactorization took 0.84 s on a
V100 (4.1 s in double) and 0.29 s on a B200, and 4–5 refinement steps of a few milliseconds each
reached 5·10⁻¹³; on GPUs with weak double precision it can be the faster choice.
`test/cudss_solver_tests.jl` and `test/cudss_gradient_tests.jl` (with
`DEO_SLOW_TESTS=1`) skip without CUDSS.jl or a GPU, unless `DEO_REQUIRE_CUDSS=1`.

The previous extension went through GridapPardiso. Its `solve!` performs the forward
solve only (iparm(12) is fixed with the solver and flipped for the CSC layout), so
the adjoint was formed as conj(A⁻¹ conj b), which equals Aᴴ⁻¹ b only for Aᵀ = A and
is wrong for oblique Bloch operators. The extension now needs Pardiso.jl alone.
`test/backend_adjoint_tests.jl` checks forward and adjoint residuals of every
backend that loads, and `test/pardiso_oblique_gradient_tests.jl` (run with
`DEO_SLOW_TESTS=1`, skipped without Pardiso.jl and MKL) checks oblique gradients and
agreement with UMFPACK.

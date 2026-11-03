# General-angle illumination and collection

`FieldConfig` accepts `tilt_x` and `tilt_y` in degrees. They specify direction
cosines sx=sin(tilt_x), sy=sin(tilt_y), not successive Euler rotations. The
physical downward ray is (sx,sy,-sqrt(1-sx²-sy²)); grazing/nonpropagating pairs
are rejected. `n_incident` is the lossless fluid index (air by default) and must
match the environment. The time convention is exp(+iωt), material index n−ik,
and physical E=envelope exp(−i k_parallel·r). The trial curl has the negative
Bloch shift and the bilinear test curl has the positive shift. Oblique matrices
are generally not complex symmetric. Use Pardiso `mtype=13` with
`assume_symmetric_maxwell=false`; symmetric storage now checks the matrix before
discarding a triangle and refuses incompatible operators.

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

Geometry L/W are widths of the independent mirror patch; a periodic direction
has twice that field width. All fields share one model. A weighted collection
containing x and y tilts therefore uses the full union domain, including its
normal pump. It does not yet give each tilt its own half-cell factorization.
The grid still contains only design_shape independent 2D DOFs; material values
are reflected into the enlarged field domain, and their adjoints scatter back
through the same reflection. There is no diagonal x/y exchange constraint.
Objectives and gradients are divided by the domain multiplicity (2 or 4), so
quarter/half/full values have the same independent-patch normalization.

The new builder defaults to a structured hexahedral mesh with matched mirror
intervals. `structured=false` keeps independent unstructured meshing; its
finite-mesh symmetry error must then be checked by refinement. Existing
`genmesh` calls retain their original tetrahedral behavior.

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
incident electric-field amplitude of 1/2, matching the old normal convention.

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
not establish that error for patterned structures.

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
cells. `test/mixed_output_gradient_tests.jl` covers the corrected shared-model
vector/material mapping. `scripts/benchmark_angular_solves.jl` compares local
sequential/concurrent factorization; no distributed scheduling layer is added.

Primary background: [Bloch envelopes](https://mpb.readthedocs.io/en/latest/Developer_Information/),
[Maxwell weak curl](https://mfem.org/fem_weak_form/),
[vector mirror symmetry](https://meep.readthedocs.io/en/latest/Exploiting_Symmetry/),
and [oblique Fresnel validation](https://doc.comsol.com/6.3/doc/com.comsol.help.models.rf.fresnel_equations/fresnel_equations.html).

The eigenmode problem is not extended here. Nonzero eigenmode angles now refuse
explicitly; the old generalized eigenproblem cannot represent this corrected
angle-dependent frequency-domain operator by silently reusing its old shift.

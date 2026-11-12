# Void-side surface objective

`SERSObjective(surface=true, volume=false, surface_mode=:tv_void)` integrates a void-side
surface weight over the design and Raman regions. The default `surface_mode=:band` retains
the existing `pt*(1-pt)` weight. Volume and surface terms can be enabled together.

The physical material indicator is `chi=pt` in the design layer, one in the substrate,
and zero elsewhere. A Helmholtz solve smooths it over `t_shell`, with zero normal flux
at mesh boundaries. The surface weight is `2*(1-chi)*sqrt(|grad chi_s|^2 + epsilon_grad^2)`.
It samples fields on the void side of material interfaces, including roofs and exposed
substrate. `t_shell` uses the mesh length unit; `ε_grad` has inverse-length units.
For an isolated planar interface, unbounded domain and zero regularization, the weight
integrates to unit area along the normal. Finite boundaries, nearby interfaces, quadrature
and positive regularization affect that normalization. Verify it for the selected geometry.
The regularization also creates a nonzero weight floor in flat void regions.

`t_shell` must be positive and finite, and `ε_grad` must be strictly positive and
finite for `:tv_void`, preventing division by zero in the shell adjoint. The mesh must have
a `Substrate` region unless `require_substrate=false` deliberately selects a substrate-free
geometry. The objective currently requires a single simulation mesh across field pairs;
construction and evaluation refuse unsupported simulation bundles.

The shell factorization is cached for its simulation and thickness. Its smoothed field is
cached by the assembled right-hand-side values, so in-place design changes and successive
legacy projected-field closures cannot reuse a stale shell. Legacy foundry interpolation
captures an evaluation-local copy of density values. The gradient includes both the
explicit material-indicator term and a Helmholtz adjoint, with the damage model differentiated
when enabled.

```julia
objective = SERSObjective(surface=true, volume=false, surface_mode=:tv_void,
                          t_shell=5.0, ε_grad=1e-6)
```

`surface_tv_void_tests.jl` checks gradients, active damage, fail-closed guards, and a
stateful flat/design/design sequence that verifies cache refresh and finite differences.
`sidewall_tvvoid_tests.jl` checks the combined surface, taper and damage path.

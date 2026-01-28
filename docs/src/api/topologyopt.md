# TopologyOpt

Filtering, projection, and constraint functions for topology optimization.

## Material Interpolation

```@docs
christiansen_ε
∂christiansen_ε
```

## Filtering (2D Grid)

```@docs
filter_grid
filter_grid_adjoint
```

## Filtering (3D Helmholtz)

```@docs
filter_helmholtz!
filter_helmholtz_adjoint!
```

## Projection

```@docs
project_grid
project_grid_adjoint
project_fe
project_ssp
```

## Constraints

```@docs
glc_solid
glc_void
glc_solid_fe
glc_void_fe
```

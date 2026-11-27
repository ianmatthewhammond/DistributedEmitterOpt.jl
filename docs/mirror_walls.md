# Mirror wall tags

A non-periodic axis of a `genmesh` cell ends at two mirror walls. For the polarization
along that axis the walls are PEC: `build_simulation(...; dir_x=true)` (or `dir_y`) makes
`FacesX` (or `FacesY`) a Dirichlet tag, and the tangential electric field must vanish on
every edge of the wall (and, from Nédélec order 1, on every face of it). Gridap makes a
degree of freedom Dirichlet only when the edge or face that owns it carries the tag.

## What a mesh must carry

Plain `genmesh` tags the walls on their surfaces only. Gmsh keeps the CAD curves of a wall
as separate entities: the lines where the wall meets the other wall, the layer interfaces,
and the top and bottom faces. This leaves some tangential degrees of freedom on the PEC wall curves free.

`genmesh(...; mirror_curves=true)` adds the wall curves and points to `FacesX`/`FacesY` (and
the bottom plane's curves to `BottomZ`). Structured meshes and newly generated angular-cell meshes include the curve groups. The quick start, the PML scripts and selected examples and test fixtures request it. `genmesh` keeps `mirror_curves=false` as its default, so an existing script still writes
the same mesh file as before.

`build_simulation` does not check whether a mesh it is given closes its PEC walls (#65).

## Upgrading a mesh

Regenerate it with the same arguments plus `mirror_curves=true`. The file gains boundary
elements and labels on the wall curves and points, and on the bottom plane's perimeter
(`BottomZ`). Where the old mesh left some wall edges free:

- the field space has fewer free degrees of freedom than before;
- the field, objective values, gradients and optimized designs can change. They need not,
  since the old solution can already satisfy the added constraints. Compare a result across
  the change only after re-evaluating it on the upgraded mesh.

Start fresh after changing the mesh. Use `DistributedEmitterOpt.load_checkpoint_design!` only after verifying that the design indices represent the same physical locations.

# Threaded assembly

One evaluation (objective and gradient) spends most of its time outside the linear solver in
Gridap cell loops: the Maxwell matrix, the objective integrals, the adjoint sources and the
sensitivity integrals. Gridap runs each loop on one thread. `src/Utils/ThreadedAssembly.jl` runs
them on every Julia thread and avoids reassembling the part of the matrix that does not depend on
the design. Start Julia with threads (`julia -t N`, or `JULIA_NUM_THREADS`); nothing else changes.

## What it does

- **Integrals and vector assembly.** Every `sum(∫(...)dΩ)` and every `assemble_vector` splits its
  cells into one contiguous chunk per thread. Each chunk has its own Gridap caches and its own
  output, and the outputs are added at the end.
- **Maxwell matrix.** `assemble_maxwell` writes the matrix as `A_bg + A_d(pt)`. Only the design
  term `A_d(pt) = -k² ∫ v·(ε_m(pt) u) dΩ_design` is recomputed. `A_bg` is taken once per
  (simulation, physical parameters) from the ordinary assembly at the zero design, so it holds
  every design-independent term of the formulation in use (legacy, Bloch, PML). The element
  matrices of the design term are computed in parallel and gathered into a copy of `A_bg`
  through an index map built once; no thread holds a copy of the matrix. The sparsity pattern is
  `A_bg`'s on every evaluation, so a solver's symbolic analysis stays valid. A plan is used by
  one task at a time and is rebuilt when one of the simulation's spaces or measures is replaced.
- **Projection derivative.** `dpt_dpf_field` is linear in the test function, so its two
  coefficients are evaluated once per quadrature point instead of once per test function.

Results differ from the serial ones only by the order of floating-point additions
(`test/threaded_assembly_tests.jl`: objectives to 1e-10, gradients to 1e-8, matrices to 1e-13).
The first evaluation of every split plan is compared with the ordinary assembly; on a mismatch
that problem is assembled directly from then on, with a logged warning. A design term the split
path does not handle (cells with different numbers of unknowns, block terms) is also assembled
directly. The first evaluation therefore costs two ordinary assemblies per field configuration.

## Switches

| | Default | Meaning |
|---|---|---|
| `DistributedEmitterOpt.THREADED[]` | `true` | threaded loops when Julia has more than one thread |
| `DistributedEmitterOpt.SPLIT_ASSEMBLY[]` | `true` | assemble only the design term per evaluation |
| `DistributedEmitterOpt.THREAD_MIN_CELLS[]` | `2000` | shorter loops stay serial |
| `DistributedEmitterOpt.THREAD_CHUNKS[]` | `0` | chunks per loop; 0 means one per thread |
| `clear_assembly_caches!()` | | frees the buffers and the split plans |

## Memory

The split assembly keeps, per (simulation, physical parameters): one copy of the matrix, the
element matrices of the design term (16 bytes × entries per element matrix × design cells) and
the index map (4 or 8 bytes per element-matrix entry, plus 8 bytes per stored matrix entry).
The threaded vector assembly keeps one vector per thread for each vector length it has seen.
At most `MAX_SPLIT_PLANS[]` (8) plans are kept.

## Scope

The threaded methods dispatch on Gridap types (`Base.sum` of a lazy cell array, and
`numeric_loop_vector!` for a `Vector` and a `GenericSparseMatrixAssembler`), so they apply to every
such loop in the session while this package is loaded. They cover `Float32`, `Float64` and their
complex types only, and defer to Gridap whenever threading is off, a loop is short, the element
type is anything else, or a term is not a plain cell-vector and cell-ids pair.

# =============================================================================
# Shared configuration — first-epoch (beta=8) OLD-vs-NEW divergence study.
# Quest tim-130116-82977.  Single source of truth for the matched config across
# all 3 arms (OLD Emitter3DTopOpt / NEW DEO :nlopt / NEW DEO :standalone_ccsaq).
# Pure constants only — no package dependencies, safe to `include` anywhere.
# =============================================================================

# --- Roots (override via env; defaults are the Supercloud worktrees) ---------
const OLD_ROOT = get(ENV, "FIRSTEPOCH_OLD_ROOT",
    "/home/gridsan/ihammond/GitHub/worktrees/Emitter3D-firstepoch-20260629")
const NEW_ROOT = get(ENV, "FIRSTEPOCH_NEW_ROOT",
    "/home/gridsan/ihammond/GitHub/worktrees/DEO-firstepoch-20260629")
const RUNROOT = get(ENV, "FIRSTEPOCH_RUNROOT",
    "/home/gridsan/ihammond/firstepoch-compare-20260629")

# ONE shared mesh for all arms. We REUSE the authentic 38MB paper mesh
# (copied into MESHDIR via login-safe cp), so no mesh is ever generated on a
# node. SRC_MESH is the provenance source; build_mesh! in new_common.jl is a
# documented fallback only and is intentionally NOT called.
const SRC_MESH = get(ENV, "FIRSTEPOCH_SRC_MESH",
    "/home/gridsan/ihammond/GitHub/DistributedEmitterOpt.jl/scripts/test-paper/test-metal/runs/constrained_metal_nominal/mesh.msh")
const MESHDIR  = joinpath(RUNROOT, "mesh")
const MESHFILE = joinpath(MESHDIR, "mesh.msh")

# --- Paper-grade metal-nominal geometry --------------------------------------
# Mirrors scripts/experiments/paper/constrained_metal_nominal.jl, with the
# fine ("paper-grade") l-values from the 07d22e3 commit (l2 = 2.9).
const GEOM = (
    L    = 184.28757605350117,
    W    = 184.28757605350117,
    hair = 599.8135303462574,
    hs   = 399.8756868975049,
    ht   = 199.93784344875246,
    hd   = 100.0,
    hsub = 50.0,
    dpml = 0.0,
    l1   = 12.5,
    l2   = 2.9,
    l3   = 12.5,
)

# --- Physics / control (metal nominal) ---------------------------------------
const MAT_DESIGN    = "Ag"
const MAT_SUBSTRATE = "Ag"
const N_FLUID       = 1.33
const LAMBDA        = 532.0
const THETA         = 0.0
const FILTER_RADIUS = (20.0, 20.0, 20.0)
const R_SSP         = 1.5
const BETA          = 8.0
const ETA           = 0.5
const ETA_EROSION   = 0.75
const ETA_DILATION  = 0.25
const B1            = 6.0e-6      # constraint hyperparam (inert: constraints OFF)
const C0            = 25600.0     # constraint hyperparam (inert: constraints OFF)
const ALPHA_LOSS    = 0.0
const E_THRESHOLD   = 10000.0

# --- Optimization ------------------------------------------------------------
const MAX_ITER   = 200
const TOL        = 1e-15
const INIT_VALUE = 0.5           # uniform-0.5 start (trap-2: broadcast over np)

# Snapshot cadence: iters 1..10, then every 10th.
should_snapshot(iter::Integer) = (iter <= 10) || (iter % 10 == 0)

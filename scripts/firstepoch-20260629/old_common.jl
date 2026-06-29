# =============================================================================
# OLD (Emitter3DTopOpt) builder — metal-nominal config, UMFPACK solver.
# Assumes the caller has already done:
#     import Emitter3DTopOpt as e3
#     include(e3.includesolver("Umfpack")); import .UmfpackSolver as OldUmfpackSolver
#     include(e3.includescript("Setup")); using .Setup
# and included config.jl.  Loads the shared mesh (genmesh=false): `root` must
# already contain a `mesh.msh` copy.  Mirrors experimental/.../metal-nominal/
# resume.yaml (Ag/Ag, subpixel SSP R=1.5, filter R=20), single wavelength
# (lambda2 = lambda1 = 532) so the OLD objective equals the NEW single-input one.
# =============================================================================

const ALPHA_P_OLD = Matrix{ComplexF64}(LinearAlgebra.I, 3, 3)

function build_old_objective(root::String; meshfile::String="mesh.msh")
    reuse_y1 = OldUmfpackSolver.UMFPACK_Reuse()
    reuse_y2 = reuse_y1
    reuse_x  = OldUmfpackSolver.UMFPACK_Reuse()

    obj = Setup.SetupAll(e3;
        root = endswith(root, "/") ? root : root * "/",
        genmesh = false,
        meshfile = meshfile,

        λ1 = LAMBDA, λ2 = LAMBDA,          # single wavelength (matches NEW)
        θ1 = THETA,  θ2 = THETA,
        mat_m = MAT_DESIGN, mat_s = MAT_SUBSTRATE,
        nf = N_FLUID,
        norder = 0, qorder = 6,
        α = ALPHA_LOSS, αₚ = ALPHA_P_OLD,

        mesh_type = "Box",
        L = GEOM.L, W = GEOM.W, hair = GEOM.hair, hs = GEOM.hs, ht = GEOM.ht,
        hd = GEOM.hd, hsub = GEOM.hsub, l1 = GEOM.l1, l2 = GEOM.l2, l3 = GEOM.l3,
        full_cell = false, full_x = false, full_y = false,
        bidirectional = false, nonlocal = false,

        flag_f = true, flag_t = true, flag_r = false, flag_c = true,
        flag_foundry = true, flagS = false, flagV = true,
        flag_nd = false, flag_e2 = false,
        R_f = FILTER_RADIUS, R_er = 0.0, R_nl = 0.0, R_s = R_SSP,
        β = BETA, η = ETA, ηe = ETA_EROSION, ηd = ETA_DILATION,
        b1 = B1, c0 = C0, γ = 1.0, Eₜₕ = E_THRESHOLD,
        subpixel = true, firststep = false,

        dir_x = false, dir_y = true,

        reuse_y1 = reuse_y1, reuse_y2 = reuse_y2, reuse_x = reuse_x,

        optimization = false,
        init = [INIT_VALUE],
        β_list = [BETA], α_list = [ALPHA_LOSS],
        max_iter = MAX_ITER,
    )
    return obj
end

using Test
using Serialization

# NLopt.jl and the standalone libmmaccsa BOTH export `nlopt_*` symbols.
# Loading both into one Julia process causes dynamic-linker interposition
# and the comparison becomes meaningless. Each backend therefore runs in
# its own subprocess.

const HELPER = joinpath(@__DIR__, "parity_helper.jl")
const JULIA = joinpath(Sys.BINDIR, "julia")
const PROJ = joinpath(@__DIR__, "..")

function run_helper(backend::String, case::String, algorithm::String, mode::String)
    out = tempname() * ".bin"
    cmd = `$JULIA --project=$PROJ $HELPER $backend $case $algorithm $mode $out`
    proc = run(pipeline(cmd; stdout=devnull, stderr=stderr); wait=true)
    @assert success(proc)
    result = open(deserialize, out)
    rm(out; force=true)
    return result
end

# Both backends are configured to be running the same algorithm:
#   - identical sigma_init (NLopt's default initial-step heuristic, computed
#     in Julia and passed explicitly into both)
#   - identical dual algorithm: NLOPT_LD_LBFGS = Luksan PLIS in both
#     (NLopt's MMA defaults to recursive MMA-on-MMA for the dual; we
#     override via params["dual_algorithm"])
#   - mma.c / ccsa_quadratic.c are the same source
#
# Most cases match to 1e-10 (effectively bit-identical). One case
# (rosenbrock / CCSAQ) drifts to ~5e-8 relative error after 60 iterations
# from accumulated floating-point order-of-operations differences. Tolerance
# below covers both cleanly.
const ATOL_F = 1e-8
const RTOL_F = 1e-7
const ATOL_X = 1e-7
const RTOL_X = 1e-7

function compare(label::AbstractString, ref, got;
                 atol_f=ATOL_F, rtol_f=RTOL_F,
                 atol_x=ATOL_X, rtol_x=RTOL_X)
    @testset "$label" begin
        @test isapprox(ref.fmin, got.fmin; atol=atol_f, rtol=rtol_f)
        @test isapprox(ref.p, got.p; atol=atol_x, rtol=rtol_x)
    end
end

# ---------------------------------------------------------------------------
# NLopt-vs-standalone parity
# ---------------------------------------------------------------------------
# Both backends run the same algorithm (matched sigma_init, matched dual
# solver). On converged problems, results are bit-identical. On non-
# converged problems the standalone finishes the current outer iteration
# before honoring maxeval (for checkpoint correctness), so it may do a few
# extra evals vs NLopt which exits mid-inner-loop. This causes a small
# trajectory divergence proportional to one outer iteration's improvement.

const CONVERGED_CASES = ("mccormick", "quadratic_constrained")

@testset "NLopt vs standalone backend (converged cases)" begin
    for case in CONVERGED_CASES, alg in ("mma", "ccsaq")
        @info "Parity check: $case / $alg"
        ref  = run_helper("nlopt",  case, alg, "full")
        got  = run_helper("stdlib", case, alg, "full")
        compare("$case / $alg :: stdlib ≈ NLopt", ref, got)
    end
end

@testset "NLopt vs standalone backend (non-converged, exit-semantics divergence)" begin
    for alg in ("mma", "ccsaq")
        @info "Parity check: rosenbrock / $alg"
        ref  = run_helper("nlopt",  "rosenbrock", alg, "full")
        got  = run_helper("stdlib", "rosenbrock", alg, "full")
        # MMA has 1 eval/outer-iter so both exit at outer boundary → tight.
        # CCSAQ has inner retries so standalone finishes 1 extra iter → looser.
        tol_f = alg == "mma" ? RTOL_F : 0.01
        tol_x = alg == "mma" ? RTOL_X : 0.01
        compare("rosenbrock / $alg :: stdlib ≈ NLopt", ref, got;
                rtol_f=tol_f, atol_f=tol_f, rtol_x=tol_x, atol_x=tol_x)
    end
end

# ---------------------------------------------------------------------------
# Standalone checkpoint roundtrip — all cases bit-identical
# ---------------------------------------------------------------------------
# The standalone only exits at outer-iteration boundaries (maxeval is
# checked at the top of the outer loop, not mid-inner-loop). This means
# checkpoint-resume always starts from a clean state and reproduces the
# uninterrupted trajectory exactly. The test helper accounts for the
# possible eval overrun by giving phase 2 (total - actual_phase1) budget.

const CHECKPOINT_CASES = ("rosenbrock", "mccormick", "quadratic_constrained")

@testset "Standalone checkpoint roundtrip (bit-identical)" begin
    for case in CHECKPOINT_CASES, alg in ("mma", "ccsaq")
        full   = run_helper("stdlib", case, alg, "full")
        ckpted = run_helper("stdlib", case, alg, "checkpoint")
        compare("$case / $alg :: stdlib full ≡ stdlib checkpoint",
                full, ckpted; atol_f=1e-14, rtol_f=1e-14,
                atol_x=1e-14, rtol_x=1e-14)
    end
end

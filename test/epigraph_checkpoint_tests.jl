# Resume-time validation of the epigraph configuration. Pure metadata — no PDE, seconds.
# The real write -> resume round trip through optimize! is in test/epigraph_resume_tests.jl.
using Test, DistributedEmitterOpt
const DEO = DistributedEmitterOpt

base(; extra...) = merge(Dict{String,Any}(
    "checkpoint_version" => DEO.OPTIMIZER_CHECKPOINT_VERSION,
    "current_epoch" => 1, "epoch_evals" => 20, "max_iter" => 200,
    "beta_schedule" => [8.0], "alpha_schedule" => nothing,
    "backend" => "standalone_ccsaq", "use_constraints" => false),
    Dict{String,Any}(string(k) => v for (k, v) in extra))

const Z0 = 0.0
samples = [(0.0, Z0), (tand(10.0), Z0)]
M = DEO._robust_samples_matrix(samples)
cfg = DEO._epigraph_config(:epigraph, 1e-8, 1.0, 1.6e-7, 1e-3)
epi_keys = Dict("robust_mode" => "epigraph", "robust_samples" => M, "robust_sharpness" => nothing,
                "epigraph_tol" => 1e-8, "epigraph_sigma" => 1.0, "linewidth_eps" => 1.6e-7,
                "linewidth_tol" => 1e-3, "epigraph_tau" => 1.0370000000000001, "epigraph_T" => 3.25)

ok(md; kw...) = DEO._validate_checkpoint_metadata!(md; filepath="ck", max_iter=200, beta_schedule=[8.0],
    alpha_schedule=nothing, backend=:standalone_ccsaq, use_constraints=false, kw...)
# a refusal counts only if it is THIS guard (pinned message), not any ArgumentError
refused(f) = try f(); false catch err
    err isa ArgumentError && occursin("robust sidewall configuration", sprint(showerror, err))
end

@testset "epigraph checkpoint metadata" begin
    @test DEO._epigraph_config(:ks, 1e-8, 1.0, 1e-8, 1e-3) === nothing
    @test cfg == (tol=1e-8, sigma=1.0, eps=1.6e-7, linewidth_tol=1e-3)

    md = DEO._checkpoint_metadata(base(; (Symbol(k) => v for (k, v) in epi_keys)...))
    @test md.robust_mode === :epigraph
    @test md.epigraph_config == cfg
    # τ and T come back BIT-identical (they rebuild x = [p; τ] on resume)
    @test md.epigraph_tau === 1.0370000000000001 && md.epigraph_T === 3.25
    @test md.robust_sharpness === nothing

    # same configuration resumes; sharpness is irrelevant to the epigraph
    @test ok(md; robust_samples=samples, robust_sharpness=200.0, robust_mode=:epigraph, epigraph_config=cfg) === nothing
    @test ok(md; robust_samples=samples, robust_sharpness=7.0, robust_mode=:epigraph, epigraph_config=cfg) === nothing

    # epigraph <-> KS refused both ways
    @test refused(() -> ok(md; robust_samples=samples, robust_sharpness=200.0))
    md_ks = DEO._checkpoint_metadata(base(robust_samples=M, robust_sharpness=200.0))   # pre-epigraph file
    @test md_ks.robust_mode === :ks && md_ks.epigraph_config === nothing && md_ks.epigraph_tau === nothing
    @test ok(md_ks; robust_samples=samples, robust_sharpness=200.0) === nothing        # KS still resumes
    @test refused(() -> ok(md_ks; robust_samples=samples, robust_sharpness=200.0,
                              robust_mode=:epigraph, epigraph_config=cfg))

    # a changed sample set, and each epigraph config field, refused
    @test refused(() -> ok(md; robust_samples=[(0.0, Z0), (tand(5.0), Z0)], robust_mode=:epigraph, epigraph_config=cfg))
    @test refused(() -> ok(md; robust_samples=[(0.0, -1.0), (tand(10.0), -1.0)], robust_mode=:epigraph, epigraph_config=cfg))
    @test refused(() -> ok(md; robust_samples=samples[1:1], robust_mode=:epigraph, epigraph_config=cfg))
    for (field, v) in ((:tol, 1e-6), (:sigma, 0.5), (:eps, 1e-8), (:linewidth_tol, 1e-8))
        bad = merge(cfg, NamedTuple{(field,)}((v,)))
        @test refused(() -> ok(md; robust_samples=samples, robust_mode=:epigraph, epigraph_config=bad))
    end
    # NEGATIVE CONTROL for `refused`: a non-robust mismatch (beta schedule) is an ArgumentError
    # with a DIFFERENT message, and must not count as the robust guard firing.
    @test !refused(() -> DEO._validate_checkpoint_metadata!(md; filepath="ck", max_iter=200,
        beta_schedule=[16.0], alpha_schedule=nothing, backend=:standalone_ccsaq, use_constraints=false,
        robust_samples=samples, robust_mode=:epigraph, epigraph_config=cfg))
end

@testset "epigraph fields propagate into per-iteration checkpoint metadata" begin
    nt = DEO._iteration_checkpoint_metadata(1, 20, 200, [8.0], nothing, :standalone_ccsaq, false,
        samples, nothing; robust_mode=:epigraph, epigraph_config=cfg, epigraph_tau=1.25, epigraph_T=3.0)
    @test nt.robust_mode === :epigraph && nt.epigraph_config == cfg
    @test nt.epigraph_tau == 1.25 && nt.epigraph_T == 3.0
    # KS call site (positional, as before) is unchanged apart from the recorded mode
    nt0 = DEO._iteration_checkpoint_metadata(1, 20, 200, [8.0], nothing, :standalone_ccsaq, false, samples, 200.0)
    @test nt0.robust_mode === :ks && nt0.robust_sharpness == 200.0 && nt0.epigraph_config === nothing
    @test nt0.epigraph_tau === nothing && nt0.epigraph_T === nothing
end
println("EPIGRAPH-CHECKPOINT-TESTS-DONE")

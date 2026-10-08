# Resume-time validation of the robust sidewall configuration .
# Pure metadata tests — no PDE. A checkpoint written with one (samples, sharpness) must refuse
# to resume under another, and a checkpoint without the keys must still resume when robust is off.
using Test, DistributedEmitterOpt
const DEO = DistributedEmitterOpt

base(; extra...) = merge(Dict{String,Any}(
    "checkpoint_version" => DEO.OPTIMIZER_CHECKPOINT_VERSION,
    "current_epoch" => 1, "epoch_evals" => 20, "max_iter" => 200,
    "beta_schedule" => [8.0], "alpha_schedule" => nothing,
    "backend" => "standalone_ccsaq", "use_constraints" => false), Dict{String,Any}(string(k) => v for (k, v) in extra))

samples = [(0.0, -1.0), (tand(10.0), -1.0)]
M = DEO._robust_samples_matrix(samples)

@testset "robust checkpoint metadata" begin
    @test size(M) == (2, 2) && M[2, 1] == tand(10.0) && M[1, 2] == -1.0
    @test DEO._robust_samples_matrix(nothing) === nothing

    md = DEO._checkpoint_metadata(base(robust_samples=M, robust_sharpness=200.0))
    @test md.exact_resume && md.robust_samples == M && md.robust_sharpness == 200.0
    ok(md; kw...) = DEO._validate_checkpoint_metadata!(md; filepath="ck", max_iter=200, beta_schedule=[8.0],
        alpha_schedule=nothing, backend=:standalone_ccsaq, use_constraints=false, kw...)

    # same configuration resumes
    @test ok(md; robust_samples=samples, robust_sharpness=200.0) === nothing
    # different slope, sharpness, sample count, or robust off: all refused
    @test_throws ArgumentError ok(md; robust_samples=[(0.0, -1.0), (tand(5.0), -1.0)], robust_sharpness=200.0)
    @test_throws ArgumentError ok(md; robust_samples=samples, robust_sharpness=100.0)
    @test_throws ArgumentError ok(md; robust_samples=samples[1:1], robust_sharpness=200.0)
    @test_throws ArgumentError ok(md; robust_samples=nothing, robust_sharpness=200.0)
    # different z0 is a different objective too
    @test_throws ArgumentError ok(md; robust_samples=[(0.0, 1.0), (tand(10.0), 1.0)], robust_sharpness=200.0)

    # legacy checkpoint (no robust keys): resumes with robust off, refuses with robust on
    md0 = DEO._checkpoint_metadata(base())
    @test md0.robust_samples === nothing && md0.robust_sharpness === nothing
    @test ok(md0) === nothing
    @test ok(md0; robust_samples=nothing, robust_sharpness=200.0) === nothing   # sharpness irrelevant when off
    @test_throws ArgumentError ok(md0; robust_samples=samples, robust_sharpness=200.0)
    # the running campaign's existing checks are untouched
    @test_throws ArgumentError DEO._validate_checkpoint_metadata!(md0; filepath="ck", max_iter=200, beta_schedule=[16.0],
        alpha_schedule=nothing, backend=:standalone_ccsaq, use_constraints=false)
end
@testset "robust config propagates into per-iteration checkpoint metadata" begin
    nt = DEO._iteration_checkpoint_metadata(1, 20, 200, [8.0], nothing, :standalone_ccsaq, false, samples, 200.0)
    @test nt.robust_samples == samples && nt.robust_sharpness == 200.0
    nt0 = DEO._iteration_checkpoint_metadata(1, 20, 200, [8.0], nothing, :standalone_ccsaq, false)
    @test nt0.robust_samples === nothing && nt0.robust_sharpness === nothing
    @test DEO._iteration_checkpoint_metadata(0, 20, 200, [8.0], nothing, :standalone_ccsaq, false, samples, 200.0) === nothing
end
println("ROBUST-CHECKPOINT-TESTS-DONE")

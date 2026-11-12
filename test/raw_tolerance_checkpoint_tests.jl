# Pure checkpoint metadata checks: no mesh, filter, or PDE work.
using Test, DistributedEmitterOpt
const DEO=DistributedEmitterOpt
base() = Dict{String,Any}("checkpoint_version"=>DEO.OPTIMIZER_CHECKPOINT_VERSION,
    "current_epoch"=>1,"epoch_evals"=>3,"max_iter"=>5,"beta_schedule"=>[Inf],
    "alpha_schedule"=>nothing,"backend"=>"standalone_mma","use_constraints"=>true)
validate(md;kw...) = DEO._validate_checkpoint_metadata!(md;filepath="test-checkpoint",
    max_iter=5,beta_schedule=[Inf],alpha_schedule=nothing,backend=:standalone_mma,
    use_constraints=true,kw...)
@testset "plain constraint tolerance is part of resume configuration" begin
    legacy=DEO._checkpoint_metadata(base())
    @test legacy.constraint_tol_raw==1e-8
    @test validate(legacy)===nothing
    @test_throws ArgumentError validate(legacy;constraint_tol_raw=6e-14)
    custom=DEO._checkpoint_metadata(merge(base(),Dict("constraint_tol_raw"=>6e-14)))
    @test custom.constraint_tol_raw==6e-14
    @test validate(custom;constraint_tol_raw=6e-14)===nothing
    @test_throws ArgumentError validate(custom)
    @test_throws ArgumentError validate(custom;constraint_tol_raw=1e-13)
    for raw in (1e-8,6e-14)
        md=DEO._iteration_checkpoint_metadata(1,3,5,[Inf],nothing,:standalone_mma,true;
            constraint_tol_raw=raw)
        @test md.constraint_tol_raw==raw
    end
end
@testset "constraint scale and normalization are recorded and checked on resume" begin
    md=DEO._checkpoint_metadata(base())
    @test md.g_norm===nothing && md.constraint_scale===nothing
    @test validate(md;constraint_scale=1e8)===nothing     # older checkpoint: not checked
    scaled=DEO._checkpoint_metadata(merge(base(),Dict("constraint_scale"=>1e8,"g_norm"=>7.0)))
    @test scaled.constraint_scale==1e8 && scaled.g_norm==7.0
    @test validate(scaled;constraint_scale=1e8)===nothing
    @test_throws ArgumentError validate(scaled;constraint_scale=1.0)
    md2=DEO._iteration_checkpoint_metadata(1,3,5,[Inf],nothing,:standalone_mma,true;
        g_norm=7.0,constraint_scale=1e8)
    @test md2.g_norm==7.0 && md2.constraint_scale==1e8
end

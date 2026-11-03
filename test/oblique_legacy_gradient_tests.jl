include("oblique_common.jl")
@testset "General-angle legacy grid projection gradients" begin
    fc=FieldConfig(532;tilt_x=20,pol=:x)
    half=angular_test_problem([fc],[fc])
    half.control.foundry_projection_mode=:legacy
    _,half_errors=angular_ladder(half)
    @test half_errors[end]<1e-4
    inputs=[FieldConfig(532;tilt_x=0,pol=:y)]
    samples=reduce_mirror_samples(rim_collection_samples(0.45;normal_weight=0.3,rim_weight=0.175);
                                  mirror_x=true,mirror_y=true)
    full=angular_test_problem(inputs,collection_configs(532,samples))
    full.control.foundry_projection_mode=:legacy
    @test DEO.num_solves(full.pde)==6
    _,full_errors=angular_ladder(full)
    @test full_errors[end]<1e-4
end

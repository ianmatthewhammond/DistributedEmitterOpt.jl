include("oblique_common.jl")
@testset "General-angle foundry gradients" begin
    @testset "configuration" begin
        @test cache_key(FieldConfig(532.0)) == (532.0,0.0,:y)
        @test DEO.configs_equal(FieldConfig(532;tilt_x=20),FieldConfig(532;tilt_x=20,weight=2))
        @test !DEO.configs_equal(FieldConfig(532;tilt_x=20),FieldConfig(532;tilt_y=20))
        @test_throws ArgumentError FieldConfig(532;tilt_x=60,tilt_y=60)
        @test_throws ArgumentError FieldConfig(532;θ=20,formulation=:legacy)
        @test_throws ArgumentError PhysicalParams(2pi/532,20,1.0+0im,2.0+0im,
            1.5+0im,1.0,0.0,30.0,0.0,false)
        for tx in (0.0,20.0), ty in (0.0,15.0)
            fx=FieldConfig(532;tilt_x=tx,tilt_y=ty,pol=:x)
            fy=FieldConfig(532;tilt_x=tx,tilt_y=ty,pol=:y)
            ex,ey=polarization_vector(fx),polarization_vector(fy)
            k=DEO.VectorValue(sind(tx),sind(ty),-sqrt(1-sind(tx)^2-sind(ty)^2))
            @test norm(ex) ≈ 1
            @test norm(ey) ≈ 1
            @test abs(dot(ex,ey)) < 1e-14
            @test abs(dot(ex,k)) < 1e-14
            @test abs(dot(ey,k)) < 1e-14
        end
    end
    damage_baselines=Dict{Tuple{Symbol,Symbol},Float64}()
    for damage in (false,true), axis in (:x,:y), pol in (:x,:y)
        @testset "one-axis $axis $pol damage=$damage" begin
            fc=axis==:x ? FieldConfig(532;tilt_x=20,pol) : FieldConfig(532;tilt_y=20,pol)
            prob=angular_test_problem([fc],[fc];damage)
            sim=DEO.default_sim(prob.sim)
            @test (sim.per_x,sim.per_y) == (axis==:x,axis==:y)
            println("CASE axis=$axis pol=$pol damage=$damage nV=$(sim.nV)"); flush(stdout)
            g,errs=angular_ladder(prob)
            @test g>0 && isfinite(g)
            if damage
                @test g < 0.9damage_baselines[(axis,pol)]
            else
                damage_baselines[(axis,pol)]=g
            end
            @test minimum(errs) < 1e-4
            @test errs[end] < 1e-4
            @test_throws ArgumentError DEO.build_phys_params(
                axis==:x ? FieldConfig(532;tilt_y=20,pol) : FieldConfig(532;tilt_x=20,pol), prob.pde.env,sim)
        end
    end
    for damage in (false,true)
    @testset "quarter-cell mixed polarization damage=$damage" begin
        inputs=[FieldConfig(532;tilt_x=0,pol=:y)]
        outputs=collection_configs(532,[(0.0,0.0,1.0)])
        prob=angular_test_problem(inputs,outputs;damage)
        @test !DEO.default_sim(prob.sim).per_x && !DEO.default_sim(prob.sim).per_y
        _,errs=angular_ladder(prob)
        @test errs[end]<1e-4
    end
    @testset "two-axis full cell damage=$damage" begin
        fc=FieldConfig(532;tilt_x=15,tilt_y=20,pol=:x)
        prob=angular_test_problem([fc],[fc];damage)
        @test all((DEO.default_sim(prob.sim).per_x,DEO.default_sim(prob.sim).per_y))
        _,errs=angular_ladder(prob)
        @test errs[end]<1e-4
    end
    end
    for damage in (false,true)
        @testset "weighted normal and signed rim damage=$damage" begin
            inputs=[FieldConfig(532;tilt_x=0,pol=:y)]
            outputs=collection_configs(532,rim_collection_samples(0.45;normal_weight=0.3,rim_weight=0.175))
            prob=angular_test_problem(inputs,outputs;damage)
            @test DEO.num_solves(prob.pde)==10
            _,errs=angular_ladder(prob)
            @test errs[end]<1e-4
        end
    end
end

include("oblique_common.jl")

function dielectric_problem(inputs,outputs;cell=:auto,resolution=12.0)
    geo=angular_test_geometry(;resolution)
    sims=build_angular_simulation(geo,vcat(inputs,outputs);
        meshfile=joinpath(mktempdir(),"symmetry.msh"),cell,design_shape=(7,7))
    pde=MaxwellProblem(env=Environment(mat_design=2.0,mat_substrate=1.77,mat_fluid=1.0),inputs=inputs,outputs=outputs)
    obj=SERSObjective(volume=false,surface=true,surface_mode=:tv_void,t_shell=7.0)
    control=Control(use_filter=true,R_filter=(8.0,8.0,8.0),β=4.0,use_ssp=false)
    OptimizationProblem(pde,obj,sims,UmfpackSolver();foundry_mode=true,control)
end

@testset "Reflection and cell reduction physics" begin
    rng=MersenneTwister(915)
    p=0.4 .+ 0.2.*rand(rng,49)
    rel(a,b)=abs(a-b)/max(abs(a),abs(b))
    for pol in (:x,:y)
        fx=FieldConfig(532;tilt_x=20,pol)
        fy=FieldConfig(532;tilt_y=20,pol=pol==:x ? :y : :x)
        hx=dielectric_problem([fx],[fx]); hy=dielectric_problem([fy],[fy])
        gx=objective_and_gradient!(Float64[],p,hx)
        gy=objective_and_gradient!(Float64[],collect(vec(transpose(reshape(p,7,7)))),hy)
        println("SYMMETRY transpose pol=$pol x=$gx y=$gy relative=$(rel(gx,gy))");flush(stdout)
        @test rel(gx,gy)<1e-9
        full=dielectric_problem([fx],[fx];cell=:full)
        gf=objective_and_gradient!(Float64[],p,full)
        println("SYMMETRY half-full pol=$pol half=$gx full=$gf relative=$(rel(gx,gf))");flush(stdout)
        @test rel(gx,gf)<1e-9
    end
    # Reuse one full periodic mesh for +/− and weighted/reduced sums so the
    # comparison does not conflate symmetry with independent meshing error.
    pump=[FieldConfig(532;tilt_x=0,pol=:y)]
    samples=rim_collection_samples(0.45;normal_weight=0.3,rim_weight=0.175)
    outputs=collection_configs(532,samples)
    prob=dielectric_problem(pump,outputs;cell=:full)
    gfull=objective_and_gradient!(Float64[],p,prob)
    reduced=reduce_mirror_samples(samples;mirror_x=true,mirror_y=true)
    reduced_pde=MaxwellProblem(env=prob.pde.env,inputs=pump,outputs=collection_configs(532,reduced))
    pred=OptimizationProblem(reduced_pde,prob.objective,prob.sim,UmfpackSolver();foundry_mode=true,control=prob.control)
    gred=objective_and_gradient!(Float64[],p,pred)
    println("SYMMETRY rim signed=$gfull reduced=$gred relative=$(rel(gfull,gred)) solves=$(DEO.num_solves(reduced_pde))");flush(stdout)
    @test DEO.num_solves(reduced_pde)==6
    @test rel(gfull,gred)<1e-9
    # Compare individual opposite directions, with the same fixed normal pump.
    for axis in (:x,:y),pol in (:x,:y)
        theta=asind(0.45)
        cfg(sign)=axis==:x ? FieldConfig(532;tilt_x=sign*theta,pol) : FieldConfig(532;tilt_y=sign*theta,pol)
        values=Float64[]
        for sign in (1,-1)
            pde=MaxwellProblem(env=prob.pde.env,inputs=pump,outputs=[cfg(sign)])
            pp=OptimizationProblem(pde,prob.objective,prob.sim,UmfpackSolver();foundry_mode=true,control=prob.control)
            push!(values,objective_and_gradient!(Float64[],p,pp))
        end
        println("SYMMETRY opposite axis=$axis pol=$pol values=$values relative=$(rel(values...))");flush(stdout)
        @test rel(values...)<1e-9
    end
end

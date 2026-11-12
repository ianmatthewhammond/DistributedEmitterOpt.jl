using DistributedEmitterOpt, LinearAlgebra, Random, Test
const DEO = DistributedEmitterOpt
BLAS.set_num_threads(2)

function angular_test_geometry(;resolution=15.0)
    geo = SymmetricGeometry(532.0; L=60.0,W=60.0,hd=30.0,hsub=60.0)
    geo.hair=160; geo.hs=100; geo.ht=40
    geo.l1=resolution; geo.l2=resolution; geo.l3=resolution
    geo
end

function angular_test_problem(inputs, outputs; damage=false, cell=:auto, resolution=15.0)
    geo = angular_test_geometry(;resolution)
    configs = vcat(inputs,outputs)
    sim = build_angular_simulation(geo,configs;
        meshfile=joinpath(mktempdir(),"angular.msh"),cell,design_shape=(7,7))
    env = Environment(mat_design="Ag",mat_substrate=1.77,mat_fluid=1.0)
    pde = MaxwellProblem(;env,inputs,outputs)
    obj = SERSObjective(volume=false,surface=true,surface_mode=:tv_void,t_shell=7.0,
        use_damage_model=damage,γ_damage=4.0,E_threshold=0.0)
    control = Control(use_filter=true,R_filter=(8.0,8.0,8.0),β=4.0,use_ssp=false,
                      foundry_projection_mode=:current)
    OptimizationProblem(pde,obj,sim,UmfpackSolver();foundry_mode=true,control)
end

function angular_ladder(prob; seed=612)
    rng=MersenneTwister(seed)
    p=0.4 .+ 0.2 .* rand(rng,length(prob.p))
    grad=zeros(length(p)); g=objective_and_gradient!(grad,p,prob)
    d=randn(rng,length(p)); d ./= norm(d)
    adj=dot(grad,d)
    errors=Float64[]
    for h in (1e-3,1e-4,1e-5)
        gp=objective_and_gradient!(Float64[],p+h*d,prob)
        gm=objective_and_gradient!(Float64[],p-h*d,prob)
        fd=(gp-gm)/(2h)
        rel=abs(adj-fd)/max(abs(fd),abs(adj),eps())
        println("LADDER h=$h g=$g adj=$adj fd=$fd relative=$rel"); flush(stdout)
        push!(errors,rel)
    end
    g,errors
end

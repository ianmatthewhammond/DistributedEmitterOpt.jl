# Local sparse-factorization benchmark. Run with JULIA_NUM_THREADS=2 and
# OPENBLAS_NUM_THREADS=2. Compare separate sequential/concurrent processes.
# Matrix assembly and compilation are outside the reported solve wall time.
using DistributedEmitterOpt, LinearAlgebra
using Base.Threads
const DEO=DistributedEmitterOpt
BLAS.set_num_threads(2)
mode=isempty(ARGS) ? "sequential" : ARGS[1]
mode in ("sequential","concurrent") || error("mode must be sequential or concurrent")
resolution=length(ARGS)>1 ? parse(Float64,ARGS[2]) : 10.0
geo=SymmetricGeometry(488.0;L=100.0,W=100.0,hd=20.0,hsub=100.0)
geo.hair=200;geo.hs=130;geo.ht=50
geo.l1=resolution;geo.l2=resolution;geo.l3=resolution
samples=reduce_mirror_samples(rim_collection_samples(0.45);mirror_x=true,mirror_y=true)
configs=collection_configs(488,samples)
bundle=build_angular_simulation(geo,configs;meshfile=joinpath(mktempdir(),"bench.msh"),design_shape=(17,17))
env=Environment(mat_design="Ag",mat_substrate="SiO2",mat_fluid=1.0)
sim=DEO.default_sim(bundle)
pt=DEO.interpolate(x->0.5,sim.Pf)
matrices=[assemble_maxwell(pt,sim,DEO.build_phys_params(fc,env,sim)) for fc in configs]
sources=[assemble_source(sim,DEO.build_phys_params(fc,env,sim)) for fc in configs]
# Warm the direct solver on a small matrix, then compile the task paths.
factor_solve(A,b)=(F=lu(A);(F,F\b))
factor_solve(matrices[1][1:20,1:20],sources[1][1:20])
function run_batch(mode,matrices,sources)
    if mode=="sequential"
        [factor_solve(A,b) for (A,b) in zip(matrices,sources)]
    else
        tasks=[Threads.@spawn factor_solve(A,b) for (A,b) in zip(matrices,sources)]
        fetch.(tasks)
    end
end
run_batch(mode,[matrices[1][1:20,1:20]],[sources[1][1:20]])
GC.gc()
rss_kib()=parse(Int,strip(read(`ps -o rss= -p $(getpid())`,String)))
baseline=rss_kib();peak=Ref(baseline);running=Threads.Atomic{Bool}(true)
monitor=Threads.@spawn begin
    while running[]
        peak[]=max(peak[],rss_kib())
        sleep(0.05)
    end
end
elapsed=@elapsed results=run_batch(mode,matrices,sources)
peak[]=max(peak[],rss_kib());running[]=false;wait(monitor)
residuals=[norm(A*x-b)/norm(b) for (A,b,(_,x)) in zip(matrices,sources,results)]
println("BENCH mode=$mode resolution=$resolution threads=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads()) factorizations=$(length(configs)) nV=$(sim.nV) seconds=$elapsed baseline_MiB=$(baseline/1024) peak_MiB=$(peak[]/1024) process_maxrss_bytes=$(Sys.maxrss()) max_residual=$(maximum(residuals))")

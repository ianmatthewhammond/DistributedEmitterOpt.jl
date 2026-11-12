using DistributedEmitterOpt, LinearAlgebra, Test
const DEO = DistributedEmitterOpt
BLAS.set_num_threads(2)

# Independent 2x2 interface composition, with tangential-electric amplitudes
# and exp(+iωt). No FEM assembly or boundary helper enters this reference.
function film_reference(k, sx, sy, nfilm, ns, h, tm)
    κ2=k^2*(sx^2+sy^2)
    q(n)=sqrt(complex((k*n)^2-κ2))
    y(n)=tm ? (k*n)^2/q(n) : q(n)
    y0,y1,y2=y(1),y(nfilm),y(ns)
    r01=(y0-y1)/(y0+y1); r12=(y1-y2)/(y1+y2)
    phase=exp(-im*q(nfilm)*h)
    r=(r01+r12*phase^2)/(1+r01*r12*phase^2)
    t=(1+r01)*(1+r12)*phase/(1+r01*r12*phase^2)
    (;r,t,R=abs2(r),T=real(y2)/real(y0)*abs2(t),q0=q(1),qs=q(ns),y0,ys=y2)
end

function film_mesh(resolution)
    geo=SymmetricGeometry(600.0;L=100.0,W=100.0,hd=40.0,hsub=100.0)
    geo.hair=240;geo.hs=160;geo.ht=70
    geo.l1=resolution;geo.l2=resolution;geo.l3=resolution
    mesh=joinpath(mktempdir(),"film.msh")
    lo,hi=genmesh(geo,mesh;per_x=true,per_y=true)
    sim=build_simulation(mesh;dir_x=false,dir_y=false,foundry_mode=true,degree=4)
    geo,lo,hi,sim
end

function film_result(geo,lo,hi,sim,fc)
    env=Environment(mat_design=1.5,mat_substrate=1.77,mat_fluid=1.0)
    phys=DEO.build_phys_params(fc,env,sim)
    pt=DEO.interpolate(x->1.0,sim.Pf)
    A=assemble_maxwell(pt,sim,phys);b=assemble_source(sim,phys)
    E=DEO.FEFunction(sim.U,A\b)
    axis=fc.polarization==:x ? 1 : 2
    et=polarization_vector(fc)[axis];k=2pi/fc.λ
    tm=axis==1 ? fc.tilt_x!=0 : fc.θ!=0
    ref=film_reference(k,sind(fc.tilt_x),sind(fc.θ),1.5,1.77,geo.hd,tm)
    avg(measure,component)=sum(DEO.∫((u->u[component])∘E)measure)/sum(DEO.∫(1)measure)
    zsource=hi+geo.hs;ztop=hi+geo.hair;zbot=lo-geo.hsub
    rf=(avg(sim.dS_top,axis)/(0.5et*exp(-im*ref.q0*(ztop-zsource)))-1)*exp(2im*ref.q0*(zsource-hi))
    tf=avg(sim.dS_bottom,axis)/(0.5et*exp(im*ref.q0*(hi-zsource))*exp(im*ref.qs*(zbot-lo)))
    # A signed normal component distinguishes opposite Bloch-shift conventions;
    # R/T in a uniform film alone cannot distinguish the two tilt signs.
    κ=inplane_wavevector(fc)
    # H(curl) elements control the tangential trace, not the normal trace.
    # Fit the homogeneous substrate plane wave in its volume instead.
    phase=x->exp(-im*ref.qs*(x[3]-lo))
    bulk(component)=sum(DEO.∫(((u->u[component])∘E)*phase)sim.dΩ_substrate)/sum(DEO.∫(1)sim.dΩ_substrate)
    ez=bulk(3)
    ez_expected=(κ[1]*bulk(1)+κ[2]*bulk(2))/ref.qs
    (;ref,R=abs2(rf),T=real(ref.ys)/real(ref.y0)*abs2(tf),ez,ez_expected)
end

@testset "Oblique film Fresnel physics" begin
    worst=Float64[]
    for resolution in (15.0,10.0)
        geo,lo,hi,sim=film_mesh(resolution)
        errors=Float64[]
        for angle in (10.0,30.0),axis in (:x,:y),pol in (:x,:y)
            fc=axis==:x ? FieldConfig(600;tilt_x=angle,pol) : FieldConfig(600;tilt_y=angle,pol)
            r=film_result(geo,lo,hi,sim,fc)
            err=max(abs(r.R-r.ref.R),abs(r.T-r.ref.T))
            push!(errors,err)
            println("FRESNEL mesh=$resolution axis=$axis pol=$pol angle=$angle nV=$(sim.nV) R=$(r.R) exactR=$(r.ref.R) T=$(r.T) exactT=$(r.ref.T) absolute=$err ez=$(r.ez) expectedEz=$(r.ez_expected)");flush(stdout)
            @test err<0.006
            @test abs(r.R+r.T-1)<0.006
            if (axis==pol)
                @test abs(r.ez-r.ez_expected)/abs(r.ez_expected)<0.04
                @test abs(r.ez+r.ez_expected)>1.8abs(r.ez_expected)
            else
                @test abs(r.ez)<0.005
            end
        end
        push!(worst,maximum(errors))
    end
    println("FRESNEL worst coarse=$(worst[1]) fine=$(worst[2])")
    @test worst[2]<worst[1]
end

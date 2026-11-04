# The Bloch boundary is exact for the specular order only, so a lattice on which a nonspecular
# diffraction order propagates must be refused, by name. At 325 nm and a 45° tilt the largest
# period with every air order evanescent is λ/(1 + sin 45°) ≈ 190.4 nm: 170 nm passes, 340 nm
# (geo.L = 170, since the angular builder's L is half the lattice period) does not.
using DistributedEmitterOpt, Gridap, Test
const DEO = DistributedEmitterOpt

@testset "Propagating diffraction orders are refused" begin
    λ = 325.0
    metal = sqrt(complex(-15.1, -2.43))        # metal-like substrate, n − ik
    real(metal) < 0 && (metal = -metal)
    message(f) = try
        f(); ""
    catch err
        err isa ArgumentError ? err.msg : rethrow()
    end
    for pol in (:x, :y)
        fc = FieldConfig(λ; tilt_y=45, pol)
        @test DEO.check_diffraction_orders((170.0, 170.0), fc, metal) === nothing
        msg = message(() -> DEO.check_diffraction_orders((340.0, 340.0), fc, metal))
        println("GUARD pol=$pol 340 nm: ", msg)
        @test occursin("(0, -1) in the incidence medium", msg)
        @test occursin("190.4 × 190.4", msg)
        # A lossless substrate carries the (0, -1) order already at 170 nm.
        msg = message(() -> DEO.check_diffraction_orders((170.0, 170.0), fc, 1.45 + 0im))
        @test occursin("(0, -1) in the substrate", msg)
        @test !occursin("incidence medium", msg)
    end
    @test DEO.check_diffraction_orders((170.0, 170.0), FieldConfig(λ; tilt_y=0, pol=:y), metal) === nothing
    @test DEO.check_diffraction_orders((300.0, 300.0), FieldConfig(λ; tilt_y=0, pol=:y), metal) === nothing
    @test !isempty(message(() -> DEO.check_diffraction_orders((330.0, 330.0), FieldConfig(λ; tilt_y=0, pol=:y), metal)))

    # Through the builder and build_phys_params. "Al" from the material data is a lossy substrate.
    env = Environment(mat_design="Al", mat_substrate="Al", mat_fluid=1.0)
    for (box, ok) in ((85.0, true), (170.0, false))
        geo = SymmetricGeometry(λ; L=box, W=box, hd=30.0, hsub=60.0)
        geo.hair = 300; geo.hs = 200; geo.ht = 80
        geo.l1 = 30.0; geo.l2 = 15.0; geo.l3 = 30.0
        fc = FieldConfig(λ; tilt_y=45, pol=:x)
        sim = build_angular_simulation(geo, [fc]; meshfile=joinpath(mktempdir(), "cell.msh"),
            design_shape=(5, 5), structured=false)
        s = DEO.sim_for(sim, fc)
        @test DEO.lattice_periods(s) == (2box, 2box)
        if ok
            @test DEO.build_phys_params(fc, env, s) isa DEO.PhysicalParams
        else
            @test_throws ArgumentError DEO.build_phys_params(fc, env, s)
        end
    end
end

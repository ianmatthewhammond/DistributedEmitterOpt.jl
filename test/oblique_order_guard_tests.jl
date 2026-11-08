# The Bloch boundary is exact for the specular order only, so a lattice on which a nonspecular
# diffraction order propagates must be refused, by name. At 325 nm and a 45° tilt the largest
# period with every air order evanescent is λ/(1 + sin 45°) ≈ 190.4 nm: 170 nm passes, 340 nm
# (geo.L = 170, since the angular builder's L is half the lattice period) does not.
using DistributedEmitterOpt, Gridap, Test
using SparseArrays: nonzeros
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

    # A PEC bottom has no admittance term, so its substrate orders are not checked: the lossless
    # 1.45 substrate that carries (0, -1) at 170 nm is fine there. The incidence medium still is.
    for pol in (:x, :y)
        fc = FieldConfig(λ; tilt_y=45, pol)
        @test DEO.check_diffraction_orders((170.0, 170.0), fc, 1.45 + 0im; bottom_pec=true) === nothing
        msg = message(() -> DEO.check_diffraction_orders((340.0, 340.0), fc, 1.45 + 0im; bottom_pec=true))
        @test occursin("(0, -1) in the incidence medium", msg)
        @test !occursin("substrate", msg)
    end

    # Orders at cutoff (q = 0) do not decay: refused, as is an order within the cutoff
    # tolerance; one clearly below cutoff passes.
    normal = FieldConfig(λ; tilt_y=0, pol=:y)
    msg = message(() -> DEO.check_diffraction_orders((λ, λ), normal, metal))
    println("GUARD cutoff P = λ: ", msg)
    @test occursin("in the incidence medium", msg)
    @test !isempty(message(() -> DEO.check_diffraction_orders((λ * (1 - 1e-8), λ * (1 - 1e-8)), normal, metal)))
    @test DEO.check_diffraction_orders((0.99λ, 0.99λ), normal, metal) === nothing

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
        if ok
            # A dielectric substrate is refused at this period unless the bottom is PEC.
            glass = Environment(mat_design="Al", mat_substrate=1.45, mat_fluid=1.0)
            @test_throws ArgumentError DEO.build_phys_params(fc, glass, s)
            pec = build_angular_simulation(geo, [fc]; meshfile=joinpath(mktempdir(), "pec.msh"),
                design_shape=(5, 5), structured=false, bot_PEC=true)
            backed = Environment(mat_design="Al", mat_substrate=1.45, mat_fluid=1.0, bot_PEC=true)
            @test DEO.build_phys_params(fc, backed, DEO.sim_for(pec, fc)) isa DEO.PhysicalParams
        end
    end

    # Assembly over a PEC bottom never builds the substrate admittance. A lossless n = 0.5
    # substrate under a 30° tilt from air has its specular order exactly at cutoff
    # (q² = k²/4 − (k/2)² = 0), where that admittance is undefined.
    geo = SymmetricGeometry(λ; L=85.0, W=85.0, hd=30.0, hsub=60.0)
    geo.hair = 300; geo.hs = 200; geo.ht = 80
    geo.l1 = 30.0; geo.l2 = 15.0; geo.l3 = 30.0
    fc = FieldConfig(λ; tilt_y=30, pol=:x)
    pec = build_angular_simulation(geo, [fc]; meshfile=joinpath(mktempdir(), "cutoff.msh"),
        design_shape=(5, 5), structured=false, bot_PEC=true)
    s = DEO.sim_for(pec, fc)
    cutoff = Environment(mat_design="Al", mat_substrate=0.5, mat_fluid=1.0, bot_PEC=true)
    phys = DEO.build_phys_params(fc, cutoff, s)
    pt = DEO.interpolate(x -> 0.5, s.Pf)
    A = DEO.assemble_maxwell(pt, s, phys)
    @test size(A, 1) == size(A, 2) > 0
    @test all(isfinite, nonzeros(A))
end

# Legacy foundry mode projects on the 2D grid, which has no sidewall model. A sidewall
# request there must be refused, not evaluated as a vertical extrusion.
using DistributedEmitterOpt
using Test

const DEO = DistributedEmitterOpt

@testset "legacy foundry mode refuses sidewalls" begin
    x = collect(range(0.0, 40.0; length=5))
    sim = (; grid=(; x, y=copy(x)))
    pf = vec([0.1 + 0.8 * (i + j) / 8 for i in 0:4, j in 0:4])   # nonuniform ramp
    control = DEO.Control(; foundry_projection_mode=:legacy)

    pt = DEO.smoothed_projection_vec(pf, control, sim)   # positive control
    @test length(pt) == 25 && all(isfinite, pt)

    control.use_sidewall = true
    control.sidewall_slope = tan(deg2rad(10))
    control.sidewall_z0 = -20.0
    err = try
        DEO.smoothed_projection_vec(pf, control, sim)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("foundry_projection_mode=:current", sprint(showerror, err))
end

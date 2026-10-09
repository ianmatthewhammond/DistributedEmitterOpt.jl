# The serialized-state header reader (state_dims / state_header) and WHY it is needed:
# a state restored into a problem with a different n or m makes the C optimizer return
# a negative code, and MmaccsaBackend.minimize! returns that code without throwing.
# Pure C + Julia, no PDE: seconds.
using Test
using DistributedEmitterOpt
const MB = DistributedEmitterOpt.MmaccsaBackend

quad(c) = (x, g) -> begin
    isempty(g) || (g .= 2 .* (x .- c))
    sum(abs2, x .- c)
end

@testset "MMA state header" begin
    @test MB.library_available()
    @test sizeof(MB.MmaStateSerialHeader) == 80   # C ABI layout of mma_state_serial_header

    dir = mktempdir()
    # n = 3, m = 1: a real state written by the C optimizer
    st = MB.MmaState()
    x = [0.9, 0.1, 0.5]
    con = ((x, g) -> begin isempty(g) || (g .= 1.0); sum(x) - 2.0 end, 1e-8)
    commits = Tuple{Float64,Int}[]
    ret, fmin, _ = MB.minimize!(:ccsaq, quad([0.2, 0.3, 0.4]), x;
        lb=zeros(3), ub=ones(3), maxeval=6, state=st, constraints=[con],
        checkpoint_callback=(xc, mf, ne, k) -> push!(commits, (mf, k)))
    @test ret > 0
    path = joinpath(dir, "n3m1.mma_state")
    MB.save_state(path, st)

    @test MB.state_dims(path) == (3, 1)
    h = MB.state_header(path)
    # The header holds the LAST COMMIT (outer-iteration boundary), which is what the
    # checkpoint callback saw — not the later point minimize! returns after maxeval.
    # The epigraph resume check (-minf == τ saved at that commit) relies on exactly this.
    @test !isempty(commits)
    @test h.minf == commits[end][1]
    @test Int(h.k) == commits[end][2]
    @test h.minf != fmin     # the committed objective differs from the best objective returned

    # a state for n = 4, m = 0 reads back as such (the reader is not returning constants)
    st4 = MB.MmaState()
    x4 = fill(0.5, 4)
    MB.minimize!(:ccsaq, quad(fill(0.1, 4)), x4; lb=zeros(4), ub=ones(4), maxeval=3, state=st4)
    path4 = joinpath(dir, "n4m0.mma_state")
    MB.save_state(path4, st4)
    @test MB.state_dims(path4) == (4, 0)

    # NEGATIVE CONTROL for the whole reason the reader exists: resuming the n=3,m=1
    # state into an n=4 problem does NOT throw — minimize! returns ret < 0 and leaves
    # x untouched. A caller that ignores ret gets a silent no-op link.
    bad = MB.load_state(path)
    x_bad = fill(0.5, 4)
    ret_bad, _, _ = MB.minimize!(:ccsaq, quad(fill(0.1, 4)), x_bad;
        lb=zeros(4), ub=ones(4), maxeval=5, state=bad, constraints=[con])
    @test ret_bad < 0
    @test x_bad == fill(0.5, 4)

    # corrupt / foreign files are refused by name, not misread
    junk = joinpath(dir, "junk.mma_state")
    write(junk, zeros(UInt8, 100))
    err = try MB.state_dims(junk); nothing catch e; e end
    @test err isa ArgumentError && occursin("bad magic", sprint(showerror, err))
    short = joinpath(dir, "short.mma_state")
    write(short, zeros(UInt8, 10))
    err = try MB.state_dims(short); nothing catch e; e end
    @test err isa ArgumentError && occursin("shorter than", sprint(showerror, err))
end
println("MMACCSA-STATE-HEADER-TESTS-DONE")

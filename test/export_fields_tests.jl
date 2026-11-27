# The exported physical density and fields belong to the requested design, and an export leaves
# the problem's solver state alone.
#
# `Analysis.visualize_results` must write the density and the field that the objective sees at
# the design it is given, whatever was evaluated or exported earlier on the same problem:
#
#   - the density must go through the selected filter and projection (in 3D mode the Helmholtz
#     filter into the nodal space, not the raw elementwise design vector);
#   - Maxwell must be assembled and factorized for the requested design, not solved with a factor
#     left in a cache by another design.
#
# An export must also not change what the problem computes next: it factorizes the filter and
# Maxwell operators in solver caches of its own, releases them on every exit, and leaves the
# problem's caches (objects, factors, solution vectors) exactly as it found them.
#
# Testsets:
#   0. The VTU reader used below refuses encodings it does not read.
#   1. Files and arrays: for each mode (3D, foundry :current, foundry :legacy) and nonuniform
#      designs A and B, export B, A, B on one problem (warm) and B on a fresh problem (cold). The
#      named arrays read back from design.vtu ("p") and fields.vtu ("E_real", "E_imag") must
#      equal the same arrays written from an independent production evaluation at B (the
#      objective's filter and projection steps written out here, then `solve_forward!`); the
#      reference writes only those three arrays, and the exported "E_sq" must equal the sum of
#      squares of the reference real and imaginary arrays. The in-memory density, filtered
#      vector, field and objective must match production.
#   2. Forward at A, export B, adjoint: the adjoint must equal the adjoint without the export, and
#      the problem's caches (objects and solution-vector values) must be unchanged by the export,
#      also when the export fails, and with two configurations when the second one is exported.
#   3. Export-owned factors are released after success, after a failed solve and after a refusal
#      raised once the density is built.
#   4. 3D filter: an export leaves the problem's filter cache unchanged; an export after a radius
#      change, and the next production density after exports, use the current radius.
#   5. A refused export (wrong design length, refused by `visualize_results` before anything is
#      printed; field configuration refused by its simulation), a successful export and an export whose write fails leave every state of a problem that has
#      run unchanged: its fields, best record and history, controls, solver caches and, in
#      foundry mode, the shared grid values.
#   6. The exported density and field stay valid on a backend that clears a factor's solution
#      storage when the factor is released.
#   7. A backend whose new factor empties another factor's idle solution buffer declares
#      `returns_owned_solutions` false and is refused by both entry points in every mode, before
#      any factorization, solve, release or print: the caller's retained field keeps its entries.
#   8. A backend that declares `returns_owned_solutions` false (one reusable solution vector on
#      the solver object) is refused by `design_and_fields` for the y and the x polarization of a
#      bundle and by `visualize_results` for its first input, y, before any solve, factorization,
#      print or file: the caller's retained fields, cache vectors and factors keep their objects
#      and values, and the accepting solver of a mixed pool does no work either. The Maxwell
#      solver of the exported configuration is checked; the default pool's solver (the filter's)
#      is checked only when a 3D filter solve is needed, so in foundry mode and in 3D with the
#      filter off it does not veto an export of x. The internal density helper refuses such a
#      filter solver itself, and `physical_density` stays usable on it.
#   9. `design_and_fields` and `visualize_results` refuse a foundry control with `use_dct=false`
#      before anything is logged, printed, loaded into the material cache, factorized or solved.
#  10. Each entry point reads its solver objects once at entry: a logger that replaces the pool
#      during the near-cutoff warning does not make the export solve on the replacement.
#  11. The density helper refuses a wrong design length (one too many, none) and `use_dct=false`
#      under its own name, before any factorization.
#  12. Both entry points refuse the backend first, `use_dct=false` second, a wrong length third.
#  13. The refusals that read only the control and the configuration's flags (projection mode,
#      sidewalls with the legacy projection, a tilt across a mirror wall, Bloch with the ±z PML)
#      come before any material table is loaded, anything is logged or printed, or any solve.
#  14. Export cleanup leaves no filter factor and no filter key, also when the release throws.
#
# All testsets sit in one outer testset, so a failure in one does not stop the others from
# running and reporting when this file is run on its own. Comments marked "regression:" say
# what the assertions below them catch.
using DistributedEmitterOpt
using Gridap
using LinearAlgebra
using Logging
using Random
using Test

isdefined(@__MODULE__, :build_test_problem) || include("gradient_common.jl")

const XDEO = DistributedEmitterOpt

"Physical density sampled at the quadrature points of the design region, as one flat vector."
function sampled_density(pt, sim0)
    pts = Gridap.CellData.get_cell_points(sim0.dΩ_design)
    cf = pt isa Function ? CellField(pt, sim0.Ω_design) : pt
    vals = cf(pts)
    return reduce(vcat, [collect(Float64, vals[i]) for i in 1:length(vals)])
end

"Production forward path at `p`, written out step by step (as in the gradient coordinator)."
function production_reference(prob, p)
    (; pde, sim, pool, control) = prob
    sim0 = default_sim(sim)
    local pt
    if prob.foundry_mode
        pf_vec = filter_grid(p, sim0, control)
        if control.foundry_projection_mode == :legacy
            pt_vec = XDEO.smoothed_projection_vec(pf_vec, control, sim0)
            pt = XDEO.build_foundry_pf(pt_vec, sim0, control)
        else
            pf = XDEO.build_foundry_pf(pf_vec, sim0, control)
            pt = project_ssp(pf, control, sim0.Ω)
        end
    else
        pf_vec = filter_helmholtz!(p, default_pool(pool).filter_cache, sim0, control)
        pf = FEFunction(sim0.Pf, pf_vec)
        pt = project_ssp(pf, control, sim0.Ω)
    end
    fields = solve_forward!(pde, pt, sim, pool)
    return (; pf_vec, pt, fields)
end

relerr(a, b) = norm(a .- b) / max(norm(b), floatmin(Float64))

const EXPORT_CASES = (
    ("3D", (; foundry_mode=false)),
    ("foundry current", (; foundry_mode=true, foundry_projection_mode=:current)),
    ("foundry legacy", (; foundry_mode=true, foundry_projection_mode=:legacy)),
)

"The grid values a production density at `p` writes in foundry mode: the filtered grid in the
`:current` mode, the projected grid in the `:legacy` mode."
function written_grid(p, prob)
    sim0 = default_sim(prob.sim)
    pf_vec = filter_grid(p, sim0, prob.control)
    prob.control.foundry_projection_mode == :legacy || return pf_vec
    return XDEO.smoothed_projection_vec(pf_vec, prob.control, sim0)
end

same_mesh(a, b) = Gridap.Geometry.get_node_coordinates(default_sim(a.sim).model) ==
                  Gridap.Geometry.get_node_coordinates(default_sim(b.sim).model)

# ---------------------------------------------------------------------------------------------
# Reading the arrays of a VTU file
# ---------------------------------------------------------------------------------------------

"""
    read_vtu_arrays(path) -> Dict{String,Matrix{Float64}}

The Float64 data arrays of a VTU file written by Gridap's `writevtk` with its defaults (WriteVTK
raw appended binary data, uncompressed, little-endian), keyed by their `Name`; each array is
`NumberOfComponents × number of tuples`. It reads only the named arrays whose `type` is
`Float64` and skips the others (including a Float64 array without a `Name`) without checking
them. It errors instead of guessing when the appended data block is not raw, is compressed or
big-endian, has a header type other than UInt32 or UInt64, or does not start with `_` (after
whitespace), and when a named Float64 array is not in the appended format or has no offset.
"""
function read_vtu_arrays(path::AbstractString)
    bytes = read(path)
    text = read(path, String)                 # byte-indexed; only the ASCII XML part is searched
    marker = findfirst("<AppendedData encoding=\"raw\">", text)
    marker === nothing && error("$path: no raw appended data block")
    xml = text[1:first(marker)-1]
    occursin("compressor=", xml) && error("$path: compressed data is not supported by this reader")
    occursin("byte_order=\"LittleEndian\"", xml) || error("$path: not little-endian")
    HT = occursin("header_type=\"UInt32\"", xml) ? UInt32 :
         occursin("header_type=\"UInt64\"", xml) ? UInt64 :
         error("$path: header type is not UInt32 or UInt64")
    first_data = findnext(c -> !isspace(c), text, last(marker) + 1)
    (first_data === nothing || text[first_data] != '_') && error("$path: the appended data does not start with '_'")
    base = first_data + 1                     # appended offsets count from the byte after '_'
    arrays = Dict{String,Matrix{Float64}}()
    for m in eachmatch(r"<DataArray\s[^>]*>", xml)
        tag = m.match
        name = match(r"\sName=\"([^\"]*)\"", tag)
        typ = match(r"\stype=\"([^\"]*)\"", tag)
        fmt = match(r"\sformat=\"([^\"]*)\"", tag)
        off = match(r"\soffset=\"(\d+)\"", tag)
        (name === nothing || typ === nothing) && continue
        typ[1] == "Float64" || continue
        (fmt !== nothing && fmt[1] == "appended" && off !== nothing) ||
            error("$path: array $(name[1]) is not raw appended data")
        ncm = match(r"\sNumberOfComponents=\"(\d+)\"", tag)
        nc = ncm === nothing ? 1 : parse(Int, ncm[1])
        start = base + parse(Int, off[1])
        nb = Int(reinterpret(HT, bytes[start:start+sizeof(HT)-1])[1])
        first_byte = start + sizeof(HT)
        data = collect(reinterpret(Float64, bytes[first_byte:first_byte+nb-1]))
        arrays[name[1]] = reshape(data, nc, :)
    end
    return arrays
end

"""
Write the density and field of an independent production evaluation (`pt`, `E`) with the same
`writevtk` calls on the same triangulations as `visualize_results`, and read the arrays back.
"""
function reference_arrays(dir, prob, pt, E, config)
    sim0 = default_sim(prob.sim)
    pt_cf = pt isa Function ? CellField(pt, sim0.Ω_design) : pt
    writevtk(sim0.Ω_design, joinpath(dir, "ref_design"), cellfields=["p" => pt_cf])
    writevtk(sim_for(prob.sim, config).Ω, joinpath(dir, "ref_fields"),
        cellfields=["E_real" => real(E), "E_imag" => imag(E)])
    return (read_vtu_arrays(joinpath(dir, "ref_design.vtu")),
        read_vtu_arrays(joinpath(dir, "ref_fields.vtu")))
end

exported_arrays(dir) = (read_vtu_arrays(joinpath(dir, "design.vtu")),
    read_vtu_arrays(joinpath(dir, "fields.vtu")))

# ---------------------------------------------------------------------------------------------
# Snapshot of a problem's solver caches
# ---------------------------------------------------------------------------------------------

pools_of(pool) = pool isa XDEO.SolverPoolBundle ? Any[pool.default; collect(values(pool.by_pol))] : Any[pool]

"Identities of a cache, its factors, its filter key and its solution vector, and a copy of that vector's values."
cache_state(c) = (ids=(c, c.A_factor, c.F_factor, c.F_key, c.E2_factor, c.x), x=copy(c.x))

"Identity of every cache, factor and solution vector held by `pool` (a pool or a bundle), and the solution values."
pool_snapshot(pool) = [(pool=p, filter=cache_state(p.filter_cache), eigen=cache_state(p.eigen_cache),
    shift=p.eigen_shift, caches=Dict(k => cache_state(c) for (k, c) in p.caches)) for p in pools_of(pool)]

same_state(a, b) = length(a.ids) == length(b.ids) && all(x === y for (x, y) in zip(a.ids, b.ids)) &&
                   isequal(a.x, b.x)

"True when `pool` holds exactly the objects recorded in `snap`, with the same solution values."
function pool_unchanged(pool, snap)
    cur = pool_snapshot(pool)
    length(cur) == length(snap) || return false
    for (a, b) in zip(cur, snap)
        (a.pool === b.pool && same_state(a.filter, b.filter) && same_state(a.eigen, b.eigen) &&
         isequal(a.shift, b.shift)) || return false
        Set(keys(a.caches)) == Set(keys(b.caches)) || return false
        all(same_state(a.caches[k], b.caches[k]) for k in keys(b.caches)) || return false
    end
    return true
end

# ---------------------------------------------------------------------------------------------
# Snapshot of the rest of a problem's state
# ---------------------------------------------------------------------------------------------

"Everything of `prob` other than its solver caches (see `pool_snapshot`), as values or identities."
function problem_state(prob)
    sim0 = default_sim(prob.sim)
    grid = prob.foundry_mode ? (copy(sim0.grid.params), sim0.grid.adj_ready, sim0.grid.legacy_ready) : nothing
    return (; pde=prob.pde, objective=prob.objective, sim=prob.sim, pool=prob.pool, root=prob.root,
        control=deepcopy(prob.control), p=copy(prob.p), g=prob.g, ∇g=copy(prob.∇g),
        g_history=copy(prob.g_history), p_best=copy(prob.p_best), g_best=prob.g_best,
        iteration=prob.iteration, filter_revision=prob.filter_revision, grid)
end

"True when `prob` holds exactly the state recorded in `st` by `problem_state`."
function problem_unchanged(prob, st)
    now = problem_state(prob)
    (now.pde === st.pde && now.objective === st.objective && now.sim === st.sim &&
     now.pool === st.pool) || return false
    all(isequal(getfield(now.control, f), getfield(st.control, f)) for f in fieldnames(typeof(st.control))) ||
        return false
    return all(isequal(getfield(now, k), getfield(st, k))
               for k in (:root, :p, :g, :∇g, :g_history, :p_best, :g_best, :iteration, :filter_revision, :grid))
end

# ---------------------------------------------------------------------------------------------
# A UMFPACK backend that counts live factors and can fail a Maxwell solve on request
# ---------------------------------------------------------------------------------------------

mutable struct CountingSolver <: XDEO.AbstractSolver
    made::Int               # factorizations made
    live::Int               # factorizations made and not yet released
    peak::Int               # largest `live` seen just after a factorization
    fail_maxwell::Bool      # throw from every complex (Maxwell) solve
end
CountingSolver() = CountingSolver(0, 0, 0, false)
count_made!(s::CountingSolver) = (s.made += 1; s.live += 1; s.peak = max(s.peak, s.live); s)
XDEO.lu!(s::CountingSolver, A) = (F = lu(A); count_made!(s); F)
XDEO.filter_lu!(s::CountingSolver, A) = (F = lu(A); count_made!(s); F)
function XDEO.solve!(s::CountingSolver, F, b::Vector)
    (s.fail_maxwell && eltype(b) <: Complex) && error("injected Maxwell solve failure")
    return F \ b
end
XDEO.solve_adjoint!(::CountingSolver, F, b::Vector) = F' \ b
XDEO.release_factor!(s::CountingSolver, F) = (s.live -= 1; nothing)

# ---------------------------------------------------------------------------------------------
# A UMFPACK backend that solves into storage owned by each factor and clears it on release. No
# solve on another factor or release of another factor writes that storage, so it keeps the
# default `returns_owned_solutions` (true).
# ---------------------------------------------------------------------------------------------

struct WorkspaceSolver <: XDEO.AbstractSolver end
struct WorkspaceFactor{F,T}
    lu::F
    x::Vector{T}            # every solve on this factor writes here and returns it
end
XDEO.lu!(::WorkspaceSolver, A) = WorkspaceFactor(lu(A), zeros(eltype(A), size(A, 2)))
XDEO.filter_lu!(s::WorkspaceSolver, A) = XDEO.lu!(s, A)
XDEO.solve!(::WorkspaceSolver, F::WorkspaceFactor, b::Vector) = ldiv!(F.x, F.lu, b)
XDEO.solve_adjoint!(::WorkspaceSolver, F::WorkspaceFactor, b::Vector) = F.lu' \ b
XDEO.release_factor!(::WorkspaceSolver, F::WorkspaceFactor) = (fill!(F.x, zero(eltype(F.x))); nothing)

# ---------------------------------------------------------------------------------------------
# A UMFPACK backend with an independent factorization and solution buffer per factor: every solve
# writes only its own factor's buffer and returns it, and release does nothing. Making a new factor
# empties the idle buffer of the factor made before it, which changes a vector a caller still
# holds from a solve on another factor, so it declares `returns_owned_solutions` false. It counts
# factorizations, solves and releases.
# ---------------------------------------------------------------------------------------------

struct TrimFactor{F,T}
    lu::F
    buffer::Vector{T}       # every solve on this factor writes here and returns it
end
mutable struct TrimSolver <: XDEO.AbstractSolver
    previous::Any           # the factor made last, or nothing
    made::Int
    solves::Int
    released::Int
end
TrimSolver() = TrimSolver(nothing, 0, 0, 0)
if isdefined(XDEO, :returns_owned_solutions)
    XDEO.returns_owned_solutions(::TrimSolver) = false
end
function XDEO.lu!(s::TrimSolver, A)
    isnothing(s.previous) || empty!(s.previous.buffer)
    f = TrimFactor(lu(A), zeros(eltype(A), size(A, 2)))
    s.previous = f
    s.made += 1
    return f
end
XDEO.filter_lu!(s::TrimSolver, A) = XDEO.lu!(s, A)
function XDEO.solve!(s::TrimSolver, f::TrimFactor, b::Vector)
    s.solves += 1
    resize!(f.buffer, length(b))
    return ldiv!(f.buffer, f.lu, b)
end
XDEO.solve_adjoint!(s::TrimSolver, f::TrimFactor, b::Vector) = (s.solves += 1; f.lu' \ b)
XDEO.release_factor!(s::TrimSolver, f::TrimFactor) = (s.released += 1; nothing)
trim_counts(s::TrimSolver) = (s.made, s.solves, s.released)

# ---------------------------------------------------------------------------------------------
# A UMFPACK backend with independent factors and one reusable complex solution vector on the
# solver object: every complex solve, on any factor, writes that vector and returns it. It
# declares `returns_owned_solutions` false, and counts factorizations and solves.
# ---------------------------------------------------------------------------------------------

mutable struct SharedSolutionSolver <: XDEO.AbstractSolver
    x::Vector{ComplexF64}
    made::Int               # factorizations made
    solves::Int             # solves, forward and adjoint
end
SharedSolutionSolver() = SharedSolutionSolver(ComplexF64[], 0, 0)
# Defined only where the interface function exists, so that this file still loads, and reports each
# assertion, on a version of the package without it.
if isdefined(XDEO, :returns_owned_solutions)
    XDEO.returns_owned_solutions(::SharedSolutionSolver) = false
end
XDEO.lu!(s::SharedSolutionSolver, A) = (s.made += 1; lu(A))
XDEO.filter_lu!(s::SharedSolutionSolver, A) = (s.made += 1; lu(A))
function XDEO.solve!(s::SharedSolutionSolver, F, b::Vector)
    s.solves += 1
    eltype(b) <: Complex || return F \ b
    length(s.x) == length(b) || resize!(s.x, length(b))
    s.x .= F \ b
    return s.x
end
XDEO.solve_adjoint!(s::SharedSolutionSolver, F, b::Vector) = (s.solves += 1; F' \ b)

# ---------------------------------------------------------------------------------------------
# A UMFPACK backend whose release throws
# ---------------------------------------------------------------------------------------------

struct ThrowingReleaseSolver <: XDEO.AbstractSolver end
XDEO.lu!(::ThrowingReleaseSolver, A) = lu(A)
XDEO.filter_lu!(::ThrowingReleaseSolver, A) = lu(A)
XDEO.solve!(::ThrowingReleaseSolver, F, b::Vector) = F \ b
XDEO.solve_adjoint!(::ThrowingReleaseSolver, F, b::Vector) = F' \ b
XDEO.release_factor!(::ThrowingReleaseSolver, F) = error("injected release failure")

# ---------------------------------------------------------------------------------------------
# A UMFPACK backend that counts factorizations, solves and releases and keeps the default
# `returns_owned_solutions` (true)
# ---------------------------------------------------------------------------------------------

mutable struct TallySolver <: XDEO.AbstractSolver
    made::Int               # factorizations made
    solves::Int             # solves, forward and adjoint
    released::Int           # factors released
end
TallySolver() = TallySolver(0, 0, 0)
XDEO.lu!(s::TallySolver, A) = (s.made += 1; lu(A))
XDEO.filter_lu!(s::TallySolver, A) = (s.made += 1; lu(A))
XDEO.solve!(s::TallySolver, F, b::Vector) = (s.solves += 1; F \ b)
XDEO.solve_adjoint!(s::TallySolver, F, b::Vector) = (s.solves += 1; F' \ b)
XDEO.release_factor!(s::TallySolver, F) = (s.released += 1; nothing)
tally(s::TallySolver) = (s.made, s.solves, s.released)

"A bundle whose y pool (also the default pool) uses `ysolver` and whose x pool uses `xsolver`."
two_pol_bundle(ysolver, xsolver) = (yp = SolverCachePool(ysolver);
    XDEO.SolverPoolBundle(yp, Dict(:y => yp, :x => SolverCachePool(xsolver))))

"Run `f` and return the exception it threw (or `nothing`), its log records at every level and what it printed."
function observe(f)
    logger = Test.TestLogger(min_level=Logging.Debug)
    err, printed = mktemp() do path, io
        e = redirect_stdout(io) do
            with_logger(logger) do
                try
                    f()
                    nothing
                catch ex
                    ex
                end
            end
        end
        flush(io)
        (e, read(path, String))
    end
    return (; err, logs=logger.logs, printed)
end

"True when `err` is an `ArgumentError` whose message contains every one of `parts`."
refusal_names(err, parts...) = err isa ArgumentError && all(occursin(q, err.msg) for q in parts)

"""
A foundry problem (the geometry of `build_test_problem`'s foundry mode) on a mesh periodic in x
and mirrored in y, in air over an air substrate, with a silver design, the filter on and
`use_dct=true`; its one input is y at 532 nm.
"""
function periodic_foundry_problem(solver)
    geo = SymmetricGeometry()
    geo.L = 200.0
    geo.W = 200.0
    geo.l1 = 40.0
    geo.l2 = 20.0
    geo.l3 = 40.0
    env = Environment(mat_design="Ag", mat_substrate=1.0, mat_fluid=1.0)
    pde = MaxwellProblem(env=env, inputs=[FieldConfig(532.0; pol=:y)], outputs=FieldConfig[])
    control = Control(use_filter=true, R_filter=(20.0, 20.0, 20.0), use_dct=true, use_projection=true,
        β=8.0, η=0.5, use_ssp=true)
    meshfile = tempname() * ".msh"
    try
        genmesh(geo, meshfile; per_x=true, per_y=false)
        prob = OptimizationProblem(pde, SERSObjective(), meshfile, solver;
            per_x=true, per_y=false, foundry_mode=true, control=control)
        init_uniform!(prob, 0.5)
        return prob
    finally
        rm(meshfile; force=true)
    end
end

"A logger that runs `action` when a warning (or anything above) is logged, and counts how often."
struct ActOnWarn <: Logging.AbstractLogger
    action::Function
    fired::Base.RefValue{Int}
end
ActOnWarn(action) = ActOnWarn(action, Ref(0))
Logging.min_enabled_level(::ActOnWarn) = Logging.Debug
Logging.shouldlog(::ActOnWarn, args...) = true
Logging.catch_exceptions(::ActOnWarn) = false
function Logging.handle_message(l::ActOnWarn, level, msg, _module, group, id, file, line; kwargs...)
    level >= Logging.Warn && (l.fired[] += 1; l.action())
    return nothing
end

"Run `f` with the package's material cache saved, and put the saved cache back afterwards."
function with_saved_material_cache(f)
    saved = copy(XDEO._MATERIAL_CACHE)
    try
        return f()
    finally
        empty!(XDEO._MATERIAL_CACHE)
        merge!(XDEO._MATERIAL_CACHE, saved)
    end
end

# ---------------------------------------------------------------------------------------------

# regression: with the testsets at top level, the first failing one stopped this file when run on
# its own and the later ones never ran. In one outer testset every testset runs and reports; the
# outer testset fails if any of them does.
@testset "Export fields" begin
    @testset "The VTU reader refuses encodings it does not read" begin
        # regression: the reader read any header type other than UInt32 as UInt64.
        mktempdir() do dir
            function fixture(name; vtkfile="byte_order=\"LittleEndian\" header_type=\"UInt64\"",
                             array="format=\"appended\" offset=\"0\"", encoding="raw", header=UInt64,
                             lead="\n_")
                xml = "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" $vtkfile>\n" *
                      "<UnstructuredGrid><Piece><PointData>" *
                      "<DataArray type=\"Float64\" Name=\"a\" NumberOfComponents=\"1\" $array/>" *
                      "</PointData></Piece></UnstructuredGrid>\n<AppendedData encoding=\"$encoding\">" * lead
                bytes = vcat(Vector{UInt8}(xml), reinterpret(UInt8, [header(16)]), reinterpret(UInt8, [1.5, -2.0]),
                             Vector{UInt8}("\n</AppendedData>\n</VTKFile>\n"))
                path = joinpath(dir, name * ".vtu")
                write(path, bytes)
                return path
            end
            @test read_vtu_arrays(fixture("ok64"))["a"] == [1.5 -2.0]
            @test read_vtu_arrays(fixture("ok32"; vtkfile="byte_order=\"LittleEndian\" header_type=\"UInt32\"",
                                          header=UInt32))["a"] == [1.5 -2.0]
            @test_throws ErrorException read_vtu_arrays(fixture("base64"; encoding="base64"))
            @test_throws ErrorException read_vtu_arrays(fixture("compressed";
                vtkfile="byte_order=\"LittleEndian\" header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\""))
            @test_throws ErrorException read_vtu_arrays(fixture("bigendian";
                vtkfile="byte_order=\"BigEndian\" header_type=\"UInt64\""))
            @test_throws ErrorException read_vtu_arrays(fixture("uint16"; vtkfile="byte_order=\"LittleEndian\" header_type=\"UInt16\""))
            @test_throws ErrorException read_vtu_arrays(fixture("ascii"; array="format=\"ascii\""))
            # regression: the reader searched forward for any '_', so text before the data block was skipped.
            @test read_vtu_arrays(fixture("spaces"; lead=" \n\t _"))["a"] == [1.5 -2.0]
            @test_throws ErrorException read_vtu_arrays(fixture("stray"; lead="\nstray text_"))
            # A named appended Float64 array without an offset is refused.
            @test_throws ErrorException read_vtu_arrays(fixture("nooffset"; array="format=\"appended\""))
        end
    end

    @testset "Exported density and fields follow the requested design" begin
        for (name, kw) in EXPORT_CASES
            @testset "$name" begin
                warm = build_test_problem(; kw...)
                cold = build_test_problem(; kw...)
                n = length(warm.p)
                rng = MersenneTwister(2911)
                A = 0.05 .+ 0.40 .* rand(rng, n)     # nonuniform, mostly void
                B = 0.55 .+ 0.40 .* rand(rng, n)     # nonuniform, mostly solid
                config = warm.pde.inputs[1]
                key = cache_key(config)

                mktempdir() do dir
                    dB1, dA, dB2, dBc, dRw, dRc = (mkdir(joinpath(dir, s)) for s in ("B1", "A", "B2", "Bc", "Rw", "Rc"))
                    # Warm: B, A, then B again on one problem. Cold: B on a fresh problem.
                    Analysis.visualize_results(warm, B; root=dB1)
                    Analysis.visualize_results(warm, A; root=dA)
                    Analysis.visualize_results(warm, B; root=dB2)
                    Analysis.visualize_results(cold, B; root=dBc)

                    # regression: the A export reused B's factor, so the A and B files were identical.
                    @test read(joinpath(dA, "fields.vtu")) != read(joinpath(dB1, "fields.vtu"))

                    # regression: a zero (or any other) density in design.vtu, or the real and
                    # imaginary parts written under each other's names. The named arrays are read back
                    # and compared with the same arrays written from production at B.
                    ref_w = production_reference(warm, B)
                    ref_c = production_reference(cold, B)
                    for (label, prob, dexp, dref, ref) in (("warm", warm, dB2, dRw, ref_w), ("cold", cold, dBc, dRc, ref_c))
                        @testset "$label files" begin
                            (des, fld) = exported_arrays(dexp)
                            (rdes, rfld) = reference_arrays(dref, prob, ref.pt, ref.fields[key], config)
                            @test haskey(des, "p") && haskey(fld, "E_real") && haskey(fld, "E_imag") && haskey(fld, "E_sq")
                            @test size(des["p"]) == size(rdes["p"]) && size(des["p"], 2) > 0
                            @test maximum(abs, rdes["p"]) > 0.1                    # the reference is not empty
                            @test relerr(des["p"], rdes["p"]) <= 1e-12
                            @test size(fld["E_real"]) == size(rfld["E_real"]) == size(fld["E_imag"])
                            @test relerr(fld["E_real"], rfld["E_real"]) <= 1e-9
                            @test relerr(fld["E_imag"], rfld["E_imag"]) <= 1e-9
                            @test relerr(rfld["E_imag"], rfld["E_real"]) > 1e-3    # a swap would be seen
                            E_sq_ref = vec(sum(abs2, rfld["E_real"]; dims=1) .+ sum(abs2, rfld["E_imag"]; dims=1))
                            @test relerr(vec(fld["E_sq"]), E_sq_ref) <= 1e-9
                        end
                    end
                    # The arrays tell A from B.
                    (desA, fldA) = exported_arrays(dA)
                    (desB, fldB) = exported_arrays(dB2)
                    @test relerr(desA["p"], desB["p"]) > 1e-3
                    @test relerr(fldA["E_real"], fldB["E_real"]) > 1e-3
                end

                # In memory: B again on the warm problem, and B on the cold problem, against the
                # independent production evaluation at B on each problem.
                sw = Analysis.design_and_fields(warm, B)
                ref_w = production_reference(warm, B)
                sc = Analysis.design_and_fields(cold, B)
                ref_c = production_reference(cold, B)

                # Objective path at B, and the objective computed from the exported field.
                g_prod = objective_and_gradient!(Float64[], B, cold)
                exported = Dict{XDEO.CacheKey,CellField}(key => sc.E)
                g_exp = compute_objective(cold.objective, cold.pde, exported, sc.pt, cold.sim)

                sim_c, sim_w = default_sim(cold.sim), default_sim(warm.sim)
                E_cold, E_ref_c = get_free_dof_values(sc.E), get_free_dof_values(ref_c.fields[key])
                E_warm, E_ref_w = get_free_dof_values(sw.E), get_free_dof_values(ref_w.fields[key])
                ρ_cold, ρ_ref_c = sampled_density(sc.pt, sim_c), sampled_density(ref_c.pt, sim_c)
                ρ_warm, ρ_ref_w = sampled_density(sw.pt, sim_w), sampled_density(ref_w.pt, sim_w)

                # regression: the 3D density was the projection of the unfiltered design, so the
                # 3D density and field differed from production. Cold export against production.
                @test relerr(sc.pf_vec, ref_c.pf_vec) <= 1e-12
                @test relerr(ρ_cold, ρ_ref_c) <= 1e-12
                @test relerr(E_cold, E_ref_c) <= 1e-9
                @test isapprox(g_exp, g_prod; rtol=1e-9)

                # Warm export against production.
                @test relerr(sw.pf_vec, ref_w.pf_vec) <= 1e-12
                @test relerr(ρ_warm, ρ_ref_w) <= 1e-12
                @test relerr(E_warm, E_ref_w) <= 1e-9

                # Warm against cold. The two problems mesh the same geometry separately; compare
                # arrays only when the meshes are identical (the mesher is deterministic).
                if same_mesh(warm, cold)
                    @test relerr(sw.pf_vec, sc.pf_vec) <= 1e-12
                    @test relerr(ρ_warm, ρ_cold) <= 1e-12
                    @test relerr(E_warm, E_cold) <= 1e-9
                else
                    @test_skip "warm and cold problems have different meshes"
                end

                # The comparisons above can tell A from B.
                sa = Analysis.design_and_fields(warm, A)
                @test relerr(get_free_dof_values(sa.E), E_ref_w) > 1e-3
                @test relerr(sampled_density(sa.pt, sim_w), ρ_ref_w) > 1e-3

                # A design vector of the wrong length is refused by the export's own length check.
                err = observe(() -> Analysis.design_and_fields(warm, zeros(n + 1))).err
                @test err isa DimensionMismatch && occursin("design_and_fields: the design vector has", err.msg)
            end
        end
    end

    @testset "An export between a forward and an adjoint solve changes nothing" begin
        for (name, kw) in EXPORT_CASES
            @testset "$name" begin
                prob = build_test_problem(; kw...)
                (; pde, sim, pool) = prob
                n = length(prob.p)
                rng = MersenneTwister(2912)
                A = 0.05 .+ 0.40 .* rand(rng, n)
                B = 0.55 .+ 0.40 .* rand(rng, n)
                key = cache_key(pde.inputs[1])

                # Forward at A, then the adjoint with a fixed right-hand side, without an export.
                fields = solve_forward!(pde, XDEO.physical_density(A, prob).pt, sim, pool)
                sources = Dict(key => copy(get_free_dof_values(fields[key])))
                λ_plain = copy(get_free_dof_values(solve_adjoint!(pde, sources, sim, pool)[key]))

                # regression: the export replaced the factor of the first configuration with B's
                # (and its solution vector), so the pending adjoint changed and the cache objects
                # differed.
                snap = pool_snapshot(pool)
                mktempdir() do dir
                    Analysis.visualize_results(prob, B; root=dir)
                end
                @test pool_unchanged(pool, snap)
                λ_export = copy(get_free_dof_values(solve_adjoint!(pde, sources, sim, pool)[key]))
                @test relerr(λ_export, λ_plain) <= 1e-12

                # regression: an export to an unwritable folder throws after its solve, which had
                # already installed B's factor in the problem's cache.
                snap_err = pool_snapshot(pool)
                notdir = tempname()
                write(notdir, "a file, not a folder")
                try
                    @test_throws SystemError Analysis.visualize_results(prob, B; root=joinpath(notdir, "out"))
                finally
                    rm(notdir; force=true)
                end
                @test pool_unchanged(pool, snap_err)
                λ_failed = copy(get_free_dof_values(solve_adjoint!(pde, sources, sim, pool)[key]))
                @test relerr(λ_failed, λ_plain) <= 1e-12

                # Control: a forward solve at B on the problem's own caches does change the adjoint,
                # so the checks above can see a replaced factor.
                solve_forward!(pde, XDEO.physical_density(B, prob).pt, sim, pool)
                λ_B = copy(get_free_dof_values(solve_adjoint!(pde, sources, sim, pool)[key]))
                @test relerr(λ_B, λ_plain) > 1e-3
            end
        end

        @testset "two configurations, export of the second polarization" begin
            # regression: an export of a configuration other than the first overwrote that
            # configuration's factor or solution vector in place, under the same objects.
            prob = build_test_problem(; complex_config=true)   # y and x polarizations
            (; pde, sim, pool) = prob
            n = length(prob.p)
            rng = MersenneTwister(2917)
            A = 0.05 .+ 0.40 .* rand(rng, n)
            B = 0.55 .+ 0.40 .* rand(rng, n)
            xconfig = only(filter(fc -> fc.polarization == :x, pde.outputs))

            fields = solve_forward!(pde, XDEO.physical_density(A, prob).pt, sim, pool)
            @test length(fields) >= 2
            sources = Dict(k => copy(get_free_dof_values(f)) for (k, f) in fields)
            adjoints() = Dict(k => copy(get_free_dof_values(f)) for (k, f) in solve_adjoint!(pde, sources, sim, pool))
            λ_plain = adjoints()

            snap = pool_snapshot(pool)
            e = Analysis.design_and_fields(prob, B; config=xconfig)
            @test pool_unchanged(pool, snap)
            λ_export = adjoints()
            @test keys(λ_export) == keys(λ_plain)
            @test all(relerr(λ_export[k], λ_plain[k]) <= 1e-12 for k in keys(λ_plain))

            # The exported field is the second configuration's field at B.
            ref = solve_forward!(pde, XDEO.physical_density(B, prob).pt, sim, pool)
            @test relerr(get_free_dof_values(e.E), get_free_dof_values(ref[cache_key(xconfig)])) <= 1e-9
        end
    end

    @testset "An export releases its own factors, also when its solve fails" begin
        # regression: the export factorized in the problem's pool and never released (two live
        # factors after a 3D export, one more after the failed solve), and the problem's pool was
        # left holding them.
        prob = build_test_problem()                  # 3D: filter and Maxwell factors
        solver = CountingSolver()
        prob.pool = SolverCachePool(solver)
        pool = prob.pool
        B = 0.55 .+ 0.40 .* rand(MersenneTwister(2914), length(prob.p))

        Analysis.design_and_fields(prob, B)
        @test solver.made == 2                       # the filter and Maxwell factors were made here
        @test solver.live == 0
        @test isempty(pool.caches)
        @test pool.filter_cache.F_factor === nothing
        # regression: the export's filter factor was released only after the Maxwell solve, so the
        # two were live together (peak 2). It is released as soon as the density is built, before
        # Maxwell is factorized.
        @test solver.peak == 1

        solver.fail_maxwell = true
        @test_throws ErrorException Analysis.design_and_fields(prob, B)
        @test solver.made == 4
        @test solver.live == 0
        @test isempty(pool.caches)
        @test pool.filter_cache.F_factor === nothing
        solver.fail_maxwell = false

        # regression: the export factorized the filter and built the density before a configuration
        # its simulation refuses was refused by the forward solve. That refusal comes before any
        # factorization.
        tilted = FieldConfig(prob.pde.inputs[1].λ; tilt_x=10.0, pol=:y)   # crosses the x mirror wall
        @test_throws ArgumentError Analysis.design_and_fields(prob, B; config=tilted)
        @test solver.made == 4
        @test solver.live == 0
        # The mirror check above now comes from the export's own flag checks. A refusal only
        # `build_phys_params` makes, an incident index that is not the lossless fluid index of a
        # Bloch configuration, must also come before the density and its filter factorization.
        badindex = FieldConfig(prob.pde.inputs[1].λ; pol=:y, formulation=:bloch, n_incident=7.0)
        err = try Analysis.design_and_fields(prob, B; config=badindex); nothing catch e; e end
        @test err isa ArgumentError && occursin("n_incident must equal the lossless fluid index", err.msg)
        @test solver.made == 4
        @test solver.live == 0

        # A refusal that needs the density (a configuration whose simulation has another model)
        # comes after the density is built, as documented: the export's filter factor is made and
        # released, no factor is left live and the problem is left as it was.
        @test prob.sim isa XDEO.SimulationBundle
        other = build_test_problem()
        sim_x = prob.sim.by_pol[:x]
        prob.sim.by_pol[:x] = other.sim.by_pol[:x]
        try
            xconfig = FieldConfig(prob.pde.inputs[1].λ; pol=:x)
            st = problem_state(prob)
            made = solver.made
            @test_throws ArgumentError Analysis.design_and_fields(prob, B; config=xconfig)
            @test solver.made == made + 1                # the filter factor, before the refusal
            @test solver.live == 0
            @test isempty(pool.caches)
            @test pool.filter_cache.F_factor === nothing
            @test problem_unchanged(prob, st)
        finally
            prob.sim.by_pol[:x] = sim_x
        end

        # regression: on a warm problem the export's filter and Maxwell factors were live together
        # (resident + 2). The problem holds its filter and Maxwell factors after an objective
        # evaluation; an export adds one factor at a time on top of them, so two Maxwell factors of
        # the configuration are live during the export's solve.
        warm_solver = CountingSolver()
        prob.pool = SolverCachePool(warm_solver)
        objective_and_gradient!(Float64[], B, prob)
        resident = warm_solver.live
        @test resident == 2                          # the problem's filter and Maxwell factors
        warm_solver.peak = resident
        Analysis.design_and_fields(prob, B)
        @test warm_solver.peak == resident + 1
        @test warm_solver.live == resident
    end

    @testset "A 3D export leaves the problem's filter cache unchanged" begin
        prob = build_test_problem()                  # 3D
        sim0 = default_sim(prob.sim)
        fcache = default_pool(prob.pool).filter_cache
        # Start from an empty filter cache (no factor, no key), as on a problem not initialized by
        # `init_random!`.
        XDEO.clear_filter_factor!(fcache)
        @test fcache.F_factor === nothing && fcache.F_key === nothing
        p = 0.05 .+ 0.90 .* rand(MersenneTwister(2913), length(prob.p))
        function fresh_filter(R)
            ctrl = deepcopy(prob.control)
            ctrl.R_filter = R
            return filter_helmholtz!(p, SolverCache(UmfpackSolver()), sim0, ctrl)
        end
        R1, R2 = (5.0, 5.0, 5.0), (40.0, 40.0, 40.0)
        f1, f2 = fresh_filter(R1), fresh_filter(R2)
        @test relerr(f1, f2) > 1e-3                  # control: the two radii filter differently

        # regression: the export factorized the filter in the problem's cache.
        prob.control.R_filter = R1
        snap = pool_snapshot(prob.pool)
        e1 = Analysis.design_and_fields(prob, p)
        @test default_pool(prob.pool).filter_cache === fcache
        @test fcache.F_factor === nothing
        @test fcache.F_key === nothing
        @test pool_unchanged(prob.pool, snap)
        @test relerr(e1.pf_vec, f1) <= 1e-12

        # regression: an export after a radius change reused the factor the first export left in
        # the problem's cache, and returned the radius-R1 filter.
        prob.control.R_filter = R2
        e2 = Analysis.design_and_fields(prob, p)
        @test relerr(e2.pf_vec, f2) <= 1e-12
        @test fcache.F_factor === nothing
        @test fcache.F_key === nothing

        # regression: the next production density (the optimization path) after the exports used
        # the radius-R1 factor the first export left behind.
        d2 = XDEO.physical_density(p, prob)
        @test relerr(d2.pf_vec, f2) <= 1e-12

        # With a primed problem cache, an export leaves its factor object and key in place.
        snap2 = pool_snapshot(prob.pool)
        F_primed, key_primed = fcache.F_factor, fcache.F_key
        @test F_primed !== nothing && key_primed !== nothing
        Analysis.design_and_fields(prob, p)
        @test fcache.F_factor === F_primed
        @test fcache.F_key === key_primed
        @test pool_unchanged(prob.pool, snap2)

        # Primed at another radius: the problem's cache holds the radius-R2 factor and key, the
        # export at R1 uses R1 and leaves that factor and key in place, and the next production
        # density replaces both and uses R1.
        prob.control.R_filter = R1
        snap3 = pool_snapshot(prob.pool)
        e3 = Analysis.design_and_fields(prob, p)
        @test relerr(e3.pf_vec, f1) <= 1e-12
        @test fcache.F_factor === F_primed
        @test fcache.F_key === key_primed
        @test pool_unchanged(prob.pool, snap3)
        d3 = XDEO.physical_density(p, prob)
        @test relerr(d3.pf_vec, f1) <= 1e-12
        @test fcache.F_factor !== F_primed
        @test fcache.F_key !== nothing && !isequal(fcache.F_key, key_primed)
    end

    @testset "A refused, successful or failed export leaves a problem that has run as it was" begin
        for (name, kw) in EXPORT_CASES
            @testset "$name" begin
                prob = build_test_problem(; kw...)
                n = length(prob.p)
                rng = MersenneTwister(2915)
                A = 0.05 .+ 0.40 .* rand(rng, n)
                B = 0.55 .+ 0.40 .* rand(rng, n)
                C = 0.30 .+ 0.40 .* rand(rng, n)     # exported only by the failing write
                # A problem that has run: caches primed (and, in foundry mode, the grid written) by an
                # objective evaluation at A; history, best record and iteration set to sentinels.
                g_A = objective_and_gradient!(Float64[], A, prob)
                prob.g = g_A
                prob.g_history = [0.5 * g_A, g_A]
                prob.p_best = copy(A)
                prob.g_best = g_A
                prob.iteration = 2
                st = problem_state(prob)
                snap = pool_snapshot(prob.pool)
                prob.foundry_mode && @test st.grid[1] != reshape(written_grid(B, prob), size(st.grid[1]))

                # Each case below starts from a state recorded just before it, so a change made by an
                # earlier case cannot be blamed on a later one.

                # regression: in both foundry modes, the export built B's density, writing B's grid
                # values into the shared `sim.grid.params` (read by `pf_grid`), before the forward
                # solve refused the configuration.
                tilted = FieldConfig(prob.pde.inputs[1].λ; tilt_x=10.0, pol=:y)   # crosses the x mirror wall
                @test_throws ArgumentError Analysis.design_and_fields(prob, B; config=tilted)
                @test problem_unchanged(prob, st)
                @test pool_unchanged(prob.pool, snap)

                mktempdir() do dir
                    # regression: a wrong-length design was refused by `visualize_results` only after
                    # its announcement was printed. It is refused before anything is printed or written.
                    st = problem_state(prob)
                    snap = pool_snapshot(prob.pool)
                    r = observe(() -> Analysis.visualize_results(prob, zeros(n + 1); root=dir))
                    @test r.err isa DimensionMismatch && occursin("visualize_results", r.err.msg)
                    @test isempty(r.printed)                # refused before its announcement
                    @test isempty(readdir(dir))
                    @test problem_unchanged(prob, st)
                    @test pool_unchanged(prob.pool, snap)

                    # regression: in both foundry modes, a successful export wrote B's grid values
                    # into the shared grid.
                    st = problem_state(prob)
                    snap = pool_snapshot(prob.pool)
                    Analysis.visualize_results(prob, B; root=dir)
                    @test problem_unchanged(prob, st)
                    @test pool_unchanged(prob.pool, snap)
                end

                # regression: in both foundry modes, an export whose write fails after its solve had
                # written its grid values into the shared grid. Its design C differs from B, so a grid
                # write by this call is seen even if the successful export above had written B.
                st = problem_state(prob)
                snap = pool_snapshot(prob.pool)
                prob.foundry_mode && @test st.grid[1] != reshape(written_grid(C, prob), size(st.grid[1]))
                notdir = tempname()
                write(notdir, "a file, not a folder")
                try
                    @test_throws SystemError Analysis.visualize_results(prob, C; root=joinpath(notdir, "out"))
                finally
                    rm(notdir; force=true)
                end
                @test problem_unchanged(prob, st)
                @test pool_unchanged(prob.pool, snap)

                # Control: an objective evaluation at B does write the grid in foundry mode, so the
                # checks above can see a grid write.
                if prob.foundry_mode
                    objective_and_gradient!(Float64[], B, prob)
                    @test !problem_unchanged(prob, st)
                end
            end
        end
    end

    @testset "Exported density and field stay valid when the backend clears its solution storage on release" begin
        # regression: on a backend that solves into storage owned by each factor and clears it when
        # the factor is released, the export returned and wrote a zero filtered density and a zero
        # field: both were built on the solver's vectors, which the export released before use.
        for (name, kw) in EXPORT_CASES
            @testset "$name" begin
                prob = build_test_problem(; kw...)
                sim0 = default_sim(prob.sim)
                config = prob.pde.inputs[1]
                B = 0.55 .+ 0.40 .* rand(MersenneTwister(2916), length(prob.p))

                ref = Analysis.design_and_fields(prob, B)        # on the problem's own backend (UMFPACK)
                @test norm(ref.pf_vec) > 0
                @test norm(get_free_dof_values(ref.E)) > 0

                prob.pool = SolverCachePool(WorkspaceSolver())
                ws = Analysis.design_and_fields(prob, B)
                @test relerr(ws.pf_vec, ref.pf_vec) <= 1e-10
                @test relerr(sampled_density(ws.pt, sim0), sampled_density(ref.pt, sim0)) <= 1e-10
                @test relerr(get_free_dof_values(ws.E), get_free_dof_values(ref.E)) <= 1e-9

                # The files written on this backend hold the same arrays.
                mktempdir() do dir
                    Analysis.visualize_results(prob, B; root=dir)
                    (des, fld) = exported_arrays(dir)
                    (rdes, rfld) = reference_arrays(dir, prob, ref.pt, ref.E, config)
                    @test maximum(abs, rdes["p"]) > 0.1
                    @test relerr(des["p"], rdes["p"]) <= 1e-10
                    @test relerr(fld["E_real"], rfld["E_real"]) <= 1e-9
                    @test relerr(fld["E_imag"], rfld["E_imag"]) <= 1e-9
                end
            end
        end
    end

    @testset "A backend whose new factor empties another factor's buffer is refused" begin
        # regression: on a backend with a solution buffer per factor that empties the previous
        # factor's idle buffer when it makes a new factor, a successful export left the field a
        # caller had kept from a forward solve with no entries. Such a backend declares
        # `returns_owned_solutions` false, and both entry points refuse it before any work.
        for (name, kw) in EXPORT_CASES
            @testset "$name" begin
                prob = build_test_problem(; kw...)
                (; pde, sim) = prob
                n = length(prob.p)
                rng = MersenneTwister(2920)
                A = 0.05 .+ 0.40 .* rand(rng, n)
                B = 0.55 .+ 0.40 .* rand(rng, n)
                trim = TrimSolver()
                prob.pool = SolverCachePool(trim)
                # A problem that has run: forward fields at A kept by the caller, history set.
                fields = solve_forward!(pde, XDEO.physical_density(A, prob).pt, sim, prob.pool)
                prob.g_history = [1.0, 2.0]
                prob.p_best = copy(A)
                prob.g_best = 2.0
                prob.iteration = 2
                E_A = Dict(k => copy(get_free_dof_values(f)) for (k, f) in fields)
                @test !isempty(E_A) && all(!isempty, values(E_A))
                st = problem_state(prob)
                snap = pool_snapshot(prob.pool)
                c0 = trim_counts(trim)

                r = observe(() -> Analysis.design_and_fields(prob, B))
                @test refusal_names(r.err, "design_and_fields", "returns_owned_solutions")
                @test trim_counts(trim) == c0               # no factorization, solve or release
                @test all(get_free_dof_values(fields[k]) == E_A[k] for k in keys(E_A))
                @test pool_unchanged(prob.pool, snap)
                @test problem_unchanged(prob, st)

                mktempdir() do dir
                    r = observe(() -> Analysis.visualize_results(prob, B; root=dir))
                    @test refusal_names(r.err, "visualize_results", "returns_owned_solutions")
                    @test isempty(r.printed)
                    @test isempty(readdir(dir))
                end
                @test trim_counts(trim) == c0
                @test all(get_free_dof_values(fields[k]) == E_A[k] for k in keys(E_A))
                @test pool_unchanged(prob.pool, snap)
                @test problem_unchanged(prob, st)

                # Control: a new factor on this backend empties the kept field, so a factor made by
                # an export would be seen above.
                XDEO.lu!(trim, ComplexF64[2 0; 0 3])
                @test any(isempty(get_free_dof_values(f)) for f in values(fields))
            end
        end
    end

    @testset "A backend that declares its solutions not owned is refused before any work" begin
        # regression: on a backend with one reusable solution vector on the solver object, an
        # export's solve wrote its field into the vector the problem's cache held, and so into the
        # fields a caller had kept from a forward solve; detection after the solve could not undo
        # every case. The declaration is checked first, at both entry points.
        prob = build_test_problem(; complex_config=true)   # y and x polarizations, one bundle
        (; pde, sim) = prob
        n = length(prob.p)
        rng = MersenneTwister(2918)
        A = 0.05 .+ 0.40 .* rand(rng, n)
        B = 0.55 .+ 0.40 .* rand(rng, n)
        yconfig = pde.inputs[1]
        xconfig = only(filter(fc -> fc.polarization == :x, pde.outputs))
        bundle(ysolver, xsolver) = (yp = SolverCachePool(ysolver);
            XDEO.SolverPoolBundle(yp, Dict(:y => yp, :x => SolverCachePool(xsolver))))

        # A problem that has run on `pool`: forward fields at A kept by the caller, history set.
        function run_at_A!(pool)
            prob.pool = pool
            fields = solve_forward!(pde, XDEO.physical_density(A, prob).pt, sim, pool)
            prob.g_history = [1.0, 2.0]
            prob.p_best = copy(A)
            prob.g_best = 2.0
            prob.iteration = 2
            return fields
        end
        kept(fields) = Dict(k => copy(get_free_dof_values(f)) for (k, f) in fields)
        counts(solvers...) = [(s.made, s.solves) for s in solvers if s isa SharedSolutionSolver]

        # Both pools declare false (one solver object per polarization).
        ys, xs = SharedSolutionSolver(), SharedSolutionSolver()
        fields = run_at_A!(bundle(ys, xs))
        @test length(fields) >= 2
        @test prob.pool.by_pol[:y].caches[cache_key(yconfig)].x === ys.x   # the caches hold the shared vectors
        @test prob.pool.by_pol[:x].caches[cache_key(xconfig)].x === xs.x
        E_A = kept(fields)
        for (label, call) in (
            ("design_and_fields, y", () -> Analysis.design_and_fields(prob, B; config=yconfig)),
            ("design_and_fields, x", () -> Analysis.design_and_fields(prob, B; config=xconfig)))
            @testset "$label" begin
                st = problem_state(prob)
                snap = pool_snapshot(prob.pool)
                c0 = counts(ys, xs)
                @test refusal_names(observe(call).err, "design_and_fields", "returns_owned_solutions")
                @test counts(ys, xs) == c0                  # no factorization, no solve
                @test pool_unchanged(prob.pool, snap)       # same caches, factors and vectors, same values
                @test all(get_free_dof_values(fields[k]) == E_A[k] for k in keys(E_A))
                @test problem_unchanged(prob, st)
            end
        end
        @testset "visualize_results" begin
            st = problem_state(prob)
            snap = pool_snapshot(prob.pool)
            c0 = counts(ys, xs)
            mktempdir() do dir
                r = observe(() -> Analysis.visualize_results(prob, B; root=dir))
                @test refusal_names(r.err, "visualize_results", "returns_owned_solutions")
                @test isempty(r.printed)                    # refused before its first message
                @test isempty(readdir(dir))
            end
            @test counts(ys, xs) == c0
            @test pool_unchanged(prob.pool, snap)
            @test all(get_free_dof_values(fields[k]) == E_A[k] for k in keys(E_A))
            @test problem_unchanged(prob, st)
        end

        # Each solver an export uses is checked. The default pool's solver (the filter's) declares
        # false and the x pool's does not: an export of x, which needs the 3D filter solve, is
        # refused before the filter solve, and the accepting x solver does no work either.
        ys2, xt2 = SharedSolutionSolver(), TallySolver()
        fields2 = run_at_A!(bundle(ys2, xt2))
        E2 = kept(fields2)
        snap2 = pool_snapshot(prob.pool)
        c2, t2 = counts(ys2), tally(xt2)
        @test refusal_names(observe(() -> Analysis.design_and_fields(prob, B; config=xconfig)).err,
            "design_and_fields", "returns_owned_solutions")
        @test counts(ys2) == c2
        @test tally(xt2) == t2
        @test pool_unchanged(prob.pool, snap2)
        @test all(get_free_dof_values(fields2[k]) == E2[k] for k in keys(E2))

        # The x pool's solver (Maxwell's for x) declares false and the default pool's does not: an
        # export of x is refused with no work on either solver, and an export of y, which uses only
        # the default pool, is not.
        yt3, xs3 = TallySolver(), SharedSolutionSolver()
        fields3 = run_at_A!(bundle(yt3, xs3))
        E3 = kept(fields3)
        snap3 = pool_snapshot(prob.pool)
        c3, t3 = counts(xs3), tally(yt3)
        @test refusal_names(observe(() -> Analysis.design_and_fields(prob, B; config=xconfig)).err,
            "design_and_fields", "returns_owned_solutions")
        @test counts(xs3) == c3
        @test tally(yt3) == t3
        @test pool_unchanged(prob.pool, snap3)
        @test all(get_free_dof_values(fields3[k]) == E3[k] for k in keys(E3))
        ey = Analysis.design_and_fields(prob, B; config=yconfig)
        @test norm(get_free_dof_values(ey.E)) > 0
        @test counts(xs3) == c3
        @test tally(yt3)[1] > t3[1]                         # the accepted export did its work on y

        # Control: B's x field differs from A's, so a retained vector left holding B would be seen.
        ref = solve_forward!(pde, XDEO.physical_density(B, prob).pt, sim, bundle(UmfpackSolver(), UmfpackSolver()))
        kx = cache_key(xconfig)
        @test relerr(get_free_dof_values(ref[kx]), E_A[kx]) > 1e-3
    end

    @testset "A foundry export does not check a default pool solver it does not use" begin
        # regression: a foundry export of x was refused because the default pool's solver declared
        # `returns_owned_solutions` false, although the grid filter never solves on it.
        prob = build_test_problem(; foundry_mode=true, complex_config=true)   # y and x, one bundle
        (; pde, sim) = prob
        n = length(prob.p)
        rng = MersenneTwister(2922)
        A = 0.05 .+ 0.40 .* rand(rng, n)
        B = 0.55 .+ 0.40 .* rand(rng, n)
        yconfig = pde.inputs[1]
        xconfig = only(filter(fc -> fc.polarization == :x, pde.outputs))
        ys, xt = SharedSolutionSolver(), TallySolver()
        prob.pool = two_pol_bundle(ys, xt)
        fields = solve_forward!(pde, XDEO.physical_density(A, prob).pt, sim, prob.pool)
        E_A = Dict(k => copy(get_free_dof_values(f)) for (k, f) in fields)
        snap = pool_snapshot(prob.pool)
        c0, t0 = (ys.made, ys.solves), tally(xt)

        ex = Analysis.design_and_fields(prob, B; config=xconfig)
        @test norm(get_free_dof_values(ex.E)) > 0
        @test (ys.made, ys.solves) == c0                    # nothing on the default pool's solver
        @test tally(xt) .- t0 == (1, 1, 1)                  # one Maxwell factor, one solve, one release
        @test all(get_free_dof_values(fields[k]) == E_A[k] for k in keys(E_A))
        @test pool_unchanged(prob.pool, snap)

        # The default pool's solver is still checked when it is the exported configuration's
        # Maxwell solver: an export of y is refused with no work on either solver.
        t1 = tally(xt)
        @test refusal_names(observe(() -> Analysis.design_and_fields(prob, B; config=yconfig)).err,
            "design_and_fields", "returns_owned_solutions")
        @test (ys.made, ys.solves) == c0
        @test tally(xt) == t1
        @test pool_unchanged(prob.pool, snap)
    end

    @testset "The export's density helper refuses a filter solver that declares false" begin
        # regression: the internal density helper of the export factorized and solved the 3D filter
        # on a backend that declares `returns_owned_solutions` false; only its caller checked.
        prob = build_test_problem()                  # 3D, filter on
        B = 0.55 .+ 0.40 .* rand(MersenneTwister(2923), length(prob.p))
        ss = SharedSolutionSolver()
        prob.pool = SolverCachePool(ss)
        snap = pool_snapshot(prob.pool)
        r = observe(() -> Analysis.export_density(B, prob))
        @test refusal_names(r.err, "export_density", "returns_owned_solutions")
        @test (ss.made, ss.solves) == (0, 0)
        @test pool_unchanged(prob.pool, snap)
        # The objective's density path does not check the declaration and stays usable.
        d = XDEO.physical_density(B, prob)
        @test norm(d.pf_vec) > 0
        @test (ss.made, ss.solves) == (1, 1)
    end

    @testset "A foundry export refuses use_dct=false before it logs or loads a material table" begin
        # regression: `design_and_fields` built the configuration's physical parameters, which
        # loaded the design's material table into the package's material cache and logged a
        # diffraction-order warning for a near-cutoff Bloch configuration, before the grid filter
        # refused `use_dct=false`.
        tl = TallySolver()                               # accepts: the backend check passes
        prob = periodic_foundry_problem(UmfpackSolver())
        prob.pool = SolverCachePool(tl)
        (; pde, sim) = prob
        n = length(prob.p)
        rng = MersenneTwister(2924)
        A = 0.05 .+ 0.40 .* rand(rng, n)
        B = 0.55 .+ 0.40 .* rand(rng, n)
        periods = XDEO.lattice_periods(XDEO.sim_for(sim, pde.inputs[1]))
        bloch = FieldConfig(1.0001 * maximum(periods); pol=:y, formulation=:bloch)   # just above cutoff

        # A problem that has run with use_dct=true: forward fields at A kept by the caller, the grid
        # written, history set. Then the control is switched to use_dct=false.
        fields = solve_forward!(pde, XDEO.physical_density(A, prob).pt, sim, prob.pool)
        prob.g_history = [1.0, 2.0]
        prob.p_best = copy(A)
        prob.g_best = 2.0
        prob.iteration = 2
        E_A = Dict(k => copy(get_free_dof_values(f)) for (k, f) in fields)
        prob.control.use_dct = false
        st = problem_state(prob)
        snap = pool_snapshot(prob.pool)
        t0 = tally(tl)
        saved_cache = copy(XDEO._MATERIAL_CACHE)         # restored at the end of this testset
        XDEO.clear_material_cache!()                     # so a table loaded by the call is seen
        keys0 = Set(keys(XDEO._MATERIAL_CACHE))
        unchanged() = tally(tl) == t0 && Set(keys(XDEO._MATERIAL_CACHE)) == keys0 &&
                      problem_unchanged(prob, st) && pool_unchanged(prob.pool, snap) &&
                      all(get_free_dof_values(fields[k]) == E_A[k] for k in keys(E_A))

        for (label, config) in (("near-cutoff Bloch configuration", bloch), ("first input", pde.inputs[1]))
            @testset "design_and_fields, $label" begin
                XDEO.clear_material_cache!()             # each refusal from its own cold cache
                r = observe(() -> Analysis.design_and_fields(prob, B; config))
                @test refusal_names(r.err, "design_and_fields", "use_dct=true")
                @test isempty(r.logs)
                @test isempty(r.printed)
                @test tally(tl) == t0                    # no factorization, solve or release
                @test Set(keys(XDEO._MATERIAL_CACHE)) == keys0
                @test unchanged()
            end
        end
        @testset "visualize_results" begin
            XDEO.clear_material_cache!()
            mktempdir() do dir
                r = observe(() -> Analysis.visualize_results(prob, B; root=dir))
                @test refusal_names(r.err, "visualize_results", "use_dct=true")
                @test isempty(r.logs)
                @test isempty(r.printed)
                @test isempty(readdir(dir))
            end
            @test unchanged()
        end

        # Control: building the Bloch configuration's physical parameters logs a warning and loads
        # the design's material table, so the checks above can see both.
        logger = Test.TestLogger(min_level=Logging.Debug)
        with_logger(logger) do
            XDEO.build_phys_params(bloch, pde.env, XDEO.sim_for(sim, bloch); α=pde.α_loss)
        end
        @test any(rec -> rec.level == Logging.Warn, logger.logs)
        @test haskey(XDEO._MATERIAL_CACHE, "Ag")
        prob.control.use_dct = true
        empty!(XDEO._MATERIAL_CACHE)
        merge!(XDEO._MATERIAL_CACHE, saved_cache)
    end

    @testset "An export uses the solvers it checked, also when a pool is replaced during the call" begin
        # regression: `design_and_fields` checked the Maxwell solver of the configuration's pool,
        # then built the configuration's physical parameters, which logged a warning for a
        # near-cutoff Bloch configuration, and then looked the pool up again. A logger that
        # replaced the pool during that warning made the export solve on a backend that declares
        # `returns_owned_solutions` false, which it had never checked. Each entry point now reads
        # its solver objects once at entry and builds its caches on them.
        tl = TallySolver()
        prob = periodic_foundry_problem(UmfpackSolver())
        (; pde, sim) = prob
        B = 0.55 .+ 0.40 .* rand(MersenneTwister(2925), length(prob.p))
        periods = XDEO.lattice_periods(XDEO.sim_for(sim, pde.inputs[1]))
        bloch = FieldConfig(1.0001 * maximum(periods); pol=:y, formulation=:bloch)   # warns
        saved_pde = prob.pde
        for entry in ("design_and_fields", "visualize_results")
            @testset "$entry" begin
                prob.pool = SolverCachePool(tl)
                trim = TrimSolver()
                t0 = tally(tl)
                switch = ActOnWarn(() -> (prob.pool = SolverCachePool(trim)))
                mktempdir() do dir
                    with_logger(switch) do
                        if entry == "design_and_fields"
                            out = Analysis.design_and_fields(prob, B; config=bloch)
                            @test norm(get_free_dof_values(out.E)) > 0
                        else
                            prob.pde = MaxwellProblem(env=saved_pde.env, inputs=[bloch], outputs=FieldConfig[])
                            try
                                redirect_stdout(devnull) do
                                    Analysis.visualize_results(prob, B; root=dir)
                                end
                            finally
                                prob.pde = saved_pde
                            end
                            @test isfile(joinpath(dir, "fields.vtu"))
                        end
                    end
                    @test switch.fired[] >= 1                    # the pool was replaced mid-call
                    @test trim_counts(trim) == (0, 0, 0)         # the replacement did no work
                    t1 = tally(tl)
                    @test t1[1] > t0[1] && t1[2] > t0[2]          # the checked solver did
                end
            end
        end
    end

    @testset "The export's density helper checks the design before it builds a cache" begin
        # regression: `export_density` accepted a design one entry too long (one factorization,
        # solve and release) and factorized before it refused an empty one (`BoundsError`); on a
        # foundry problem its `use_dct` refusal came from the grid filter, naming `filter_grid`.
        prob = build_test_problem()                  # 3D, filter on
        tl = TallySolver()
        prob.pool = SolverCachePool(tl)
        n = length(prob.p)
        for m in (n + 1, 0)
            r = observe(() -> Analysis.export_density(fill(0.5, m), prob))
            @test r.err isa DimensionMismatch && occursin("export_density", r.err.msg)
            @test tally(tl) == (0, 0, 0)
        end
        fprob = periodic_foundry_problem(UmfpackSolver())
        fprob.control.use_dct = false
        try
            r = observe(() -> Analysis.export_density(fill(0.5, length(fprob.p)), fprob))
            @test refusal_names(r.err, "export_density", "use_dct=true")
        finally
            fprob.control.use_dct = true
        end
    end

    @testset "Both entry points refuse the backend first, use_dct second, the length third" begin
        # regression: `visualize_results` refused `use_dct=false` before a backend that declares
        # `returns_owned_solutions` false, the reverse of `design_and_fields`.
        prob = periodic_foundry_problem(UmfpackSolver())
        n = length(prob.p)
        calls = (("design_and_fields", p -> Analysis.design_and_fields(prob, p)),
                 ("visualize_results", p -> mktempdir(d -> Analysis.visualize_results(prob, p; root=d))))
        try
            for (entry, f) in calls
                prob.control.use_dct = false
                prob.pool = SolverCachePool(TrimSolver())              # all three invalid
                r = observe(() -> f(fill(0.5, n + 1)))
                @test refusal_names(r.err, entry, "returns_owned_solutions")
                @test isempty(r.printed)
                prob.pool = SolverCachePool(TallySolver())             # use_dct and length invalid
                r = observe(() -> f(fill(0.5, n + 1)))
                @test refusal_names(r.err, entry, "use_dct=true")
                @test isempty(r.printed)
                prob.control.use_dct = true                            # length only
                r = observe(() -> f(fill(0.5, n + 1)))
                @test r.err isa DimensionMismatch && occursin(entry, r.err.msg)
                @test isempty(r.printed)
            end
        finally
            prob.control.use_dct = true
        end
    end

    @testset "Control and configuration flag refusals come before any material table or print" begin
        # regression: with a valid design, use_dct=true and an accepting solver, the export built
        # the configuration's physical parameters (loading the design's material table) and
        # `visualize_results` printed its announcement before the projection refusals (legacy
        # mode with sidewalls, an unsupported projection mode), a tilt across a mirror wall, or
        # the Bloch formulation with the ±z PML were raised. On main the projection refusals
        # came before the parameters were built.
        with_saved_material_cache() do
            tl = TallySolver()
            prob = periodic_foundry_problem(UmfpackSolver())
            prob.pool = SolverCachePool(tl)
            B = fill(0.5, length(prob.p))
            control = prob.control
            saved_pde = prob.pde
            saved_mode, saved_sidewall = control.foundry_projection_mode, control.use_sidewall
            tilted = FieldConfig(532.0; θ=10.0, pol=:y)               # crosses the y mirror walls
            periods = XDEO.lattice_periods(XDEO.sim_for(prob.sim, saved_pde.inputs[1]))
            bloch = FieldConfig(1.0001 * maximum(periods); pol=:y, formulation=:bloch)
            pml_pde = MaxwellProblem(env=Environment(mat_design="Ag", mat_substrate=1.0, mat_fluid=1.0,
                pml=XDEO.ZPML()), inputs=[bloch], outputs=FieldConfig[])
            cases = (
                ("legacy projection with sidewalls", ArgumentError, "sidewalls need",
                    () -> (control.foundry_projection_mode = :legacy; control.use_sidewall = true), saved_pde),
                ("unsupported projection mode", ErrorException, "Unsupported foundry_projection_mode",
                    () -> (control.foundry_projection_mode = :bogus), saved_pde),
                ("tilt across a mirror wall", ArgumentError, "crosses mirror wall FacesY",
                    () -> nothing, MaxwellProblem(env=saved_pde.env, inputs=[tilted], outputs=FieldConfig[])),
                ("Bloch formulation with the ±z PML", ArgumentError, "PML is not implemented",
                    () -> nothing, pml_pde))
            for (label, T, part, setup!, pde) in cases
                @testset "$label" begin
                    try
                        setup!()
                        prob.pde = pde
                        config = pde.inputs[1]
                        for (entry, f) in (("design_and_fields", () -> Analysis.design_and_fields(prob, B; config)),
                                           ("visualize_results", () -> mktempdir() do d
                                                Analysis.visualize_results(prob, B; root=d)
                                                readdir(d)
                                            end))
                            XDEO.clear_material_cache!()          # each refusal from a cold cache
                            t0 = tally(tl)
                            r = observe(f)
                            @test r.err isa T && occursin(part, sprint(showerror, r.err))
                            @test isempty(r.logs)
                            @test isempty(r.printed)
                            @test isempty(XDEO._MATERIAL_CACHE)
                            @test tally(tl) == t0
                        end
                    finally
                        control.foundry_projection_mode, control.use_sidewall = saved_mode, saved_sidewall
                        prob.pde = saved_pde
                    end
                end
            end
        end
    end

    @testset "Export cleanup clears the filter factor and its key" begin
        # regression: export cleanup detached the filter factor but left its operator key in the
        # cache, also when the release threw.
        prob = build_test_problem()                  # 3D
        p = 0.05 .+ 0.90 .* rand(MersenneTwister(2919), length(prob.p))
        for (label, solver) in (("release returns", UmfpackSolver()), ("release throws", ThrowingReleaseSolver()))
            @testset "$label" begin
                c = SolverCache(solver)
                XDEO.physical_density(p, prob; filter_cache=c)
                @test c.F_factor !== nothing && c.F_key !== nothing   # a keyed filter cache
                if solver isa ThrowingReleaseSolver
                    @test_throws ErrorException Analysis.release_analysis_cache!(c)
                else
                    Analysis.release_analysis_cache!(c)
                end
                @test c.F_factor === nothing
                @test c.F_key === nothing
            end
        end
    end
end

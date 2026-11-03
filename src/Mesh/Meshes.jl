"""
    Meshes

Gmsh-based mesh generation for periodic unit cells.
Port of genmesh from Emitter3DTopOpt.
"""

import Gmsh: gmsh

# ═══════════════════════════════════════════════════════════════════════════════
# Geometry types
# ═══════════════════════════════════════════════════════════════════════════════

abstract type AbstractGeometry end

"""
    SymmetricGeometry

Standard layered geometry for SERS optimization.

## Layers (bottom to top)
1. Substrate (height: hsub)
2. Design region (height: hd)
3. Target/Raman region (from design top to ht above design)
4. Source plane (at hs above design)
5. Air (up to hair total above design)
"""
mutable struct SymmetricGeometry <: AbstractGeometry
    L::Float64           # Period in x
    W::Float64           # Period in y
    hair::Float64        # Total air height above design
    hs::Float64          # Source plane height above design
    ht::Float64          # Target (Raman) top height above design
    hd::Float64          # Design region height
    hsub::Float64        # Substrate height
    dpml::Float64        # PML thickness (meshed only with genmesh(...; pml=true))
    l1::Float64          # Air mesh resolution
    l2::Float64          # Design mesh resolution (finest)
    l3::Float64          # Substrate mesh resolution
    verbose::Int         # Gmsh verbosity (0=quiet, 1=normal)
end

"""Convenience constructor from wavelength."""
function SymmetricGeometry(λ::Float64;
    L::Float64=400.0, W::Float64=400.0,
    hd::Float64=200.0, hsub::Float64=100.0)
    hr = λ / sqrt(1.77) / 4  # Quarter-wave Raman offset
    hair = 500.0 + hr
    hs = 300.0 + hr
    ht = 200.0 + hr
    l1 = 40.0
    l2 = 12.0
    l3 = l1
    SymmetricGeometry(L, W, hair, hs, ht, hd, hsub, 300.0, l1, l2, l3, 0)
end

SymmetricGeometry() = SymmetricGeometry(540.0)

# ═══════════════════════════════════════════════════════════════════════════════
# Tag tracker
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct TagTracker
    tp::Int    # Points
    tl::Int    # Lines
    tcl::Int   # Curve loops
    tps::Int   # Plane surfaces
    tsl::Int   # Surface loops
    tv::Int    # Volumes
    tpg::Int   # Physical groups
    TagTracker() = new(0, 0, 0, 0, 0, 0, 0)
end

itp!(t::TagTracker) = (t.tp += 1)
itl!(t::TagTracker) = (t.tl += 1)
itcl!(t::TagTracker) = (t.tcl += 1)
itps!(t::TagTracker) = (t.tps += 1)
itsl!(t::TagTracker) = (t.tsl += 1)
itv!(t::TagTracker) = (t.tv += 1)
itpg!(t::TagTracker) = (t.tpg += 1)

# ═══════════════════════════════════════════════════════════════════════════════
# Box primitive
# ═══════════════════════════════════════════════════════════════════════════════

"""8-corner box with all geometric entities."""
mutable struct Box
    # 8 corner points
    p_fll::Any
    p_flr::Any
    p_ful::Any
    p_fur::Any
    p_bll::Any
    p_blr::Any
    p_bul::Any
    p_bur::Any
    # 12 edges
    l_fb::Any
    l_fr::Any
    l_ft::Any
    l_fl::Any
    l_bb::Any
    l_br::Any
    l_bt::Any
    l_bl::Any
    l_f2bll::Any
    l_f2blr::Any
    l_f2bul::Any
    l_f2bur::Any
    # 6 curve loops and surfaces
    cl_f::Any
    cl_r::Any
    cl_b::Any
    cl_l::Any
    cl_t::Any
    cl_u::Any
    ps_f::Any
    ps_r::Any
    ps_b::Any
    ps_l::Any
    ps_t::Any
    ps_u::Any
    # Volume
    sl::Any
    vol::Any
end
Box() = Box([nothing for _ in 1:34]...)

"""Create a box from 8 corners (x, y, z, resolution)."""
function box(tt::TagTracker, fll, flr, ful, fur, bll, blr, bul, bur; mod=gmsh.model.geo, kwargs...)
    bx = Box()
    kkeys = keys(kwargs)

    # Points
    bx.p_fll = (:p_fll in kkeys) ? getindex(kwargs, :p_fll) : (typeof(fll) <: Tuple ? mod.addPoint(fll..., itp!(tt)) : fll)
    bx.p_flr = (:p_flr in kkeys) ? getindex(kwargs, :p_flr) : (typeof(flr) <: Tuple ? mod.addPoint(flr..., itp!(tt)) : flr)
    bx.p_ful = (:p_ful in kkeys) ? getindex(kwargs, :p_ful) : (typeof(ful) <: Tuple ? mod.addPoint(ful..., itp!(tt)) : ful)
    bx.p_fur = (:p_fur in kkeys) ? getindex(kwargs, :p_fur) : (typeof(fur) <: Tuple ? mod.addPoint(fur..., itp!(tt)) : fur)
    bx.p_bll = (:p_bll in kkeys) ? getindex(kwargs, :p_bll) : (typeof(bll) <: Tuple ? mod.addPoint(bll..., itp!(tt)) : bll)
    bx.p_blr = (:p_blr in kkeys) ? getindex(kwargs, :p_blr) : (typeof(blr) <: Tuple ? mod.addPoint(blr..., itp!(tt)) : blr)
    bx.p_bul = (:p_bul in kkeys) ? getindex(kwargs, :p_bul) : (typeof(bul) <: Tuple ? mod.addPoint(bul..., itp!(tt)) : bul)
    bx.p_bur = (:p_bur in kkeys) ? getindex(kwargs, :p_bur) : (typeof(bur) <: Tuple ? mod.addPoint(bur..., itp!(tt)) : bur)

    # Lines (12 edges)
    bx.l_fb = (:l_fb in kkeys) ? -getindex(kwargs, :l_fb) : mod.addLine(bx.p_fll, bx.p_flr, itl!(tt))
    bx.l_fr = (:l_fr in kkeys) ? -getindex(kwargs, :l_fr) : mod.addLine(bx.p_flr, bx.p_fur, itl!(tt))
    bx.l_ft = (:l_ft in kkeys) ? -getindex(kwargs, :l_ft) : mod.addLine(bx.p_fur, bx.p_ful, itl!(tt))
    bx.l_fl = (:l_fl in kkeys) ? -getindex(kwargs, :l_fl) : mod.addLine(bx.p_ful, bx.p_fll, itl!(tt))
    bx.l_bb = (:l_bb in kkeys) ? -getindex(kwargs, :l_bb) : mod.addLine(bx.p_bll, bx.p_blr, itl!(tt))
    bx.l_br = (:l_br in kkeys) ? -getindex(kwargs, :l_br) : mod.addLine(bx.p_blr, bx.p_bur, itl!(tt))
    bx.l_bt = (:l_bt in kkeys) ? -getindex(kwargs, :l_bt) : mod.addLine(bx.p_bur, bx.p_bul, itl!(tt))
    bx.l_bl = (:l_bl in kkeys) ? -getindex(kwargs, :l_bl) : mod.addLine(bx.p_bul, bx.p_bll, itl!(tt))
    bx.l_f2bll = (:l_f2bll in kkeys) ? getindex(kwargs, :l_f2bll) : mod.addLine(bx.p_fll, bx.p_bll, itl!(tt))
    bx.l_f2blr = (:l_f2blr in kkeys) ? getindex(kwargs, :l_f2blr) : mod.addLine(bx.p_flr, bx.p_blr, itl!(tt))
    bx.l_f2bul = (:l_f2bul in kkeys) ? getindex(kwargs, :l_f2bul) : mod.addLine(bx.p_ful, bx.p_bul, itl!(tt))
    bx.l_f2bur = (:l_f2bur in kkeys) ? getindex(kwargs, :l_f2bur) : mod.addLine(bx.p_fur, bx.p_bur, itl!(tt))

    # Curve loops (6 faces)
    bx.cl_f = (:cl_f in kkeys) ? getindex(kwargs, :cl_f) : mod.addCurveLoop([bx.l_fb, bx.l_fr, bx.l_ft, bx.l_fl], itcl!(tt))
    bx.cl_r = (:cl_r in kkeys) ? getindex(kwargs, :cl_r) : mod.addCurveLoop([bx.l_f2blr, bx.l_br, -bx.l_f2bur, -bx.l_fr], itcl!(tt))
    bx.cl_b = (:cl_b in kkeys) ? getindex(kwargs, :cl_b) : mod.addCurveLoop([-bx.l_bb, -bx.l_bl, -bx.l_bt, -bx.l_br], itcl!(tt))
    bx.cl_l = (:cl_l in kkeys) ? getindex(kwargs, :cl_l) : mod.addCurveLoop([-bx.l_f2bll, -bx.l_fl, bx.l_f2bul, bx.l_bl], itcl!(tt))
    bx.cl_t = (:cl_t in kkeys) ? getindex(kwargs, :cl_t) : mod.addCurveLoop([-bx.l_ft, bx.l_f2bur, bx.l_bt, -bx.l_f2bul], itcl!(tt))
    bx.cl_u = (:cl_u in kkeys) ? getindex(kwargs, :cl_u) : mod.addCurveLoop([-bx.l_fb, -bx.l_f2blr, bx.l_bb, bx.l_f2bll], itcl!(tt))

    # Surfaces (legacy order: f, r, t, b, l, u)
    bx.ps_f = (:ps_f in kkeys) ? getindex(kwargs, :ps_f) : mod.addPlaneSurface([bx.cl_f], itps!(tt))
    bx.ps_r = (:ps_r in kkeys) ? getindex(kwargs, :ps_r) : mod.addPlaneSurface([bx.cl_r], itps!(tt))
    bx.ps_t = (:ps_t in kkeys) ? getindex(kwargs, :ps_t) : mod.addPlaneSurface([bx.cl_t], itps!(tt))
    bx.ps_b = (:ps_b in kkeys) ? getindex(kwargs, :ps_b) : mod.addPlaneSurface([bx.cl_b], itps!(tt))
    bx.ps_l = (:ps_l in kkeys) ? getindex(kwargs, :ps_l) : mod.addPlaneSurface([bx.cl_l], itps!(tt))
    bx.ps_u = (:ps_u in kkeys) ? getindex(kwargs, :ps_u) : mod.addPlaneSurface([bx.cl_u], itps!(tt))

    # Volume
    bx.sl = mod.addSurfaceLoop([bx.ps_f, bx.ps_r, bx.ps_t, bx.ps_b, bx.ps_l, bx.ps_u], itsl!(tt))
    bx.vol = mod.addVolume([bx.sl], itv!(tt))

    return bx
end

"""Create a box using the legacy implementation (Emitter3DTopOpt)."""
function box_legacy(tt::TagTracker, args...; kwargs...)
    bx = Box()
    kkeys = keys(kwargs)
    if :mod in kkeys
        mod = getindex(kwargs, :mod)
    else
        mod = gmsh.model.geo
    end
    # args = swap_yz.(args)

    # Add 8 corners of the box as points
    fll, flr, ful, fur, bll, blr, bul, bur = args[1:8]
    bx.p_fll = typeof(fll) <: Tuple ? mod.addPoint(fll..., itp!(tt)) : fll
    bx.p_flr = typeof(flr) <: Tuple ? mod.addPoint(flr..., itp!(tt)) : flr
    bx.p_fur = typeof(fur) <: Tuple ? mod.addPoint(fur..., itp!(tt)) : fur
    bx.p_ful = typeof(ful) <: Tuple ? mod.addPoint(ful..., itp!(tt)) : ful
    bx.p_bll = typeof(bll) <: Tuple ? mod.addPoint(bll..., itp!(tt)) : bll
    bx.p_blr = typeof(blr) <: Tuple ? mod.addPoint(blr..., itp!(tt)) : blr
    bx.p_bur = typeof(bur) <: Tuple ? mod.addPoint(bur..., itp!(tt)) : bur
    bx.p_bul = typeof(bul) <: Tuple ? mod.addPoint(bul..., itp!(tt)) : bul

    # Add 12 lines
    bx.l_fb = (:l_fb in kkeys) ? -getindex(kwargs, :l_fb) : mod.addLine(bx.p_fll, bx.p_flr, itl!(tt))
    bx.l_fr = (:l_fr in kkeys) ? -getindex(kwargs, :l_fr) : mod.addLine(bx.p_flr, bx.p_fur, itl!(tt))
    bx.l_ft = (:l_ft in kkeys) ? -getindex(kwargs, :l_ft) : mod.addLine(bx.p_fur, bx.p_ful, itl!(tt))
    bx.l_fl = (:l_fl in kkeys) ? -getindex(kwargs, :l_fl) : mod.addLine(bx.p_ful, bx.p_fll, itl!(tt))
    bx.l_bb = (:l_bb in kkeys) ? -getindex(kwargs, :l_bb) : mod.addLine(bx.p_bll, bx.p_blr, itl!(tt))
    bx.l_br = (:l_br in kkeys) ? -getindex(kwargs, :l_br) : mod.addLine(bx.p_blr, bx.p_bur, itl!(tt))
    bx.l_bt = (:l_bt in kkeys) ? -getindex(kwargs, :l_bt) : mod.addLine(bx.p_bur, bx.p_bul, itl!(tt))
    bx.l_bl = (:l_bl in kkeys) ? -getindex(kwargs, :l_bl) : mod.addLine(bx.p_bul, bx.p_bll, itl!(tt))
    bx.l_f2bll = (:l_f2bll in kkeys) ? getindex(kwargs, :l_f2bll) : mod.addLine(bx.p_fll, bx.p_bll, itl!(tt))
    bx.l_f2blr = (:l_f2blr in kkeys) ? getindex(kwargs, :l_f2blr) : mod.addLine(bx.p_flr, bx.p_blr, itl!(tt))
    bx.l_f2bul = (:l_f2bul in kkeys) ? getindex(kwargs, :l_f2bul) : mod.addLine(bx.p_ful, bx.p_bul, itl!(tt))
    bx.l_f2bur = (:l_f2bur in kkeys) ? getindex(kwargs, :l_f2bur) : mod.addLine(bx.p_fur, bx.p_bur, itl!(tt))

    # Add curve loops
    bx.cl_f = (:cl_f in kkeys) ? getindex(kwargs, :cl_f) : mod.addCurveLoop([bx.l_fb, bx.l_fr, bx.l_ft, bx.l_fl], itcl!(tt))
    bx.cl_r = (:cl_r in kkeys) ? getindex(kwargs, :cl_r) : mod.addCurveLoop([bx.l_f2blr, bx.l_br, -bx.l_f2bur, -bx.l_fr], itcl!(tt))
    bx.cl_b = (:cl_b in kkeys) ? getindex(kwargs, :cl_b) : mod.addCurveLoop([-bx.l_bb, -bx.l_bl, -bx.l_bt, -bx.l_br], itcl!(tt))
    bx.cl_l = (:cl_l in kkeys) ? getindex(kwargs, :cl_l) : mod.addCurveLoop([-bx.l_f2bll, -bx.l_fl, bx.l_f2bul, bx.l_bl], itcl!(tt))
    bx.cl_t = (:cl_t in kkeys) ? getindex(kwargs, :cl_t) : mod.addCurveLoop([-bx.l_ft, bx.l_f2bur, bx.l_bt, -bx.l_f2bul], itcl!(tt))
    bx.cl_u = (:cl_u in kkeys) ? getindex(kwargs, :cl_u) : mod.addCurveLoop([-bx.l_fb, -bx.l_f2blr, bx.l_bb, bx.l_f2bll], itcl!(tt))

    # Add surfaces
    bx.ps_f = (:ps_f in kkeys) ? getindex(kwargs, :ps_f) : mod.addPlaneSurface([bx.cl_f], itps!(tt))
    bx.ps_r = (:ps_r in kkeys) ? getindex(kwargs, :ps_r) : mod.addPlaneSurface([bx.cl_r], itps!(tt))
    bx.ps_t = (:ps_t in kkeys) ? getindex(kwargs, :ps_t) : mod.addPlaneSurface([bx.cl_t], itps!(tt))
    bx.ps_b = (:ps_b in kkeys) ? getindex(kwargs, :ps_b) : mod.addPlaneSurface([bx.cl_b], itps!(tt))
    bx.ps_l = (:ps_l in kkeys) ? getindex(kwargs, :ps_l) : mod.addPlaneSurface([bx.cl_l], itps!(tt))
    bx.ps_u = (:ps_u in kkeys) ? getindex(kwargs, :ps_u) : mod.addPlaneSurface([bx.cl_u], itps!(tt))

    # Add surface loop and volume
    bx.sl = mod.addSurfaceLoop([bx.ps_f, bx.ps_r, bx.ps_t, bx.ps_b, bx.ps_l, bx.ps_u], itsl!(tt))
    bx.vol = mod.addVolume([bx.sl], itv!(tt))

    bx
end

"""Stack a box above an existing one (shares bottom face)."""
function boxabove(bx_below::Box, tt::TagTracker, fll, flr, ful, fur, bll, blr, bul, bur; mod=gmsh.model.geo)
    tos = ["p_fll", "p_flr", "p_bll", "p_blr", "l_fb", "l_f2bll", "l_bb", "l_f2blr", "cl_u", "ps_u"]
    fros = ["p_ful", "p_fur", "p_bul", "p_bur", "l_ft", "l_f2bul", "l_bt", "l_f2bur", "cl_t", "ps_t"]
    kwargs = Dict{Symbol,Any}(Symbol(to) => getproperty(bx_below, Symbol(fro)) for (to, fro) in zip(tos, fros))
    return box(tt, fll, flr, ful, fur, bll, blr, bul, bur; mod=mod, kwargs...)
end

"""Stack a box above an existing one using legacy box order."""
function boxabove_legacy(bx::Box, tt::TagTracker, args...; mod=gmsh.model.geo)
    tos = ["p_fll", "p_flr", "p_bll", "p_blr", "l_fb", "l_f2bll", "l_bb", "l_f2blr", "cl_u", "ps_u"]
    fros = ["p_ful", "p_fur", "p_bul", "p_bur", "l_ft", "l_f2bul", "l_bt", "l_f2bur", "cl_t", "ps_t"]
    kwargs = Dict{Symbol,Any}(Symbol(to) => getproperty(bx, Symbol(fro)) for (to, fro) in zip(tos, fros))
    kwargs[:mod] = mod
    box_legacy(tt, args...; kwargs...)
end

"""Add physical group."""
function group(tt::TagTracker, dim::Int, name::String, tags)
    gmsh.model.addPhysicalGroup(dim, tags, itpg!(tt))
    gmsh.model.setPhysicalName(dim, tt.tpg, name)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mesh generators
# ═══════════════════════════════════════════════════════════════════════════════

"""
    genmesh(geo, meshfile; per_x=false, per_y=false, pml=false,
            structured_xy=nothing, center_xy=(0.0, 0.0)) -> (des_low, des_high)

Generate periodic mesh using the legacy box point order (Emitter3DTopOpt).

With `pml=true`, a layer of thickness `geo.dpml` is added below the substrate
(volume `PMLBottom`) and above the air (volume `PMLTop`). The physical layers
keep the coordinates they have without it; `TopZ` / `BottomZ` move to the
outer faces of the layers. Use with `Environment(pml=ZPML(...))`.

`center_xy` moves the cell centre in x and y (`build_angular_simulation` puts a
doubled periodic axis beside the independent patch). `structured_xy=(nx, ny)`
meshes every box as structured hexahedra with `nx`/`ny` intervals across x/y,
and also tags the CAD curves and points on the mirror walls as
`FacesX`/`FacesY`. The defaults reproduce the legacy meshes exactly.
"""
function genmesh(geo::SymmetricGeometry, meshfile::String; per_x::Bool=false, per_y::Bool=false,
                 pml::Bool=false, structured_xy::Union{Nothing,Tuple{Int,Int}}=nothing,
                 center_xy::Tuple{Float64,Float64}=(0.0,0.0))
    pml && geo.dpml <= 0 && error("genmesh: pml=true needs geo.dpml > 0")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("General.Verbosity", geo.verbose)
    gmsh.clear()
    gmsh.model.add("geometry")

    tt = TagTracker()
    z0 = -(geo.hair + geo.hd + geo.hsub) / 2
    L, R = -geo.L / 2 + center_xy[1], geo.L / 2 + center_xy[1]
    front, back = geo.W / 2 + center_xy[2], -geo.W / 2 + center_xy[2]

    # Bottom PML (optional), then substrate on top of it
    ful = (L, front, z0 + geo.hsub, geo.l2)
    fur = (R, front, z0 + geo.hsub, geo.l2)
    bul = (L, back, z0 + geo.hsub, geo.l2)
    bur = (R, back, z0 + geo.hsub, geo.l2)
    if pml
        zp = z0 - geo.dpml
        pml_bot_bx = box_legacy(tt,
            (L, front, zp, geo.l3), (R, front, zp, geo.l3),
            (L, front, z0, geo.l3), (R, front, z0, geo.l3),
            (L, back, zp, geo.l3), (R, back, zp, geo.l3),
            (L, back, z0, geo.l3), (R, back, z0, geo.l3))
        sub_bx = boxabove_legacy(pml_bot_bx, tt,
            pml_bot_bx.p_ful, pml_bot_bx.p_fur, ful, fur,
            pml_bot_bx.p_bul, pml_bot_bx.p_bur, bul, bur)
    else
        fll = (L, front, z0, geo.l3)
        flr = (R, front, z0, geo.l3)
        bll = (L, back, z0, geo.l3)
        blr = (R, back, z0, geo.l3)
        sub_bx = box_legacy(tt, fll, flr, ful, fur, bll, blr, bul, bur)
    end
    z0 += geo.hsub

    des_low = z0

    # Design region
    ful = (L, front, z0 + geo.hd, geo.l2)
    fur = (R, front, z0 + geo.hd, geo.l2)
    bul = (L, back, z0 + geo.hd, geo.l2)
    bur = (R, back, z0 + geo.hd, geo.l2)
    des_bx = boxabove_legacy(sub_bx, tt,
        sub_bx.p_ful, sub_bx.p_fur, ful, fur,
        sub_bx.p_bul, sub_bx.p_bur, bul, bur)
    z0 += geo.hd
    des_high = z0

    # Target/Raman region
    ful = (L, front, z0 + geo.ht, geo.l1)
    fur = (R, front, z0 + geo.ht, geo.l1)
    bul = (L, back, z0 + geo.ht, geo.l1)
    bur = (R, back, z0 + geo.ht, geo.l1)
    tar_bx = boxabove_legacy(des_bx, tt,
        des_bx.p_ful, des_bx.p_fur, ful, fur,
        des_bx.p_bul, des_bx.p_bur, bul, bur)

    # Source region
    ful = (L, front, z0 + geo.hs, geo.l1)
    fur = (R, front, z0 + geo.hs, geo.l1)
    bul = (L, back, z0 + geo.hs, geo.l1)
    bur = (R, back, z0 + geo.hs, geo.l1)
    src_bx = boxabove_legacy(tar_bx, tt,
        tar_bx.p_ful, tar_bx.p_fur, ful, fur,
        tar_bx.p_bul, tar_bx.p_bur, bul, bur)

    # Air region
    ful = (L, front, z0 + geo.hair, geo.l3)
    fur = (R, front, z0 + geo.hair, geo.l1)
    bul = (L, back, z0 + geo.hair, geo.l3)
    bur = (R, back, z0 + geo.hair, geo.l3)
    air_bx = boxabove_legacy(src_bx, tt,
        src_bx.p_ful, src_bx.p_fur, ful, fur,
        src_bx.p_bul, src_bx.p_bur, bul, bur)
    z0 += geo.hair

    # Top PML (optional)
    bxs = [sub_bx, des_bx, tar_bx, src_bx, air_bx]
    bot_bx, top_bx = sub_bx, air_bx
    if pml
        zp = z0 + geo.dpml
        pml_top_bx = boxabove_legacy(air_bx, tt,
            air_bx.p_ful, air_bx.p_fur, (L, front, zp, geo.l3), (R, front, zp, geo.l3),
            air_bx.p_bul, air_bx.p_bur, (L, back, zp, geo.l3), (R, back, zp, geo.l3))
        bxs = [pml_bot_bx; bxs; pml_top_bx]
        bot_bx, top_bx = pml_bot_bx, pml_top_bx
    end

    # Periodic BCs
    gmsh.model.geo.synchronize()
    for bx in bxs
        if per_x
            gmsh.model.mesh.setPeriodic(2, [bx.ps_l], [bx.ps_r],
                [1, 0, 0, -geo.L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        end
        if per_y
            gmsh.model.mesh.setPeriodic(2, [bx.ps_b], [bx.ps_f],
                [1, 0, 0, 0, 0, 1, 0, -geo.W, 0, 0, 1, 0, 0, 0, 0, 1])
        end
    end

    # Physical groups (match legacy ordering)
    group(tt, 0, "NodesZ", [
        top_bx.p_ful, top_bx.p_fur,
        top_bx.p_bul, top_bx.p_bur,
    ])
    group(tt, 1, "EdgesZ", [
        bot_bx.l_f2bll, bot_bx.l_f2blr,
        bot_bx.l_fb, bot_bx.l_bb,
        top_bx.l_ft, top_bx.l_f2bul,
        top_bx.l_f2bur, top_bx.l_bt,
    ])
    group(tt, 2, "TopZ", [top_bx.ps_t])
    group(tt, 2, "BottomZ", [bot_bx.ps_u])
    group(tt, 2, "FacesX", reduce(vcat, [[bx.ps_l, bx.ps_r] for bx in bxs]))
    group(tt, 2, "FacesY", reduce(vcat, [[bx.ps_f, bx.ps_b] for bx in bxs]))

    group(tt, 0, "DesignNodes", [
        des_bx.p_fll, des_bx.p_flr, des_bx.p_bll, des_bx.p_blr,
        des_bx.p_ful, des_bx.p_fur, des_bx.p_bul, des_bx.p_bur,
    ])
    group(tt, 1, "DesignEdges", [
        des_bx.l_fb, des_bx.l_f2bll, des_bx.l_f2blr, des_bx.l_bb,
        des_bx.l_ft, des_bx.l_f2bul, des_bx.l_f2bur, des_bx.l_bt,
    ])
    group(tt, 2, "DesignFaces", [des_bx.ps_u, des_bx.ps_t])
    group(tt, 3, "Design", [des_bx.vol])
    group(tt, 3, "Substrate", [sub_bx.vol])
    group(tt, 3, "Air", [air_bx.vol, src_bx.vol])
    group(tt, 2, "Target", [tar_bx.ps_t])
    group(tt, 2, "Source", [src_bx.ps_t])
    group(tt, 3, "Raman", [tar_bx.vol])
    if pml
        group(tt, 3, "PMLBottom", [pml_bot_bx.vol])
        group(tt, 3, "PMLTop", [pml_top_bx.vol])
    end

    gmsh.model.geo.synchronize()
    if structured_xy !== nothing
        nx,ny=structured_xy
        min(nx,ny)>1 || throw(ArgumentError("genmesh: structured axes require at least two intervals"))
        spacing=min(geo.l1,geo.l2,geo.l3)
        for (_,tag) in gmsh.model.getEntities(1)
            ends=gmsh.model.getBoundary([(1,tag)],false,false,false)
            a=gmsh.model.getValue(0,ends[1][2],Float64[])
            b=gmsh.model.getValue(0,ends[2][2],Float64[])
            delta=abs.(a-b)
            n=delta[1]>1e-8 ? nx : delta[2]>1e-8 ? ny : max(1,ceil(Int,delta[3]/spacing))
            gmsh.model.mesh.setTransfiniteCurve(tag,n+1)
        end
        for (_,tag) in gmsh.model.getEntities(2)
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(2,tag)
        end
        for (_,tag) in gmsh.model.getEntities(3)
            gmsh.model.mesh.setTransfiniteVolume(tag)
        end
        # Essential tangential conditions also constrain CAD curves/vertices
        # on the mirror. Gmsh assigns those their own entities, outside the
        # surface-only physical groups used by the legacy generator.
        for dim in (0,1), (name,axis,width) in (("FacesX",1,geo.L),("FacesY",2,geo.W))
            tags=Int[]
            for (_,tag) in gmsh.model.getEntities(dim)
                box=gmsh.model.getBoundingBox(dim,tag)
                if abs(box[axis]-box[axis+3])<1e-6 &&
                   abs(abs((box[axis]+box[axis+3])/2-center_xy[axis])-width/2)<1e-6
                    push!(tags,tag)
                end
            end
            isempty(tags) || group(tt,dim,name,tags)
        end
    end
    gmsh.model.mesh.generate(3)
    gmsh.write(meshfile)
    gmsh.finalize()

    @info "Mesh generated ($(structured_xy === nothing ? "legacy order" : "structured")): $meshfile"

    return (des_low, des_high)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: plasmon_period is defined in Physics/Materials.jl

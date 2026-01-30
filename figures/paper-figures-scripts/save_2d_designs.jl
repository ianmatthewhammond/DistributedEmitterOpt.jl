#!/usr/bin/env julia
# Save 2D design images, binary masks, and text data for all design VTUs.
# Run from repo root: julia --project=. scripts/figures/save_2d_designs.jl
#
# Output structure under figures/2d-designs/:
#   images/  — annotated grayscale PNGs with axes + scale bar
#   binary/  — raw black/white binary mask PNGs (no labels)
#   data/    — tab-separated text files (x_nm, y_nm, p)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PyCall
using CairoMakie
import Images
using Images: Gray, channelview
using FileIO
using DelimitedFiles

pv = pyimport("pyvista")
np = pyimport("numpy")

pv.global_theme.transparent_background = false
pv.global_theme.background = "white"
pv.global_theme.allow_empty_mesh = true

# --- Config ---
DATA_DIR = joinpath(@__DIR__, "..", "..", "figures", "paper-figures-scripts", "data")
BASE_DIR = joinpath(@__DIR__, "..", "..", "figures", "paper-figures-scripts", "2d-designs")
IMAGES_DIR = joinpath(BASE_DIR, "images")
BINARY_DIR = joinpath(BASE_DIR, "binary")
DATA_OUT = joinpath(BASE_DIR, "data")
mkpath(IMAGES_DIR)
mkpath(BINARY_DIR)
mkpath(DATA_OUT)

DESIGNS = [
    ("Constrained_Metal_nominal", "Constrained_Metal/nominal/y_design.vtu"),
    ("Constrained_Metal_Bipolarized", "Constrained_Metal/polarization_Bi/y_design.vtu"),
    ("Constrained_Metal_Anisotropic", "Constrained_Metal/isotropy_Anisotropic/y_design.vtu"),
    ("Constrained_Metal_Shifted", "Constrained_Metal/frequency_Shifted/y_design.vtu"),
    ("Constrained_Dielectric_nominal", "Constrained_Dielectric/nominal/y_design.vtu"),
    ("Constrained_Dielectric_Anisotropic", "Constrained_Dielectric/isotropy_Anisotropic/y_design.vtu"),
    ("Constrained_Dielectric_Shifted", "Constrained_Dielectric/frequency_Shifted/y_design.vtu"),
    ("Freeform_Metal_nominal", "Freeform_Metal/nominal/y_design.vtu"),
    ("Spheres_Bipolarized", "Spheres/polarization_Bi/y_design.vtu"),
    ("Starting_Metal", "data_starting/metal/y_design.vtu"),
    ("Starting_Dielectric", "data_starting/dielectric/y_design.vtu"),
    ("Nonlinear_Eth5", "Nonlinear/geometries/design5-0.vtu"),
    ("Nonlinear_Eth10", "Nonlinear/geometries/design10-0.vtu"),
    ("Nonlinear_Eth17_5", "Nonlinear/geometries/design17-5.vtu"),
    ("Nonlinear_Eth25", "Nonlinear/geometries/design25-0.vtu"),
    ("Nonlinear_EthInf", "Nonlinear/geometries/designinf.vtu"),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

"""
Reflect quarter-cell mesh in x and y → full period.
Returns (parts, combined_bounds_xmin, xmax, ymin, ymax, zmin, zmax).
"""
function reflect_full_period(mesh)
    xl, xr, yl, yr, zl, zr = mesh.bounds
    W = xr - xl
    L = yr - yl

    signx = 1   # flipx=false
    signy = -1  # flipy=true
    reflectx = mesh.reflect((1, 0, 0), point=(-W / 2 * signx, 0, 0))
    reflecty = mesh.reflect((0, 1, 0), point=(0, -L / 2 * signy, 0))
    reflectxy = reflecty.reflect((1, 0, 0), point=(-W / 2 * signx, 0, 0))

    parts = [mesh, reflectx, reflecty, reflectxy]
    all_b = [p.bounds for p in parts]
    cxmin = minimum(b[1] for b in all_b)
    cxmax = maximum(b[2] for b in all_b)
    cymin = minimum(b[3] for b in all_b)
    cymax = maximum(b[4] for b in all_b)

    return parts, cxmin, cxmax, cymin, cymax, zl, zr
end

"""
Render a top-down screenshot of the reflected full-period.
Returns (image, phys_w, phys_h).
"""
function render_design_topdown(mesh; window_size=(800, 800), color="#888888")
    parts, cxmin, cxmax, cymin, cymax, zl, zr = reflect_full_period(mesh)
    phys_w = cxmax - cxmin
    phys_h = cymax - cymin

    plotter = pv.Plotter(off_screen=true, window_size=window_size)
    for part in parts
        plotter.add_mesh(part, color=color, show_scalar_bar=false,
            lighting=true, show_edges=false)
    end

    cx = (cxmin + cxmax) / 2
    cy = (cymin + cymax) / 2
    cz = (zl + zr) / 2
    plotter.camera_position = [(cx, cy, zr + 500), (cx, cy, cz), (0, 1, 0)]
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    tmp = tempname() * ".png"
    plotter.screenshot(tmp)
    plotter.close()
    img = load(tmp)
    rm(tmp, force=true)
    return img, phys_w, phys_h
end

"""
Render a pure binary (black = material, white = void) top-down screenshot.
Uses only the raw mesh (no reflections), tightly framed with no margin.
Returns (image, phys_w, phys_h).
"""
function render_design_binary(mesh; base_size=800)
    xl, xr, yl, yr, zl, zr = mesh.bounds
    phys_w = xr - xl
    phys_h = yr - yl

    # Match window aspect ratio to mesh so there's zero margin
    if phys_w >= phys_h
        ws = (base_size, round(Int, base_size * phys_h / phys_w))
    else
        ws = (round(Int, base_size * phys_w / phys_h), base_size)
    end

    pv.global_theme.background = "white"
    plotter = pv.Plotter(off_screen=true, window_size=ws)
    plotter.add_mesh(mesh, color="black", show_scalar_bar=false,
        lighting=false, show_edges=false)

    cx = (xl + xr) / 2
    cy = (yl + yr) / 2
    cz = (zl + zr) / 2
    plotter.camera_position = [(cx, cy, zr + 500), (cx, cy, cz), (0, 1, 0)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = phys_h / 2

    tmp = tempname() * ".png"
    plotter.screenshot(tmp)
    plotter.close()
    img = load(tmp)
    rm(tmp, force=true)
    return img, phys_w, phys_h
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function crop_margins(img; threshold=0.98, pad=5)
    chans = channelview(img)
    if ndims(chans) == 3
        white = dropdims(all(chans[1:min(3, size(chans, 1)), :, :] .>= threshold, dims=1), dims=1)
    else
        white = chans .>= threshold
    end
    nonwhite = .!white
    rows = vec(any(nonwhite, dims=2))
    cols = vec(any(nonwhite, dims=1))
    if !any(rows) || !any(cols)
        return img
    end
    rmin = max(1, findfirst(rows) - pad)
    rmax = min(size(img, 1), findlast(rows) + pad)
    cmin = max(1, findfirst(cols) - pad)
    cmax = min(size(img, 2), findlast(cols) + pad)
    return img[rmin:rmax, cmin:cmax]
end

"""
Save annotated figure (axes + scale bar) to images/ subfolder.
"""
function save_annotated(img, label, phys_w, phys_h, outpath)
    gray = Gray.(img)
    gmat = Float64.(gray)
    gmat = reverse(gmat, dims=1)

    fig = Figure(size=(600, 600), fontsize=16, font="Times")
    ax = Axis(fig[1, 1],
        xlabel="x (nm)", ylabel="y (nm)",
        title=label, titlesize=14,
        aspect=DataAspect(),
    )
    image!(ax, (0, phys_w), (0, phys_h), gmat, colormap=:grays, colorrange=(0, 1))

    bar_candidates = [25, 50, 100, 200, 250, 500]
    idx = findfirst(c -> c >= phys_w / 5, bar_candidates)
    bar_len = isnothing(idx) ? round(Int, phys_w / 4 / 50) * 50 : bar_candidates[idx]

    bx = phys_w * 0.05
    by = phys_h * 0.05
    lines!(ax, [bx, bx + bar_len], [by, by], color=:black, linewidth=3)
    text!(ax, bx + bar_len / 2, by + phys_h * 0.02,
        text="$(bar_len) nm", align=(:center, :bottom), fontsize=14, color=:black)

    save(outpath, fig, px_per_unit=2)
end

"""
Save binary mask to binary/ subfolder — just the raw image, no axes.
"""
function save_binary(img, outpath)
    gray = Gray.(img)
    # Threshold to pure black/white
    bw = map(px -> Float64(px) < 0.5 ? Gray(0.0) : Gray(1.0), gray)
    FileIO.save(outpath, bw)
end

"""
Export the design field p as a text file.
Rasterises the raw mesh (no reflections) onto a regular grid and writes
x_nm  y_nm  p  (tab-separated, one point per line).
"""
function save_data_txt(mesh, outpath)
    xl, xr, yl, yr, zl, zr = mesh.bounds
    phys_w = xr - xl
    phys_h = yr - yl
    zmid = (zl + zr) / 2
    nx, ny = 200, 200
    xs = range(xl, xr, length=nx)
    ys = range(yl, yr, length=ny)

    grid_points = zeros(nx * ny, 3)
    idx = 1
    for iy in 1:ny, ix in 1:nx
        grid_points[idx, 1] = xs[ix]
        grid_points[idx, 2] = ys[iy]
        grid_points[idx, 3] = zmid
        idx += 1
    end

    grid = pv.PolyData(grid_points)
    sampled = grid.sample(mesh, tolerance=phys_w / nx * 2)

    # Extract p values — points outside mesh get 0
    p_arr = try
        Float64.(np.array(sampled.point_data["p"]))
    catch
        zeros(nx * ny)
    end

    open(outpath, "w") do io
        println(io, "# x_nm\ty_nm\tp")
        println(io, "# Raw mesh domain: $(round(phys_w, digits=2)) x $(round(phys_h, digits=2)) nm")
        println(io, "# Grid: $(nx) x $(ny)")
        idx = 1
        for iy in 1:ny, ix in 1:nx
            println(io, "$(round(xs[ix], digits=4))\t$(round(ys[iy], digits=4))\t$(round(p_arr[idx], digits=6))")
            idx += 1
        end
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
println("Output: $BASE_DIR")
println("  images/ — annotated grayscale with axes + scale bar")
println("  binary/ — raw black/white binary mask")
println("  data/   — tab-separated text (x_nm, y_nm, p)")
println("="^60)

for (label, relpath) in DESIGNS
    filepath = joinpath(DATA_DIR, relpath)
    if !isfile(filepath)
        println("  SKIP (not found): $relpath")
        continue
    end

    println("Processing: $label")
    mesh = pv.read(filepath)

    # 1. Annotated image
    img, phys_w, phys_h = render_design_topdown(mesh)
    img_cropped = crop_margins(img)
    save_annotated(img_cropped, label, phys_w, phys_h, joinpath(IMAGES_DIR, "$(label).png"))
    println("  images/$(label).png")

    # 2. Binary mask (raw mesh, no reflections, no margin)
    bimg, _, _ = render_design_binary(mesh)
    save_binary(bimg, joinpath(BINARY_DIR, "$(label).png"))
    println("  binary/$(label).png")

    # 3. Text data (raw mesh, no reflections)
    save_data_txt(mesh, joinpath(DATA_OUT, "$(label).txt"))
    println("  data/$(label).txt")
end

println("="^60)
println("Done.")

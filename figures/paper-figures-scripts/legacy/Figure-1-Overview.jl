# --- START OF FILE Figure-1.jl.txt ---

using Pkg; Pkg.activate("/Users/ianhammond/GitHub/Emitter3DTopOpt")
using Revise
import Emitter3DTopOpt as e3
plot_field, plot_substrate, combine_figures, add_text! = e3.plot_field, e3.plot_substrate, e3.combine_figures, e3.add_text!
plot_directionals = e3.plot_directionals
using PyCall, PyPlot # PyPlot might not be strictly needed
# GLMakie is activated but CairoMakie will be used for saving the final PDF
using GLMakie, CairoMakie, LaTeXStrings 
using Gridap, GridapGmsh, GridapMakie
using CSV, DataFrames
using Images, JLD2, Random
using ColorSchemes, Colors
using DelaunayTriangulation
using Printf
using LaTeXStrings
# GLMakie.activate!() # Activate GLMakie for potential interactive use, CairoMakie for saving

vtk = pyimport("vtk")
np = pyimport("numpy")
pv = pyimport("pyvista")

text_size = 30
title_size = 40
fig_width = 2800
pv.global_theme.font.family = "times"
root = "./"

pv.global_theme.transparent_background = false # For PNG base

np.random.seed(0)


############# Figure A - Base Raster Generation #############
path_bid = "data/Freeform_Metal/nominal/"
path_mono = "data/Constrained_Metal/nominal/"
uns_data, con_data = e3.load_bid_parameters(pv, np, path_bid, path_mono)

# Path for the temporary PNG from PyVista
temp_png_path = "temp_plots/overview_base.pdf" 

L, W, des_low, des_high, design, y_fields, x_fields, g_ar, λ_s, gys = (
    con_data[:L], con_data[:W], con_data[:des_low], con_data[:des_high], con_data[:design], con_data[:y_fields], 
    con_data[:x_fields], con_data[:g_ar], con_data[:λ_s], con_data[:gys]
)
design.point_data.set_scalars(ones(length(design.point_data.get_array("p"))) + 0.0001*rand(length(design.point_data.get_array("p"))), "p")

# Functions
function plot_material(np, pv, plotter, design, W, L; colorbar=true, title="\$\\rho(x)\$", font_size = 20, title_font_size = 24, num_periods_x=1, num_periods_y=1, ontop=false, design_field="p", contour=true, full=false, clim=(0.0, 1.0), opacity=nothing, reflectybool=true, flipx=false, flipy=false)
    pv.global_theme.allow_empty_mesh = true

    contours = design.contour(np.linspace(0, 1, 2), scalars=design_field)
    signx = (flipx) ? -1 : 1
    signy = (flipy) ? -1 : 1
    reflectx = pv.DataSetFilters.reflect(design, (1, 0, 0), point=(-W/2*signx, 0, 0))
    reflecty = pv.DataSetFilters.reflect(design, (0, 1, 0), point=(0, -L/2*signy, 0))
    reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W/2*signx, 0, 0))
    contoursx = reflectx.contour(np.linspace(0, 1, 2), scalars=design_field)
    contoursy = reflecty.contour(np.linspace(0, 1, 2), scalars=design_field)
    contoursxy = reflectxy.contour(np.linspace(0, 1, 2), scalars=design_field)
    reflectx = pv.DataSetFilters.reflect(design, (1, 0, 0), point=(-W/2*signx, 0, 0))
    reflecty = pv.DataSetFilters.reflect(design, (0, 1, 0), point=(0, -L/2*signy, 0))
    reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W/2*signx, 0, 0))

    contour_color = pv.Color("#b097d1")
    scalargs = if ontop
        ["title"=>title,"vertical"=>false, "label_font_size"=>font_size, "title_font_size"=>title_font_size, "position_x"=>0.25, "position_y"=>0.8, "color"=>"Black", "use_opacity"=>false]
    else
        ["title"=>title,"vertical"=>true, "label_font_size"=>font_size, "title_font_size"=>title_font_size, "position_x"=>0.0, "position_y"=>0.35, "color"=>"Black", "use_opacity"=>false]
    end

    for i in 0:1:num_periods_x-1
        for j in 0:1:num_periods_y-1
            ps = Vector(0:0.01:1.0)
            opacity = (isnothing(opacity)) ? (x -> ifelse(x < 1.0, 0.0, x/20) ).(ps) : opacity
            plotter.add_mesh(design.translate((i*W*2, j*L*2, 0), inplace=false), color=contour_color, clim=clim, opacity=opacity, show_scalar_bar=(colorbar & (i == j) & (j == 0)), scalar_bar_args=Dict(scalargs))
            if !full
                plotter.add_mesh(reflectx.translate((i*W*2, j*L*2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                if reflectybool
                    plotter.add_mesh(reflecty.translate((i*W*2, j*L*2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                    plotter.add_mesh(reflectxy.translate((i*W*2, j*L*2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                end
            end
        end
    end
    return plotter
end

sign1x = 1
sign1y = 1
full=false
flip1x = false
flip1y = false
water = y_fields.clip(origin=(0, 0, des_low), normal="-z").clip(origin=(0, 0, des_high+600), normal="z")
centers = y_fields.cell_centers().clip(origin=(0, 0, des_low), normal="-z").clip(origin=(0, 0, des_high+400), normal="z")
points = centers.points[np.random.randint(low=0, high=centers.n_points - 1, size=round(Int,centers.n_points * 0.000125)), :]
centers = pv.PolyData(points)
reflectxcenters = pv.DataSetFilters.reflect(centers, (1, 0, 0), point=(-W/2*sign1x, 0, 0))
reflectxwater = pv.DataSetFilters.reflect(water, (1, 0, 0), point=(-W/2*sign1x, 0, 0))
reflectycenters = pv.DataSetFilters.reflect(centers, (0, 1, 0), point=(0, -L/2*sign1y, 0))
reflectywater = pv.DataSetFilters.reflect(water, (0, 1, 0), point=(0, -L/2*sign1y, 0))
reflectxycenters = pv.DataSetFilters.reflect(reflectycenters, (1, 0, 0), point=(-W/2*sign1x, 0, 0))
reflectxywater = pv.DataSetFilters.reflect(reflectywater, (1, 0, 0), point=(-W/2*sign1x, 0, 0))

pv_window_width = round(Int,0.5*fig_width)
pv_window_height = 1200
plotter = pv.Plotter(shape=(1, 1), off_screen=true, window_size=(pv_window_width, pv_window_height), border=false)
plotter.show_axes()

for i in 0:1:1
    for j in 0:1:1
        plotter.add_mesh(centers.translate((i*W*2, j*L*2, 0), inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
        plotter.add_mesh(reflectxcenters.translate((i*W*2, j*L*2, 0), inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
        plotter.add_mesh(reflectycenters.translate((i*W*2, j*L*2, 0), inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
        plotter.add_mesh(reflectxycenters.translate((i*W*2, j*L*2, 0), inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
        plotter.add_mesh(water.translate((i*W*2, j*L*2, 0), inplace=false), color="b", opacity=0.001)
        plotter.add_mesh(reflectxwater.translate((i*W*2, j*L*2, 0), inplace=false), color="b", opacity=0.001)
        plotter.add_mesh(reflectywater.translate((i*W*2, j*L*2, 0), inplace=false), color="b", opacity=0.001)
        plotter.add_mesh(reflectxywater.translate((i*W*2, j*L*2, 0), inplace=false), color="b", opacity=0.001)
    end
end

plot_material(np, pv, plotter, design, W, L; colorbar=false, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n", font_size = text_size, title_font_size = title_size, full, flipx=flip1x, flipy=flip1y, num_periods_x=2,num_periods_y=2)
plot_substrate(np, pv, plotter, y_fields, W, L, des_low; full, flipx=flip1x, flipy=flip1y, num_periods_x=2,num_periods_y=2)
plotter.add_mesh(pv.Arrow((0,-200,des_high+800), (0,300,-500), scale=300), color="k")
arrow = [0,-300,-500]
va = arrow ./ sqrt(sum(arrow.^2))
tail = Tuple([0,300+100,des_high+800] .+ va .* 300)
plotter.add_mesh(pv.Arrow(tail, (0,300,des_high+800), scale=300), color="k")

# Comment out PyVista labels that will be replaced by Makie
# plotter.add_text("\$\\omega_p\$", position=(240-50, 950), color="k", font_size=26, shadow=false, font="times")
# plotter.add_text("\$\\omega_e\$", position=(700-75, 950), color="k", font_size=26, shadow=false, font="times")
# plotter.add_text("Substrate", position=(1000-100, 152), color="k", font_size=26, shadow=true, font="times")
plotter.add_text("\$]\$", position=(950-100, 150), color="k", font_size=40, shadow=true, font="times")
# plotter.add_text("Design", position=(1000-100, 235), color="k", font_size=26, shadow=true, font="times")
plotter.add_text("\$]\$", position=(950-100, 230), color="k", font_size=50, shadow=true, font="times")
# plotter.add_text("Fluid", position=(1000-100, 450), color="k", font_size=26, shadow=true, font="times")
plotter.add_text("\$\\rfloor\$", position=(950-100, 330), color="k", font_size=50, shadow=true, font="times")
plotter.add_text("|", position=(950-83, 400), color="k", font_size=40, shadow=false, font="times") # User adjusted x
plotter.add_text("|", position=(950-83, 450), color="k", font_size=40, shadow=false, font="times") # User adjusted x
plotter.add_text("|", position=(950-83, 500), color="k", font_size=40, shadow=false, font="times") # User adjusted x
# plotter.add_text("Raman\nMolecules", position=(850, 700), color="k", font_size=26, shadow=true, font="times")
plotter.add_mesh(pv.Arrow((-300,850,des_high+400), (-50,-300,-150).*2.0, scale=250), color="r")
plotter.add_mesh(pv.Sphere(radius=12.0, center=(-350,550,des_high+400-150)), color="r")

plotter.add_mesh(pv.Line((0,100,des_high+800), (0,100,des_high+720)), color="k", line_width=10)
plotter.add_mesh(pv.Line((0,100,des_high+700), (0,100,des_high+620)), color="k", line_width=10)
plotter.add_mesh(pv.Line((0,100,des_high+600), (0,100,des_high+520)), color="k", line_width=10)
θ_val, r_val = atand(500/300), 200 # Renamed to avoid conflict
center_arc = [0,100,des_high+600] # Renamed
pointa = center_arc .+ [0, -r_val*sind(θ_val*0.75), r_val*cosd(θ_val*0.75)]
pointb = center_arc .+ [0, -r_val*sind(θ_val*0.25), r_val*cosd(θ_val*0.25)]
arc = pv.CircularArc(pointa, pointb, center_arc)
plotter.add_mesh(arc, color="k", line_width=10)
pointa = center_arc .+ [0, r_val*sind(θ_val*0.75), r_val*cosd(θ_val*0.75)]
pointb = center_arc .+ [0, r_val*sind(θ_val*0.25), r_val*cosd(θ_val*0.25)]
arc = pv.CircularArc(pointa, pointb, center_arc)
plotter.add_mesh(arc, color="k", line_width=10)
# Comment out PyVista labels that will be replaced by Makie
# plotter.add_text("\$\\theta_p\$", position=(430-60, 1020-90), color="k", font_size=26, shadow=false, font="times")
# plotter.add_text("\$\\theta_e\$", position=(570-60, 1020-90), color="k", font_size=26, shadow=false, font="times")

plotter.camera.position = (plotter.camera.position .* 0.5 .- (-W/1.0*sign1x, -L/10.0*sign1y, des_high+1000*L/(2*92.1437880268)+200) .* 0.5)
plotter.camera.position = plotter.camera.position .+ [2000, -1800, 100+100-10]
plotter.camera.focal_point = plotter.camera.focal_point  .+ [0, 150, -100+110]

# Save PyVista output as PNG
plotter.save_graphic(temp_png_path) # Changed from save_graphic for reliable PNG
plotter.clear()
plotter.close()
println("PyVista base image saved to: ", temp_png_path)

# ------------------------------------------------------------
# Ensure PyMuPDF (fitz) and matplotlib (to locate the font) exist
# ------------------------------------------------------------
const fitz = try
    pyimport("fitz")
catch
    run(`$(PyCall.python) -m pip install --quiet --user pymupdf`)
    pyimport("fitz")
end
const fm = try
    pyimport("matplotlib.font_manager")
catch
    run(`$(PyCall.python) -m pip install --quiet --user matplotlib`)
    pyimport("matplotlib.font_manager")
end

"""
    add_theta_omega_labels(pdf_in, pdf_out; fig_px_width=2800, fig_px_height=1200)

Overlay ωₚ, ωₑ, θₚ, θₑ, and other labels (as true vector text) onto the first page of `pdf_in`
using a font that contains the subscript glyphs, and save to `pdf_out`.
"""
function add_theta_omega_labels(pdf_in::AbstractString,
                                pdf_out::AbstractString;
                                fig_px_width::Int = 2800,
                                fig_px_height::Int = 1200)

    # 1 ── locate fonts
    times_path = fm.findfont("Times")
    dejavu_path = fm.findfont("DejaVu Serif")

    # 2 ── desired labels in pure Unicode
    greek_labels = [
        (600-130-50,  950,  "ωₚ"),   # omega + subscript p (U+209A)
        (1370-130+50,  950,  "ωₑ"),   # omega + subscript e (U+2091)
        (915-130-50, 1020 - 90, "θₚ"),
        (1130-130+50, 1020 - 90, "θₑ"),
    ]

    text_labels = [
        (1800, 172, "Substrate"),  # Moved to right side
        (1800, 255, "Design"),     # Moved to right side
        (1800, 450, "Fluid"),      # Moved to right side
        (1800, 700, "Raman\nMolecules"),  # Moved to right side
    ]

    # 3 ── open PDF, compute pixel→point scaling
    doc  = fitz.open(pdf_in)
    page = doc[0]
    sx = page.rect.width  / fig_px_width
    sy = page.rect.height / fig_px_height

    # 4 ── write Greek labels with DejaVu Serif
    for (x_px, y_px, txt) in greek_labels
        x_pt = x_px * sx
        y_pt = (fig_px_height - y_px) * sy
        page.insert_text(
            fitz.Point(x_pt, y_pt),
            txt;
            fontsize = text_size,
            fontname = "DejaVuSerif",
            fontfile = dejavu_path,
            fill     = (0, 0, 0)
        )
    end

    # 5 ── write regular text labels with Times
    for (x_px, y_px, txt) in text_labels
        x_pt = x_px * sx
        y_pt = (fig_px_height - y_px) * sy
        page.insert_text(
            fitz.Point(x_pt, y_pt),
            txt;
            fontsize = text_size,
            fontname = "Times",
            fontfile = times_path,
            fill     = (0, 0, 0)
        )
    end

    # 6 ── save
    doc.save(pdf_out, deflate=true)
    doc.close()
    println("✓ wrote annotated PDF → ", pdf_out)
end


function trim_pdf_whitespace(pdf_in::AbstractString,
    pdf_out::AbstractString;
    margin_pts::Real = 2.0)

    # ── open the source PDF ────────────────────────────────────────────────────
    src = fitz.open(pdf_in)
    page = src[0]                       # we only touch page 1

    # ── collect every rectangle that actually contains objects ────────────────
    bbox = nothing                      # will grow with unions ( | )

    # text blocks
    for blk in page.get_text("blocks")
    r = fitz.Rect(blk[1], blk[2], blk[3], blk[4])
    bbox = isnothing(bbox) ? r : (bbox | r)
    end

    # vector drawings (lines, curves, images rendered as XObjects, …)
    for item in page.get_drawings()
    bbox = isnothing(bbox) ? item["rect"] : (bbox | item["rect"])
    end

    # fall back to full page if nothing was detected (very unlikely)
    bbox = isnothing(bbox) ? page.rect : bbox

    # ── expand / clamp by a small margin so nothing is cut off ────────────────
    full = page.rect
    bbox.x0 = max(full.x0, bbox.x0 - margin_pts)
    bbox.y0 = max(full.y0, bbox.y0 - margin_pts)
    bbox.x1 = min(full.x1, bbox.x1 + margin_pts)
    bbox.y1 = min(full.y1, bbox.y1 + margin_pts)

    # ── write the cropped page into a brand-new document ──────────────────────
    out = fitz.open()
    newpage = out.new_page(width = bbox.width, height = bbox.height)
    newpage.show_pdf_page(newpage.rect, src, 0; clip = bbox)

    out.save(pdf_out, deflate = true)
    out.close()
    src.close()

    println("✓ wrote trimmed PDF → ", pdf_out)
end


# example
add_theta_omega_labels("temp_plots/overview_base.pdf", "temp_plots/overview_annotated.pdf")
trim_pdf_whitespace("temp_plots/overview_annotated.pdf", "figures/Figure-1-Overview.pdf")


# --- Figure 1: Overview Script ---
# This script generates the Figure 1 overview plot using pre-computed data.

using DistributedEmitterOpt
using DistributedEmitterOpt.Visualization
using PyCall
using PyPlot
using CairoMakie
using JLD2
using Printf

# --- Setup ---
# GLMakie is used for potential interactive tweaks, CairoMakie for final vector export
CairoMakie.activate!()

# Import Python libraries via PyCall
vtk = pyimport("vtk")
np = pyimport("numpy")
pv = pyimport("pyvista")

# Plotting Configuration
text_size = 30
title_size = 40
fig_width = 2800
pv.global_theme.font.family = "times"
pv.global_theme.transparent_background = false
pv.global_theme.allow_empty_mesh = true

# Ensure output directory exists
mkpath("figures")
temp_png_path = "figures/overview_base.pdf"

println("Generating Figure 1...")

# --- Load Data ---
println("Loading data...")
# Freeform Metal Nominal
y_fields = get_figure_data("Freeform_Metal", "nominal", "fields_y")
y_design = get_figure_data("Freeform_Metal", "nominal", "design_y")
output_txt = get_figure_data("Freeform_Metal", "nominal", "output")
results_jld = get_figure_data("Freeform_Metal", "nominal", "results")

# Only loading what's strictly needed for the visual
# The original script loaded "bid" and "mono" paths but mostly used "con_data" variables
# Let's assume we need the Constrained Metal Nominal data corresponding to "con_data"
con_y_fields = get_figure_data("Constrained_Metal", "nominal", "fields_y")
con_y_design = get_figure_data("Constrained_Metal", "nominal", "design_y")
# results_jld tuple: (g_opt, p_opt, grad, g_ar, g_biggest, p_biggest) [Legacy format]
# We might need to extract scalars like L, W from the mesh or results if not explicit.
# In the new code, L and W are usually attributes of the simulation or mesh.
# PyVista meshes have bounds.

if isnothing(con_y_design) || isnothing(con_y_fields)
    error("Failed to load required data for Figure 1.")
end

# Extract Geometry from Mesh Bounds
bounds = con_y_design.bounds
W = bounds[2] - bounds[1]
L = bounds[4] - bounds[3]
des_low = bounds[5]
des_high = bounds[6]

println("Geometry loaded: W=$W, L=$L, Z=[$des_low, $des_high]")

# Add noise to design field for visualization texture (as per original script)
p_array = con_y_design.point_data.get_array("p")
con_y_design.point_data.set_scalars(ones(length(p_array)) + 0.0001 * rand(length(p_array)), "p")

# --- Plotting Main Figure ---
println("Plotting base visualization...")

# Setup Plotter
pv_window_width = round(Int, 0.5 * fig_width)
pv_window_height = 1200
plotter = pv.Plotter(shape=(1, 1), off_screen=true, window_size=(pv_window_width, pv_window_height), border=false)
plotter.show_axes()

# Helper logic for reflections (adapted from original script)
sign1x = 1
sign1y = 1
full = false
flip1x = false
flip1y = false

# Water/Fluid Region
water = con_y_fields.clip(origin=(0, 0, des_low), normal="-z").clip(origin=(0, 0, des_high + 600), normal="z")
centers = con_y_fields.cell_centers().clip(origin=(0, 0, des_low), normal="-z").clip(origin=(0, 0, des_high + 400), normal="z")
points = centers.points[np.random.randint(low=0, high=centers.n_points - 1, size=round(Int, centers.n_points * 0.000125)), :]
centers_poly = pv.PolyData(points)

# Reflect and Add Meshes
reflect_filters = [
    (mesh) -> pv.DataSetFilters.reflect(mesh, (1, 0, 0), point=(-W / 2 * sign1x, 0, 0)),
    (mesh) -> pv.DataSetFilters.reflect(mesh, (0, 1, 0), point=(0, -L / 2 * sign1y, 0)),
    (mesh) -> pv.DataSetFilters.reflect(pv.DataSetFilters.reflect(mesh, (0, 1, 0), point=(0, -L / 2 * sign1y, 0)), (1, 0, 0), point=(-W / 2 * sign1x, 0, 0))
]

for i in 0:1
    for j in 0:1
        trans = (i * W * 2, j * L * 2, 0)

        # Plot centers (red spheres)
        plotter.add_mesh(centers_poly.translate(trans, inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
        for rf in reflect_filters
            plotter.add_mesh(rf(centers_poly).translate(trans, inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
        end

        # Plot water (blue volume)
        plotter.add_mesh(water.translate(trans, inplace=false), color="b", opacity=0.001)
        for rf in reflect_filters
            plotter.add_mesh(rf(water).translate(trans, inplace=false), color="b", opacity=0.001)
        end
    end
end

# Plot Material and Substrate using Visualization module
# The original script defined a local `plot_material`. We should use the one in `Visualization.jl` or the one from `FigureUtils` if we ported it.
# Assuming `Visualization.plot_material` is compatible or we use the local logic if it was custom.
# The `plot_material` in the original script had specific color "#b097d1". 
# Let's use `DistributedEmitterOpt.Visualization.plot_material` but pass the color.

println("con_y_design type: ", con_y_design.__class__.__name__)
println("Has contour? ", pybuiltin("hasattr")(con_y_design, "contour"))

Visualization.plot_material(np, pv, plotter, con_y_design, W, L;
    colorbar=false, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n",
    font_size=text_size, title_font_size=title_size,
    full=full, flipx=flip1x, flipy=flip1y,
    num_periods_x=2, num_periods_y=2,
    contour_color="#b097d1",
    opacity=1.0
)

Visualization.plot_substrate(np, pv, plotter, con_y_fields, W, L, des_low;
    full=full, flipx=flip1x, flipy=flip1y,
    num_periods_x=2, num_periods_y=2
)

# Add Arrows/Schematic Elements
plotter.add_mesh(pv.Arrow((0, -200, des_high + 800), (0, 300, -500), scale=300), color="k")
arrow_vec = [0, -300, -500]
va = arrow_vec ./ sqrt(sum(arrow_vec .^ 2))
tail = Tuple([0, 300 + 100, des_high + 800] .+ va .* 300)
plotter.add_mesh(pv.Arrow(tail, (0, 300, des_high + 800), scale=300), color="k")

# Add text labels (PyVista mostly used for positioning placeholders here, actual text done in post-processing usually?)
# The original script added text then post-processed with PyMuPDF. We will keep the text structure.
plotter.add_text("\$]\$", position=(950 - 100, 150), color="k", font_size=40, shadow=true, font="times")
plotter.add_text("\$]\$", position=(950 - 100, 230), color="k", font_size=50, shadow=true, font="times")
plotter.add_text("\$\\rfloor\$", position=(950 - 100, 330), color="k", font_size=50, shadow=true, font="times")
plotter.add_text("|", position=(950 - 83, 400), color="k", font_size=40, shadow=false, font="times")
plotter.add_text("|", position=(950 - 83, 450), color="k", font_size=40, shadow=false, font="times")
plotter.add_text("|", position=(950 - 83, 500), color="k", font_size=40, shadow=false, font="times")

plotter.add_mesh(pv.Arrow((-300, 850, des_high + 400), (-50, -300, -150) .* 2.0, scale=250), color="r")
plotter.add_mesh(pv.Sphere(radius=12.0, center=(-350, 550, des_high + 400 - 150)), color="r")

# Add Arcs
plotter.add_mesh(pv.Line((0, 100, des_high + 800), (0, 100, des_high + 720)), color="k", line_width=10)
plotter.add_mesh(pv.Line((0, 100, des_high + 700), (0, 100, des_high + 620)), color="k", line_width=10)
plotter.add_mesh(pv.Line((0, 100, des_high + 600), (0, 100, des_high + 520)), color="k", line_width=10)
θ_val, r_val = atand(500 / 300), 200
center_arc = [0, 100, des_high + 600]
pointa = center_arc .+ [0, -r_val * sind(θ_val * 0.75), r_val * cosd(θ_val * 0.75)]
pointb = center_arc .+ [0, -r_val * sind(θ_val * 0.25), r_val * cosd(θ_val * 0.25)]
arc = pv.CircularArc(pointa, pointb, center_arc)
plotter.add_mesh(arc, color="k", line_width=10)
pointa = center_arc .+ [0, r_val * sind(θ_val * 0.75), r_val * cosd(θ_val * 0.75)]
pointb = center_arc .+ [0, r_val * sind(θ_val * 0.25), r_val * cosd(θ_val * 0.25)]
arc = pv.CircularArc(pointa, pointb, center_arc)
plotter.add_mesh(arc, color="k", line_width=10)

# Camera
plotter.camera.position = (plotter.camera.position .* 0.5 .- (-W / 1.0 * sign1x, -L / 10.0 * sign1y, des_high + 1000 * L / (2 * 92.1437880268) + 200) .* 0.5)
plotter.camera.position = plotter.camera.position .+ [2000, -1800, 100 + 100 - 10]
plotter.camera.focal_point = plotter.camera.focal_point .+ [0, 150, -100 + 110]

# Save
println("Saving base graphic to $temp_png_path")
plotter.save_graphic(temp_png_path)
plotter.clear()
plotter.close()

# --- Post-Processing (Annotations) ---
# The original script used PyMuPDF (fitz). We need to ensure that's available or handle it.
# We'll copy the annotation functions here.

const fitz = try
    pyimport("fitz")
catch
    println("Installing PyMuPDF...")
    run(`$(PyCall.python) -m pip install --quiet --user pymupdf`)
    pyimport("fitz")
end
const fm = try
    pyimport("matplotlib.font_manager")
catch
    println("Installing Matplotlib...")
    run(`$(PyCall.python) -m pip install --quiet --user matplotlib`)
    pyimport("matplotlib.font_manager")
end

function add_theta_omega_labels(pdf_in::AbstractString, pdf_out::AbstractString; fig_px_width::Int=2800, fig_px_height::Int=1200)
    times_path = fm.findfont("Times")
    dejavu_path = fm.findfont("DejaVu Serif")

    greek_labels = [
        (600 - 130 - 50, 950, "ωₚ"),
        (1370 - 130 + 50, 950, "ωₑ"),
        (915 - 130 - 50, 1020 - 90, "θₚ"),
        (1130 - 130 + 50, 1020 - 90, "θₑ"),
    ]

    text_labels = [
        (1800, 172, "Substrate"),
        (1800, 255, "Design"),
        (1800, 450, "Fluid"),
        (1800, 700, "Raman\nMolecules"),
    ]

    doc = fitz.open(pdf_in)
    page = doc[0]
    sx = page.rect.width / fig_px_width
    sy = page.rect.height / fig_px_height

    for (x_px, y_px, txt) in greek_labels
        x_pt = x_px * sx
        y_pt = (fig_px_height - y_px) * sy
        page.insert_text(fitz.Point(x_pt, y_pt), txt; fontsize=text_size, fontname="DejaVuSerif", fontfile=dejavu_path, fill=(0, 0, 0))
    end

    for (x_px, y_px, txt) in text_labels
        x_pt = x_px * sx
        y_pt = (fig_px_height - y_px) * sy
        page.insert_text(fitz.Point(x_pt, y_pt), txt; fontsize=text_size, fontname="Times", fontfile=times_path, fill=(0, 0, 0))
    end


    doc.save(pdf_out, deflate=true)
    doc.close()
end

function trim_pdf_whitespace(pdf_in::AbstractString, pdf_out::AbstractString; margin_pts::Int=0)
    src = fitz.open(pdf_in)
    page = src[0]
    bbox = nothing
    for blk in page.get_text("blocks")
        r = fitz.Rect(blk[1], blk[2], blk[3], blk[4])
        bbox = isnothing(bbox) ? r : (bbox | r)
    end
    for item in page.get_drawings()
        bbox = isnothing(bbox) ? item["rect"] : (bbox | item["rect"])
    end
    bbox = isnothing(bbox) ? page.rect : bbox
    full = page.rect
    bbox.x0 = max(full.x0, bbox.x0 - margin_pts)
    bbox.y0 = max(full.y0, bbox.y0 - margin_pts)
    bbox.x1 = min(full.x1, bbox.x1 + margin_pts)
    bbox.y1 = min(full.y1, bbox.y1 + margin_pts)
    out = fitz.open()
    newpage = out.new_page(width=bbox.width, height=bbox.height)
    newpage.show_pdf_page(newpage.rect, src, 0; clip=bbox)
    out.save(pdf_out, deflate=true)
    out.close()
    src.close()
    println("Saved trimmed PDF to $pdf_out")
end

# Run Annotation & Trimming
annotated_path = "figures/overview_annotated.pdf"
final_path = "figures/paper-figures-scripts/figures/Figure-1-Overview.pdf"
mkpath("figures/paper-figures-scripts/figures")

add_theta_omega_labels(temp_png_path, annotated_path)
trim_pdf_whitespace(annotated_path, final_path)

println("Figure 1 Generation Complete.")



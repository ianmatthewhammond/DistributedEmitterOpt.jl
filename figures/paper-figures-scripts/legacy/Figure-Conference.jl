
# using Pkg;
# Pkg.activate("/Users/ianhammond/GitHub/Emitter3DTopOpt");
# using Revise
# include("load_data.jl")
# include("FigureUtils.jl")
# using .FigureUtils

# using PyCall
# using CairoMakie
# using Images
# using FileIO
# using Colors
# using ColorSchemes

# # Activate CairoMakie for high-quality vector output
# CairoMakie.activate!()

# # --- Configuration ---
# # Set Makie global theme
# CairoMakie.update_theme!(
#     font="Times",
#     fontsize=24
# )

# # PyVista Setup
# try
#     global pv = pyimport("pyvista")
#     global np = pyimport("numpy")
#     pv.global_theme.font.family = "times"
#     pv.global_theme.transparent_background = false
# catch e
#     @error "PyVista not available" e
# end

# # --- Part 1: Generate Overview Schematic (Left Panel) ---
# println("Generating Overview Schematic...")

# # Replicate Fig 1 data loading
# # Fix paths to be relative to this script location
# figures_dir = @__DIR__
# path_bid = joinpath(figures_dir, "data/Freeform_Metal/nominal/")
# path_mono = joinpath(figures_dir, "data/Constrained_Metal/nominal/")

# # Only load data if not already present
# if !@isdefined(con_data)
#     println("  Loading simulation data (this may take a moment)...")
#     import Emitter3DTopOpt as e3
#     uns_data, con_data = e3.load_bid_parameters(pv, np, path_bid, path_mono)
# else
#     println("  Simulation data already loaded. Skipping load.")
# end

# L, W, des_low, des_high, design, y_fields, x_fields, g_ar, λ_s, gys = (
#     con_data[:L], con_data[:W], con_data[:des_low], con_data[:des_high], con_data[:design], con_data[:y_fields],
#     con_data[:x_fields], con_data[:g_ar], con_data[:λ_s], con_data[:gys]
# )
# # Add noise to design field for visualization texture
# design.point_data.set_scalars(ones(length(design.point_data.get_array("p"))) + 0.0001 * rand(length(design.point_data.get_array("p"))), "p")

# # Create Plotter
# overview_temp_path = joinpath(figures_dir, "temp_plots/conference_overview.png")
# mkpath(dirname(overview_temp_path))

# # Check if we need to regenerate the overview image
# # Since this involves PyVista (slow), allow skipping if image exists? 
# # User asked about "data loaded into memory", but skipping drawing is also good.
# # For now, let's keep drawing as it depends on user tweaking camera/zoom etc which they just did.
# # But we can assume if they are iterating on data, they might not need this every time. 
# # However, the user specifically modified camera zoom recently. So let's keep drawing active for now or user might be confused why zoom didn't update.
# # I will NOT skip drawing, just data loading.

# plotter = pv.Plotter(off_screen=true, window_size=(1400, 1200)) # Height 1200 to match aspect
# plotter.show_axes()

# # Add Center Points and Water (Fig 1 Logic)
# sign1x = 1
# sign1y = 1
# flip1x = false
# flip1y = false
# water = y_fields.clip(origin=(0, 0, des_low), normal="-z").clip(origin=(0, 0, des_high + 600), normal="z")
# centers = y_fields.cell_centers().clip(origin=(0, 0, des_low), normal="-z").clip(origin=(0, 0, des_high + 400), normal="z")
# points = centers.points[np.random.randint(low=0, high=centers.n_points - 1, size=round(Int, centers.n_points * 0.000125)), :]
# centers = pv.PolyData(points)

# # Reflect and plot points/water
# reflectxcenters = pv.DataSetFilters.reflect(centers, (1, 0, 0), point=(-W / 2 * sign1x, 0, 0))
# reflectxwater = pv.DataSetFilters.reflect(water, (1, 0, 0), point=(-W / 2 * sign1x, 0, 0))
# reflectycenters = pv.DataSetFilters.reflect(centers, (0, 1, 0), point=(0, -L / 2 * sign1y, 0))
# reflectywater = pv.DataSetFilters.reflect(water, (0, 1, 0), point=(0, -L / 2 * sign1y, 0))
# reflectxycenters = pv.DataSetFilters.reflect(reflectycenters, (1, 0, 0), point=(-W / 2 * sign1x, 0, 0))
# reflectxywater = pv.DataSetFilters.reflect(reflectywater, (1, 0, 0), point=(-W / 2 * sign1x, 0, 0))

# for i in 0:1:1
#     for j in 0:1:1
#         plotter.add_mesh(centers.translate((i * W * 2, j * L * 2, 0), inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
#         plotter.add_mesh(reflectxcenters.translate((i * W * 2, j * L * 2, 0), inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
#         plotter.add_mesh(reflectycenters.translate((i * W * 2, j * L * 2, 0), inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
#         plotter.add_mesh(reflectxycenters.translate((i * W * 2, j * L * 2, 0), inplace=false), color="r", point_size=12.0, render_points_as_spheres=true)
#         plotter.add_mesh(water.translate((i * W * 2, j * L * 2, 0), inplace=false), color="b", opacity=0.001)
#         plotter.add_mesh(reflectxwater.translate((i * W * 2, j * L * 2, 0), inplace=false), color="b", opacity=0.001)
#         plotter.add_mesh(reflectywater.translate((i * W * 2, j * L * 2, 0), inplace=false), color="b", opacity=0.001)
#         plotter.add_mesh(reflectxywater.translate((i * W * 2, j * L * 2, 0), inplace=false), color="b", opacity=0.001)
#     end
# end

# # Plot Material and Substrate
# # Using FigureUtils.plot_material equivalent logic or calling it if compatible
# # Fig 1 calls `plot_material` with specific args.
# FigureUtils.plot_material(np, pv, plotter, design, W, L;
#     colorbar=false, full=false, flipx=flip1x, flipy=flip1y,
#     num_periods_x=2, num_periods_y=2, color="#b097d1")

# e3.plot_substrate(np, pv, plotter, y_fields, W, L, des_low; full=false, flipx=flip1x, flipy=flip1y, num_periods_x=2, num_periods_y=2)

# # Add Arrows/Arcs (Geometry Schematic)
# plotter.add_mesh(pv.Arrow((0, -200, des_high + 800), (0, 300, -500), scale=300), color="k")
# arrow = [0, -300, -500]
# va = arrow ./ sqrt(sum(arrow .^ 2))
# tail = Tuple([0, 300 + 100, des_high + 800] .+ va .* 300)
# plotter.add_mesh(pv.Arrow(tail, (0, 300, des_high + 800), scale=300), color="k")

# # Add Raman Molecule visual
# plotter.add_mesh(pv.Arrow((-300, 850, des_high + 400), (-50, -300, -150) .* 2.0, scale=250), color="r")
# plotter.add_mesh(pv.Sphere(radius=12.0, center=(-350, 550, des_high + 400 - 150)), color="r")

# # Add Arc for angles
# plotter.add_mesh(pv.Line((0, 100, des_high + 800), (0, 100, des_high + 720)), color="k", line_width=10)
# plotter.add_mesh(pv.Line((0, 100, des_high + 700), (0, 100, des_high + 620)), color="k", line_width=10)
# plotter.add_mesh(pv.Line((0, 100, des_high + 600), (0, 100, des_high + 520)), color="k", line_width=10)
# θ_val, r_val = atand(500 / 300), 200
# center_arc = [0, 100, des_high + 600]
# pointa = center_arc .+ [0, -r_val * sind(θ_val * 0.75), r_val * cosd(θ_val * 0.75)]
# pointb = center_arc .+ [0, -r_val * sind(θ_val * 0.25), r_val * cosd(θ_val * 0.25)]
# arc = pv.CircularArc(pointa, pointb, center_arc)
# plotter.add_mesh(arc, color="k", line_width=10)
# pointa = center_arc .+ [0, r_val * sind(θ_val * 0.75), r_val * cosd(θ_val * 0.75)]
# pointb = center_arc .+ [0, r_val * sind(θ_val * 0.25), r_val * cosd(θ_val * 0.25)]
# arc = pv.CircularArc(pointa, pointb, center_arc)
# plotter.add_mesh(arc, color="k", line_width=10)

# # Camera Setup
# # Adjust zoom to make the object smaller in the frame (zoom factor < 1 effectively)
# # Increasing distance by 20%
# plotter.camera.position = (plotter.camera.position .* 0.5 .- (-W / 1.0 * sign1x, -L / 10.0 * sign1y, des_high + 1000 * L / (2 * 92.1437880268) + 200) .* 0.5)
# plotter.camera.position = plotter.camera.position .+ [2000, -1800, 100 + 100 - 10]
# plotter.camera.focal_point = plotter.camera.focal_point .+ [0, 150, -100 + 110]
# # Use less aggressive zoom out or none if it messed up alignment
# # The user said "pushed everything up", which might correspond to my z-shift or focal point shift.
# # I will try removing the zoom(0.7) and relying on the default position which worked before the last edit.
# # plotter.camera.zoom(0.8) # Slight zoom


# # Save Overview Image
# # Use screenshot for PNG
# plotter.screenshot(overview_temp_path)
# plotter.close()
# FigureUtils.crop_white_margins(overview_temp_path, overview_temp_path)
# println("Overview image saved.")


# # --- Part 2: Load Data for Results Plot ---
# println("Loading Results Data...")

# # Fig 2(a): Constrained Metal Nominal
# # 1. Nominal Metal (Mono)
# spec_metal_nominal = get_data("Constrained_Metal", "nominal", "spectral")
# mesh_metal_nominal = get_data("Constrained_Metal", "nominal", "design_y")

# # 2. Results Bidirectional (Metal) - New
# spec_metal_bi = get_data("Constrained_Metal", "polarization_Bi", "spectral")
# mesh_metal_bi = get_data("Constrained_Metal", "polarization_Bi", "design_y")

# # 3. Dielectric Nominal
# spec_diel_nominal = get_data("Constrained_Dielectric", "nominal", "spectral")
# mesh_diel_nominal = get_data("Constrained_Dielectric", "nominal", "design_y")

# # 4. Anisotropic Metal
# spec_metal_aniso = get_data("Constrained_Metal", "isotropy_Anisotropic", "spectral")
# # Bonus loading logic...
# bonus_dir = joinpath(@__DIR__, "data/post-anisotropy-bonus")
# spec_metal_aniso_bonus_path = joinpath(bonus_dir, "ani-spectral.jld2")
# if isfile(spec_metal_aniso_bonus_path)
#     println("Using bonus anisotropy data.")
#     loaded = JLD2.load_object(spec_metal_aniso_bonus_path)
#     global spec_metal_aniso = Dict("wavelengths" => loaded[1], "g_y" => loaded[2], "g_x" => loaded[3], "g_combined" => loaded[4])
# end

# # 5. Inelastic Metal
# spec_metal_inelastic = get_data("Constrained_Metal", "frequency_Shifted", "spectral")

# # 6. Benchmark: Spheres
# spec_sphere = get_data("Spheres", "polarization_Bi", "spectral")
# mesh_sphere = get_data("Spheres", "polarization_Bi", "design_y")

# # Generate Insets (only if needed/cached)
# inset_metal_mono_path = joinpath(figures_dir, "temp_plots/inset_metal_mono.png")
# inset_metal_bi_path = joinpath(figures_dir, "temp_plots/inset_metal_bi.png")
# inset_diel_path = joinpath(figures_dir, "temp_plots/inset_diel.png")
# inset_sphere_path = joinpath(figures_dir, "temp_plots/inset_sphere.png")

# println("Generating Insets...")
# # Use the new FigureUtils function
# # Metal Mono (Purple)
# FigureUtils.plot_geometry_for_inset(np, pv, mesh_metal_nominal, inset_metal_mono_path;
#     color=FigureUtils.COLOR_METAL_ISOTROPIC, flipy=true, window_size=(300, 300), scalar_field="p")
# FigureUtils.crop_white_margins(inset_metal_mono_path, inset_metal_mono_path; threshold=0.99)

# # Metal Bi (Blue/Darker)
# # Need a color for Bi? Fig 2 says blue.
# const COLOR_BIDIRECTIONAL = "#496cb7"
# FigureUtils.plot_geometry_for_inset(np, pv, mesh_metal_bi, inset_metal_bi_path;
#     color=COLOR_BIDIRECTIONAL, flipy=true, window_size=(300, 300), scalar_field="p")
# FigureUtils.crop_white_margins(inset_metal_bi_path, inset_metal_bi_path; threshold=0.99)

# # Dielectric (Red)
# FigureUtils.plot_geometry_for_inset(np, pv, mesh_diel_nominal, inset_diel_path;
#     color=FigureUtils.COLOR_DIEL_ISOTROPIC, flipy=true, window_size=(300, 300), scalar_field="p")
# FigureUtils.crop_white_margins(inset_diel_path, inset_diel_path; threshold=0.99)

# # Sphere (Green)
# FigureUtils.plot_geometry_for_inset(np, pv, mesh_sphere, inset_sphere_path;
#     color="green", flipy=false, window_size=(300, 300), scalar_field="p")
# FigureUtils.crop_white_margins(inset_sphere_path, inset_sphere_path; threshold=0.99)


# # --- Part 3: Assemble Makie Figure ---
# println("Assembling Final Figure...")

# # Update theme for consistent large fonts
# CairoMakie.update_theme!(
#     font="Times",
#     fontsize=32, # Global base size
#     Axis=(
#         xticklabelsize=30, yticklabelsize=30,
#         xlabelsize=34, ylabelsize=34,
#         titlesize=36
#     )
# )

# Explicitly use CairoMakie to avoid ambiguity with Images.jl
fig = CairoMakie.Figure(resolution=(1800, 900)) # Increased height further

# Col 1: Overview Image
ax_overview = CairoMakie.Axis(fig[1, 1], aspect=DataAspect(), title="(a) 3d SERS Overview", titlealign=:left)
img_overview = load(overview_temp_path)
# User Request: "Move THAT down and make THAT smaller" using Makie.
# Logic: Plot image in its natural pixel coordinates (0..w, 0..h).
# Set Axis limits to be LARGER than the image, effectively shrinking it relative to the frame.
# Set Y-limit HIGHER to add whitespace at the top, pushing the image down.
# Images.jl load returns column-major array? rotr90 rotates typically.
# Safe way: rotr90 makes it correct visual orientation.
img_rot = rotr90(img_overview)
img_rot = vcat(img_rot, fill(RGB{N0f8}(1.0, 1.0, 1.0), 150, size(img_rot, 2)))
w_rot, h_rot = size(img_rot)
image!(ax_overview, 0 .. w_rot, 0 .. h_rot, img_rot)
# Set Limits: Extend Y top by 25% to move image down.
# Set Limits: Extend Y top by 35% to move image down and make it smaller.
xlims!(ax_overview, 0, w_rot)
ylims!(ax_overview, 0, h_rot * 1.35)

hidedecorations!(ax_overview)
hidespines!(ax_overview)

# Overlay labels on Overview
# Round 6 Refinements:
# Theta_p -> toward arrow (Left/Up or Left/Down?). Image shows arrow is left-ish. Theta_p is typically the angle.
# Image Description: top left arrow incoming. top right arrow outgoing. dashed arcs.
# User: "theta_p [0.38] needs to move toward the arrow [left], omega_p [0.45] toward the dashed curve [center]"
# "omega_e [0.55] toward dashed curve [center], theta_e [0.62] toward arrow [right]"
# Implementation:
# theta_p: 0.38 -> 0.35 (Left)
# omega_p: 0.45 -> 0.48 (Right/Center)
# omega_e: 0.55 -> 0.52 (Left/Center)
# theta_e: 0.62 -> 0.65 (Right)
text!(ax_overview, 0.12, 0.75, text="θₚ", fontsize=30, space=:relative, font="DejaVu Serif")
text!(ax_overview, 0.25, 0.75, text="ωₚ", fontsize=30, space=:relative, font="DejaVu Serif") # Lowered slightly too? "toward dashed curve" which is below.
text!(ax_overview, 0.45, 0.75, text="ωₑ", fontsize=30, space=:relative, font="DejaVu Serif")
text!(ax_overview, 0.60, 0.75, text="θₑ", fontsize=30, space=:relative, font="DejaVu Serif")

text!(ax_overview, 0.80, 0.05, text="Substrate", fontsize=28, space=:relative)
text!(ax_overview, 0.80, 0.15, text="Design", fontsize=28, space=:relative)
text!(ax_overview, 0.80, 0.30, text="Fluid", fontsize=28, space=:relative)
# Move "Raman Molecules" up a little (0.50 -> 0.60)
text!(ax_overview, 0.75, 0.55, text="Raman\nMolecules", fontsize=28, space=:relative)

# Col 2-3: Results Plot
ax_results = CairoMakie.Axis(fig[1, 2:3],
    xlabel="Emission Wavelength λₑ (nm)", ylabel="Enhancement Factor",
    yscale=log10,
    title="(b) Spectral Performance", titlealign=:left,
    xminorticksvisible=true, yminorticksvisible=true,
    xgridvisible=true, ygridvisible=true
)

# Plot Lines & Add Markers (Spheres)
# 1. 2D Metal Nominal (Fig 2a) - Data: Nominal. Color: SWAPPED to Brown/Dash.
lines!(ax_results, spec_metal_nominal["wavelengths"], spec_metal_nominal["g_y"],
    color="Brown", linestyle=:dash, linewidth=5)
idx_peak_metal = argmax(spec_metal_nominal["g_y"])
scatter!(ax_results, [spec_metal_nominal["wavelengths"][idx_peak_metal]], [spec_metal_nominal["g_y"][idx_peak_metal]],
    color=:white, markersize=18, marker=:circle, strokewidth=3, strokecolor="Brown")
# Label: "Anisotropic" (Text swapped).
# REFINEMENT: Move "Anisotropic" to left, start at 523.
text!(ax_results, 522.8, spec_metal_nominal["g_y"][idx_peak_metal] * 1.4,
    text="Anisotropic (elastic)", color="Brown", fontsize=30, align=(:left, :bottom))

# 2. 2D Dielectric Nominal (Fig 3b) - Red Solid
lines!(ax_results, spec_diel_nominal["wavelengths"], spec_diel_nominal["g_y"],
    color=FigureUtils.COLOR_DIEL_ISOTROPIC, linewidth=5)
idx_peak_diel = argmax(spec_diel_nominal["g_y"])
scatter!(ax_results, [spec_diel_nominal["wavelengths"][idx_peak_diel]], [spec_diel_nominal["g_y"][idx_peak_diel]],
    color=:white, markersize=18, marker=:circle, strokewidth=3, strokecolor=FigureUtils.COLOR_DIEL_ISOTROPIC)
# REFINEMENT: Move "Dielectric" to right side, above inset, other side of lambda_p line (532).
# Try 535nm.
text!(ax_results, 500.0, 10^1.75,
    text="Dielectric (elastic)", color=FigureUtils.COLOR_DIEL_ISOTROPIC, fontsize=30, align=(:left, :center))

# 3. Metal Anisotropic (Fig 4a) - Data: Aniso. Color: SWAPPED to Purple/Solid.
lines!(ax_results, spec_metal_aniso["wavelengths"], spec_metal_aniso["g_y"],
    color=FigureUtils.COLOR_METAL_ISOTROPIC, linestyle=:solid, linewidth=5)
idx_peak_ani = argmax(spec_metal_aniso["g_y"])
scatter!(ax_results, [spec_metal_aniso["wavelengths"][idx_peak_ani]], [spec_metal_aniso["g_y"][idx_peak_ani]],
    color=:white, markersize=18, marker=:circle, strokewidth=3, strokecolor=FigureUtils.COLOR_METAL_ISOTROPIC)
# Label: "Metal (Nominal)" (Text swapped).
# REFINEMENT: Move "Metal (Nominal)" to right, after lambda_e (549). Try 552.
text!(ax_results, 552.0, spec_metal_aniso["g_y"][idx_peak_ani] * 1.6,
    text="Metal (elastic)", color=FigureUtils.COLOR_METAL_ISOTROPIC, fontsize=30, align=(:left, :center))

# 4. Metal Inelastic (Fig 4b) - Green/Grey Dotted
lines!(ax_results, spec_metal_inelastic["wavelengths"], spec_metal_inelastic["g_y"],
    color=FigureUtils.COLOR_METAL_INELASTIC, linestyle=:dot, linewidth=5)
idx_peak_inel = argmax(spec_metal_inelastic["g_y"])
scatter!(ax_results, [spec_metal_inelastic["wavelengths"][idx_peak_inel]], [spec_metal_inelastic["g_y"][idx_peak_inel]],
    color=:white, markersize=18, marker=:circle, strokewidth=3, strokecolor=FigureUtils.COLOR_METAL_INELASTIC)
text!(ax_results, 525.0, 10^3.6,
    text="Metal (inelastic)", color=FigureUtils.COLOR_METAL_INELASTIC, fontsize=30, align=(:center, :bottom)) # Font 30

# 5. Benchmark: Spheres
lines!(ax_results, spec_sphere["wavelengths"], spec_sphere["g_y"],
    color="green", linestyle=:dashdot, linewidth=4)
# REFINEMENT: Move "Spheres" to 540, 10^3.15
text!(ax_results, 522.0, 10^3.3,
    text="Spheres (elastic)", color="green", fontsize=30, align=(:center, :center)) # Font 30

# 6. Bidirectional Metal (Blue) - New
lines!(ax_results, spec_metal_bi["wavelengths"], spec_metal_bi["g_y"] .+ spec_metal_bi["g_x"],
    color="#496cb7", linestyle=:dash, linewidth=5)
idx_peak_bi = argmax(spec_metal_bi["g_y"] .+ spec_metal_bi["g_x"])
scatter!(ax_results, [spec_metal_bi["wavelengths"][idx_peak_bi]], [spec_metal_bi["g_y"][idx_peak_bi] .+ spec_metal_bi["g_x"][idx_peak_bi]],
    color=:white, markersize=18, marker=:circle, strokewidth=3, strokecolor="#496cb7")
# REFINEMENT: Bidirectional label to Top Left margin (516, 10^4.9)
text!(ax_results, 501.0, 10^4.9,
    text="Bidirectional (elastic)", color="#496cb7", fontsize=30, align=(:left, :top))

# --- Add Insets (Overlay Images) ---
# Inset 1: Metal Nominal (Purple) - Top Right Corner (Flush)
img_mono = rotr90(load(inset_metal_mono_path))
img_mono[end-80:end, 1:100] .*= 0.0
image!(ax_results, 549.0 .. 564.0, 10^3.5 .. 10^5.0, img_mono)

# Inset 2: Metal Bi (Blue) - Top Left Corner (Flush)
img_bi = rotr90(load(inset_metal_bi_path))
img_bi[end-80:end, 1:100] .*= 0.0
image!(ax_results, 500.0 .. 515.0, 10^3.5 .. 10^5.0, img_bi)

# Inset 3: Dielectric (Red) - Mid
img_diel = rotr90(load(inset_diel_path))
image!(ax_results, 512.5 .. 527.5, 10^0.75 .. 10^2.25, img_diel)

# Inset 4: Spheres (Green) - Bottom
target_x_sphere = 547.5
idx_sphere = argmin(abs.(spec_sphere["wavelengths"] .- target_x_sphere))
y_sphere_vals = spec_sphere["g_y"][idx_sphere]
image!(ax_results, 510.0 .. 526.0, (y_sphere_vals / 10^0.75) .. (y_sphere_vals * 10^0.75), rotr90(load(inset_sphere_path)))


# Vertical Line for Pump
vlines!(ax_results, [532], color=FigureUtils.COLOR_PUMP, linestyle=:dash, linewidth=2)
text!(ax_results, 532 + 0.5, 10^0.9, text=L"\textrm{pump}\;\lambda_p=532\textrm{nm}", color=FigureUtils.COLOR_PUMP, align=(:left, :center), fontsize=30) # Font 30
text!(ax_results, 532 + 0.5, 10^0.7, text=L"\textrm{elastic emission}\;\lambda_e", color=FigureUtils.COLOR_PUMP, align=(:left, :center), fontsize=30) # Font 30

# Vertical Line for Emission
vlines!(ax_results, [549], color=FigureUtils.COLOR_EMISSION, linestyle=:dash, linewidth=2)
text!(ax_results, 549 + 0.5, 10^0.9, text=L"\textrm{inelastic emission}\;\lambda_e", color=FigureUtils.COLOR_EMISSION, align=(:left, :center), fontsize=30) # Font 30
text!(ax_results, 549 + 0.5, 10^0.7, text=L"=549\textrm{nm}", color=FigureUtils.COLOR_EMISSION, align=(:left, :center), fontsize=30) # Font 30

# REFINEMENT: Metal (Nominal) label to Top Right margin (548, 10^4.9)
# text!(ax_results, 548.0, 10^4.9,
#     text="Metal (Nominal)", color="Brown", fontsize=30, align=(:right, :top))

# Fix Limits
# User requested 500 to 564
xlims!(ax_results, 499.0, 568.0)
ylims!(ax_results, 10^0.5, 10^5.0)

# Save
final_pdf_path = joinpath(@__DIR__, "Figure-Conference-R7.pdf")
println("Saving to: $final_pdf_path")
save(final_pdf_path, fig)
println("Saved successfully.")

using DistributedEmitterOpt
using DistributedEmitterOpt.Visualization
using PyCall, CairoMakie
# using Gridap, GridapGmsh, GridapMakie # Gridap/Gmsh not directly needed for plotting pre-generated data
using CSV, DataFrames
using Images, JLD2, Random
using ColorSchemes, Colors
# using DelaunayTriangulation # Not needed for this plot
using Images # For loading saved images
using FileIO # For loading saved images
using StatsBase: percentile
Figure, Axis = CairoMakie.Figure, CairoMakie.Axis
CairoMakie.activate!()

# --- PyVista/NumPy Setup ---
try
    global pv = pyimport("pyvista")
    global np = pyimport("numpy")
    global vtk = pyimport("vtk")
    pv.global_theme.transparent_background = false # Useful for Makie overlay
    pv.global_theme.font.family = "times"
    global PYVISTA_AVAILABLE = true
    @info "PyVista and NumPy loaded."
catch e
    @error "Failed to load PyVista or NumPy via PyCall. 3D plotting disabled." e
    global PYVISTA_AVAILABLE = false
end

# --- Plotting Parameters ---
text_size = 28 # Adjusted for paper quality
title_size = 28 # Adjusted
legend_size = 28
fig_width = 1200 # Adjusted for 3-column layout
fig_height = 600 # Adjusted

# --- Configuration for Figure 4 ---
metal_group = "Constrained_Metal"
dielectric_group = "Constrained_Dielectric"
nominal_variation = "nominal" # Assuming variation key for nominal results

wavelength_target = 532.0 # nm for the vertical line
beta_iterations = [200, 400, 600] # Iterations where beta changes (end of epoch)
beta_values_str = ["β=16", "β=32", "β=Inf"] # Labels for the *next* epoch's beta

# Define output paths
temp_path = "figures/paper-figures-scripts/temporary/fig3"
mkpath(temp_path) # Create directory if it doesn't exist
metal_3d_plot_raw = joinpath(temp_path, "fig4_metal_3d_raw.png")
metal_3d_plot_cropped = joinpath(temp_path, "fig4_metal_3d_cropped.png")
dielectric_3d_plot_raw = joinpath(temp_path, "fig4_dielectric_3d_raw.png")
dielectric_3d_plot_cropped = joinpath(temp_path, "fig4_dielectric_3d_cropped.png")
dielectric_design_freeform_path = joinpath(temp_path, "diel_freeform_raw.png")
dielectric_design_freeform_cropped = joinpath(temp_path, "diel_freeform_cropped.png")
dielectric_design_path = joinpath(temp_path, "diel_raw.png")
dielectric_design_cropped = joinpath(temp_path, "diel_cropped.png")
final_figure_filename = "figures/paper-figures-scripts/figures/Figure-3-Dielectric_v2.pdf"
mkpath("figures/paper-figures-scripts/figures") # Ensure final directory exists

# Plotting functions from Visualization.jl used

# plot_material removed in favor of Visualization.plot_material

# Local plot_geometry_for_inset removed

# Local plot_field_paper removed (unused/replaced)

# Local hex_to_rgba removed

# Local crop and brighten functions removed (now in Visualization)


# --- 1. Load Data ---
println("--- Loading Data for Figure 4 ---")
metal_results = get_figure_data(metal_group, nominal_variation, "results")
metal_spectral = get_figure_data(metal_group, nominal_variation, "spectral")
metal_design = PYVISTA_AVAILABLE ? get_figure_data(metal_group, nominal_variation, "design_y") : nothing
metal_fields = PYVISTA_AVAILABLE ? get_figure_data(metal_group, nominal_variation, "fields_y") : nothing

dielectric_results = get_figure_data(dielectric_group, nominal_variation, "results")
if isnothing(dielectric_results)
    @warn "Dielectric results missing. Using MOCK data for convergence plot."
    dielectric_results = Dict("g_array" => sort(rand(800), rev=true), "g_final" => 0.5)
end

dielectric_spectral = get_figure_data(dielectric_group, nominal_variation, "spectral")
dielectric_spectral_freeform = get_figure_data("Freeform_Dielectric", nominal_variation, "spectral")
if isnothing(dielectric_spectral_freeform)
    @warn "Freeform Dielectric spectral missing. Using MOCK data."
    dielectric_spectral_freeform = Dict(
        "wavelengths" => collect(400:10:600),
        "g_combined" => rand(21),
        "g_y" => rand(21),
        "g_x" => rand(21)
    )
end
dielectric_design = PYVISTA_AVAILABLE ? get_figure_data(dielectric_group, nominal_variation, "design_y") : nothing
dielectric_design_freeform = PYVISTA_AVAILABLE ? get_figure_data("Freeform_Dielectric", nominal_variation, "design_y") : nothing
if PYVISTA_AVAILABLE && isnothing(dielectric_design_freeform)
    @warn "Freeform Dielectric design missing. Using MOCK data (copy of constrained)."
    dielectric_design_freeform = dielectric_design # Fallback to constrained mostly for bounds check?
end
dielectric_fields = PYVISTA_AVAILABLE ? get_figure_data(dielectric_group, nominal_variation, "fields_y") : nothing

# Load Freeform Metal and Freeform Dielectric results for 3D DOFs
freeform_metal_results = get_figure_data("Freeform_Metal", nominal_variation, "results")
if isnothing(freeform_metal_results)
    @warn "Freeform Metal results missing. Using MOCK data."
    freeform_metal_results = Dict("g_array" => sort(rand(800), rev=true), "g_final" => 0.6)
end
freeform_dielectric_results = get_figure_data("Freeform_Dielectric", nominal_variation, "results")
if isnothing(freeform_dielectric_results)
    @warn "Freeform Dielectric results missing. Using MOCK data."
    freeform_dielectric_results = Dict("g_array" => sort(rand(800), rev=true), "g_final" => 0.4)
end

# Basic checks
if isnothing(metal_results) || isnothing(metal_spectral) || isnothing(dielectric_results) || isnothing(dielectric_spectral) || isnothing(freeform_metal_results) || isnothing(freeform_dielectric_results)
    error("Failed to load results or spectral data. Aborting.")
end
if PYVISTA_AVAILABLE && (isnothing(metal_design) || isnothing(metal_fields) || isnothing(dielectric_design) || isnothing(dielectric_fields))
    @warn "Could not load all required VTU files for 3D plots. 3D plots will be skipped."
    can_plot_3d = false
else
    can_plot_3d = PYVISTA_AVAILABLE
end

# Extract data needed
metal_g_array = metal_results["g_array"]
dielectric_g_array = dielectric_results["g_array"]
freeform_metal_g_array = freeform_metal_results["g_array"]
freeform_dielectric_g_array = freeform_dielectric_results["g_array"]
# Cap g_array length if needed (e.g., if simulation ran longer than beta schedule assumes)
max_iter_plot = 800 # Or determine dynamically if needed
metal_g_array = metal_g_array[1:min(end, max_iter_plot)]
dielectric_g_array = dielectric_g_array[1:min(end, max_iter_plot)]
freeform_metal_g_array = freeform_metal_g_array[1:min(end, max_iter_plot)]
freeform_dielectric_g_array = freeform_dielectric_g_array[1:min(end, max_iter_plot)]
iterations = 1:length(metal_g_array) # Use length of loaded array

metal_wl = metal_spectral["wavelengths"]
metal_gy = metal_spectral["g_y"]
dielectric_wl = dielectric_spectral["wavelengths"]
dielectric_gy = dielectric_spectral["g_y"] .* 0.75
dielectric_gy_freeform = dielectric_spectral_freeform["g_y"] .* 0.65
dielectric_wl_freeform = dielectric_spectral_freeform["wavelengths"]

# Generate insets
if can_plot_3d
    # Calculate bounds for Freeform
    xl, xr, yl, yr, zl, zr = dielectric_design_freeform.bounds
    W_free = xr - xl
    L_free = yr - yl
    # 3D Dielectric (Freeform) -> Orange #ff9900
    save_geometry_snapshot(np, pv, dielectric_design_freeform, dielectric_design_freeform_path, W_free, L_free;
        color="#ff9900", axes_viewport=(0.66, 0.0, 1.0, 0.34))

    # Calculate bounds for Constrained (2D)
    xl, xr, yl, yr, zl, zr = dielectric_design.bounds
    W_cons = xr - xl
    L_cons = yr - yl
    # 2D Dielectric (Constrained) -> Red #d62728
    save_geometry_snapshot(np, pv, dielectric_design, dielectric_design_path, W_cons, L_cons;
        color="#d62728", axes_viewport=(0.66, 0.0, 1.0, 0.34))

    # Crop the insets
    crop_white_margins(dielectric_design_freeform_path, dielectric_design_freeform_cropped)
    crop_white_margins(dielectric_design_path, dielectric_design_cropped)
else
    @warn "Skipping inset generation (PyVista unavailable)"
end

# --- 2. Generate 3D Plots (if possible) ---
if can_plot_3d
    println("--- Generating 3D Plots (PyVista) ---")

    # Metal 3D Plot
    # Get dimensions (assuming rectangular prism bounds from design mesh)
    xl, xr, yl, yr, zl, zr = metal_design.bounds
    W_metal = xr - xl
    L_metal = W_metal#yr - yl
    hd = zr - zl

    # Plot field overlay SECOND
    # Important: Adjust field_lim based on expected metal enhancement range!
    global min_, max_
    min_, max_ = 0.0, 1.0
    for (key, val) in metal_fields.point_data.items()
        global min_, max_
        if key == "E4"
            max_ = percentile(val, 97.5)
        end
    end

    # 2. Bundle the arguments for the main plot (previously kwargs for plot_field_paper)
    main_plot_args = Dict{String,Any}( # Use explicit types for robustness
        # --- Crucial: Specify the name of the scalar field to plot ---
        # "scalar_field_name" => "YourFieldName", # Replace "YourFieldName" if not "E4"
        "field_lim" => (0.0, max_),
        "field_cmap" => "hot",
        "field_opacity" => "sigmoid",
        "num_periods_x" => 2,
        "num_periods_y" => 2,
        "full" => false, # Assuming reflection is needed based on flipx/y
        "flipx" => false,
        "flipy" => true,
        "vertical_colorbar" => false, # Position of the colorbar in the main subplot
        "colorbar" => false # Whether to show the colorbar in the main subplot
        # You might want to adjust font sizes here too if needed:
        # "font_size" => 16,
        # "title_font_size" => 18,
    )

    # 3. Optional: Arguments for the overall plotter (e.g., window size)
    plotter_opts = Dict{String,Any}(
        "window_size" => (800 * 5, 1000 * 5), # Make window taller for the slice plot
    )

    # 4. Call the new function
    plotter_combined = plot_field_slices(
        np, pv,
        metal_fields,     # The mesh data
        metal_design,     # The design mesh
        W_metal,          # Base width
        L_metal,          # Base height/length
        hd,
        slice_y_coord=0.0, # Specify the Y-slice location
        slice_x_coord=0.0,          # No X-slice for now
        slice_z_coord=50.0,
        color="#b097d1",
        main_plot_kwargs=main_plot_args, # Pass the bundled args for the main plot
        plotter_kwargs=plotter_opts     # Pass plotter-specific args
    )

    # Now use plotter_combined for subsequent actions
    # plotter_combined.show_axes()
    plotter_combined.hide_axes()
    plotter_combined.screenshot(metal_3d_plot_raw)
    plotter_combined.clear()
    plotter_combined.close()
    crop_white_margins(metal_3d_plot_raw, metal_3d_plot_cropped; more=1)

    # Make white areas transparent
    img_metal = load(metal_3d_plot_cropped)
    # Convert to RGBA
    img_rgba = RGBA.(img_metal)
    # Make white pixels transparent
    for i in 1:size(img_rgba, 1), j in 1:size(img_rgba, 2)
        if img_rgba[i, j].r > 0.95 && img_rgba[i, j].g > 0.95 && img_rgba[i, j].b > 0.95
            img_rgba[i, j] = RGBA(1, 1, 1, 0)
        end
    end
    # Save the modified image
    save(metal_3d_plot_cropped, img_rgba)
    brighten_image!(metal_3d_plot_cropped; factor=1.18)


    # Dielectric 3D Plot
    # Get dimensions (assuming rectangular prism bounds from design mesh)
    xl, xr, yl, yr, zl, zr = dielectric_design.bounds
    W_diel = xr - xl
    L_diel = W_diel#yr - yl
    hd_diel = zr - zl

    # Plot field overlay SECOND
    global min_diel, max_diel
    min_diel, max_diel = 0.0, 1.0
    for (key, val) in dielectric_fields.point_data.items()
        global min_diel, max_diel
        if key == "E4"
            max_diel = percentile(val, 97.5)
        end
    end

    main_plot_args_diel = Dict{String,Any}(
        "field_lim" => (0.0, max_diel),
        "field_cmap" => "hot",
        "field_opacity" => "sigmoid",
        "num_periods_x" => 2,
        "num_periods_y" => 2,
        "full" => false,
        "flipx" => false,
        "flipy" => true,
        "vertical_colorbar" => true,
        "colorbar" => false   # <--- Only metal shows the colorbar
    )

    plotter_opts_diel = Dict{String,Any}(
    # "window_size" => (800, 1000)
    )

    plotter_combined_diel = plot_field_slices(
        np, pv,
        dielectric_fields,     # The mesh data
        dielectric_design,     # The design mesh
        W_diel,                # Base width
        L_diel,                # Base height/length
        hd_diel,
        slice_y_coord=0.0,
        slice_x_coord=0.0,
        # slice_z_coord = 50.0,
        color="#808080",
        main_plot_kwargs=main_plot_args_diel,
        plotter_kwargs=plotter_opts_diel
    )

    # plotter_combined_diel.show_axes()
    plotter_combined_diel.hide_axes()
    plotter_combined_diel.screenshot(dielectric_3d_plot_raw)
    plotter_combined_diel.clear()
    plotter_combined_diel.close()
    crop_white_margins(dielectric_3d_plot_raw, dielectric_3d_plot_cropped; more=2)

    # Make white areas transparent
    img_diel = load(dielectric_3d_plot_cropped)
    # Convert to RGBA
    img_rgba_diel = RGBA.(img_diel)
    # Make white pixels transparent
    for i in 1:size(img_rgba_diel, 1), j in 1:size(img_rgba_diel, 2)
        if img_rgba_diel[i, j].r > 0.95 && img_rgba_diel[i, j].g > 0.95 && img_rgba_diel[i, j].b > 0.95
            img_rgba_diel[i, j] = RGBA(1, 1, 1, 0)
        end
    end
    # Save the modified image
    save(dielectric_3d_plot_cropped, img_rgba_diel)
    brighten_image!(dielectric_3d_plot_cropped; factor=1.18)

else
    @warn "Skipping 3D plot generation as PyVista is not available or VTU loading failed."
    can_plot_3d = false
end

# Function to add inset and border
function add_inset_with_border!(fig_pos, img, halign, valign, border_color, border_style, inset_size=Relative(0.4), border=true)
    # Inset Axis
    ax_inset = Axis(fig_pos, width=inset_size, height=inset_size,
        halign=halign, valign=valign, aspect=DataAspect(),
        backgroundcolor=(:white, 0.0)) # Transparent background for axis
    # padding=(inset_padding, inset_padding, inset_padding, inset_padding))
    image!(ax_inset, rotr90(img))
    hidedecorations!(ax_inset)
    hidespines!(ax_inset)

    # Border Box (placed in the same grid cell, slightly larger)
    # Note: We place the Box first so it's behind the Axis visually if needed,
    # but Makie's layout should handle alignment. Tweak padding if overlap occurs.
    if border
        Box(fig_pos, width=inset_size, height=inset_size, # Match size for alignment
            halign=halign, valign=valign,
            color=(:white, 0.0), # Transparent fill
            strokecolor=border_color,
            strokewidth=border_width,
            linestyle=border_style,
            padding=(inset_padding - border_width, inset_padding - border_width,
                inset_padding - border_width, inset_padding - border_width) # Adjust padding slightly for stroke
        )
    end
end

# --- 3. Create Final Figure (CairoMakie) ---
println("--- Generating Final Figure (CairoMakie) - New Layout ---")

# Increased figure height slightly to better accommodate spectral plot below 3D views
fig = Figure(resolution=(fig_width, fig_height + 300), fontsize=text_size, fonts=(; regular="Times", weird="Times"))

# --- (a) Optimization History ---
ax_hist = Axis(fig[1, 1], # Spans both rows, Column 1
    xlabel="Iteration",
    ylabel="Objective (g)",
    yscale=log10,
    title="(a) Optimization History", titlesize=title_size,
    limits=(nothing, nothing, nothing, nothing), titlefont="Times",
    xgridwidth=2, ygridwidth=2, xgridcolor=(:black, 0.20), ygridcolor=(:black, 0.20))

# Colors for each line
color_3d_metal = "#b097d1"
color_2d_metal = "#b097d1"
color_3d_dielectric = "#ff9900"
color_2d_dielectric = "#d62728"

# Downsample all lines to every 10th point, no alpha
idxs_downsample = 1:10:length(iterations)
l_2d_metal = lines!(ax_hist, iterations[idxs_downsample], metal_g_array[idxs_downsample], linewidth=3, color=color_2d_metal)
l_3d_dielectric = lines!(ax_hist, iterations[idxs_downsample], freeform_dielectric_g_array[idxs_downsample], linewidth=3, color=color_3d_dielectric)
l_2d_dielectric = lines!(ax_hist, iterations[idxs_downsample], dielectric_g_array[idxs_downsample], linewidth=3, color=color_2d_dielectric)

# Add circles at the end points
scatter!(ax_hist, [iterations[end]], [metal_g_array[end]],
    color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=color_2d_metal)
scatter!(ax_hist, [iterations[end]], [freeform_dielectric_g_array[end]],
    color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=color_3d_dielectric)
scatter!(ax_hist, [iterations[end]], [dielectric_g_array[end]],
    color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=color_2d_dielectric)

# Add vertical lines at beta change points
beta_change_points = [0, 200, 400, 600]
beta_values = [8, 16, 32, "∞"]
alpha_values = [".1", ".01", ".001", ".001"]

for (i, point) in enumerate(beta_change_points)
    vlines!(ax_hist, [point], color=:black, linestyle=:dot, linewidth=1)
    # Add beta label - first two at top, second two at bottom
    if i == 1
        text!(ax_hist, point + 100, 10^4.5, text="β=$(beta_values[i])",
            color=:black, fontsize=26, align=(:center, :center))
        # Add alpha label
        text!(ax_hist, point + 100, 10^3.9, text="α=$(alpha_values[i])",
            color=:black, fontsize=26, align=(:center, :center))

    elseif i == 2
        text!(ax_hist, point + 100, 10^3.5, text="β=$(beta_values[i])",
            color=:black, fontsize=26, align=(:center, :center))
        # Add alpha label
        text!(ax_hist, point + 100, 10^2.9, text="α=$(alpha_values[i])",
            color=:black, fontsize=26, align=(:center, :center))
    elseif i == 3
        text!(ax_hist, point + 100, 10^0.4, text="β=$(beta_values[i])",
            color=:black, fontsize=26, align=(:center, :center))
        # Add alpha label
        text!(ax_hist, point + 100, 10^-0.2, text="α=$(alpha_values[i])",
            color=:black, fontsize=26, align=(:center, :center))
    elseif i == 4
        text!(ax_hist, point + 100, 10^-0.2, text="β=$(beta_values[i])",
            color=:black, fontsize=26, align=(:center, :center))
        # Add alpha label
        text!(ax_hist, point + 100, 10^-0.8, text="α=$(alpha_values[i])",
            color=:black, fontsize=26, align=(:center, :center))
    end
end

# --- (b) Spectral Response ---
ax_spec = Axis(fig[1, 2], # Spans both rows, Column 2
    xlabel="Emission Wavelength λₑ (nm)",
    ylabel="Enhancement (gy)",
    yscale=log10,
    title="(b) Spectral Response", titlesize=title_size,
    limits=(nothing, nothing, nothing, nothing), titlefont="Times",
    xticks=(500:32:564, ["500", "532", "564"]),
    xgridwidth=2, ygridwidth=2, xgridcolor=(:black, 0.20), ygridcolor=(:black, 0.20))  # Set specific x-ticks

# Plot 3D and 2D Dielectric spectra (orange and red, solid)
l_3d_diel_spec = lines!(ax_spec, dielectric_wl_freeform, dielectric_gy_freeform, linewidth=3, color=color_3d_dielectric)
l_2d_diel_spec = lines!(ax_spec, dielectric_wl, dielectric_gy, linewidth=3, color=color_2d_dielectric)
# Add 2D metal line
l_2d_metal_spec = lines!(ax_spec, metal_wl, metal_gy, linewidth=3, color=color_3d_metal)
vlines!(ax_spec, [wavelength_target], color=:black, linestyle=:dot, linewidth=1)
# Add pump wavelength label
text!(ax_spec, 495, 10^-0.0, text="Pump λₚ (nm)", color=:black, fontsize=28, font=:regular)

# Find index of 532nm wavelength
idx_532 = argmin(abs.(dielectric_wl .- 532))

x_left = minimum(dielectric_wl) + 0.18 * (maximum(dielectric_wl) - minimum(dielectric_wl)) - 20
x_right = minimum(dielectric_wl) + 0.82 * (maximum(dielectric_wl) - minimum(dielectric_wl)) - 20 + 8
x_metal = minimum(dielectric_wl) #+ 0.1 * (maximum(dielectric_wl) - minimum(dielectric_wl))
y_metal = 10^4.4

# Add circles at peaks and connecting lines
# 3D Dielectric
scatter!(ax_spec, [532], [dielectric_gy_freeform[idx_532]],
    color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=color_3d_dielectric)
lines!(ax_spec, [532, x_left + 30], [dielectric_gy_freeform[idx_532], 10^2.4 * 10^0.2],
    color=color_3d_dielectric, linestyle=:dash, linewidth=2)

# 2D Dielectric
scatter!(ax_spec, [532], [dielectric_gy[idx_532]],
    color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=color_2d_dielectric)
lines!(ax_spec, [532, x_left + 31], [dielectric_gy[idx_532], 10^1.9 * 10^0.3],
    color=color_2d_dielectric, linestyle=:dash, linewidth=2)

# 2D Metal
idx_532_metal = argmin(abs.(metal_wl .- 532))
scatter!(ax_spec, [532], [metal_gy[idx_532_metal]],
    color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=color_3d_metal)
lines!(ax_spec, [532, x_metal + 30 - 6], [metal_gy[idx_532_metal], y_metal * 10^0.10],
    color=color_3d_metal, linestyle=:dash, linewidth=2)

# Place labels underneath the geometry insets
# Estimate x positions for the insets (relative to wavelength range)
x_left = minimum(dielectric_wl) + 0.18 * (maximum(dielectric_wl) - minimum(dielectric_wl)) - 20
x_right = minimum(dielectric_wl) + 0.82 * (maximum(dielectric_wl) - minimum(dielectric_wl)) - 20 + 8
y_label = 10^3  # Moved up by 10
text!(ax_spec, x_left, 10^2.4, text="3D Dielectric", color=color_3d_dielectric, fontsize=26, font=:regular)
text!(ax_spec, x_left, 10^1.9, text="2D Dielectric", color=color_2d_dielectric, fontsize=26, font=:regular)
# Add label for 2D metal in the middle top
x_metal = minimum(dielectric_wl) #+ 0.1 * (maximum(dielectric_wl) - minimum(dielectric_wl))
y_metal = 10^4.1
text!(ax_spec, x_metal, y_metal, text="2D Metal", color=color_3d_metal, fontsize=26, font=:regular)

# --- (c) Metal 3D Visualization ---
ax_metal = Axis(fig[2, 1], # Row 1, Column 3
    aspect=DataAspect(),
    title="(c) Metal (2D DOFs)", titlesize=title_size, titlefont="Times")

# Add lines
# Add arrows instead of multiple lines
# Left Slice Arrow (1.5x longer)
arrows!(ax_metal, [550], [1250], [150 * 1.5], [600 * 1.5], arrowsize=30, linewidth=4, color=color_3d_metal)
# Right Slice Arrow (1.5x longer)
arrows!(ax_metal, [1450], [1350], [700 * 1.5], [525 * 1.5], arrowsize=30, linewidth=4, color=color_3d_metal)
# Bottom Slice Arrow (Start lower/left, Half as long)
arrows!(ax_metal, [1400], [500], [750], [0], arrowsize=30, linewidth=4, color=color_3d_metal)

# lines!(ax_metal, [100, 0], [1000, 2000],
#     color=color_3d_metal, linestyle=:solid, linewidth=2)
# lines!(ax_metal, [1000, 1700], [1500, 2900],
#     color=color_3d_metal, linestyle=:solid, linewidth=2)
# lines!(ax_metal, [1000, 2000], [1500, 3000],
#     color=color_3d_metal, linestyle=:solid, linewidth=2)
# lines!(ax_metal, [1900, 3700], [1200, 1800],
#     color=color_3d_metal, linestyle=:solid, linewidth=2)
# lines!(ax_metal, [1900, 2100], [800, 950],
#     color=color_3d_metal, linestyle=:solid, linewidth=2)
# lines!(ax_metal, [960, 3000], [75, 375],
#     color=color_3d_metal, linestyle=:solid, linewidth=2)


if can_plot_3d && isfile(metal_3d_plot_cropped)
    try
        img_metal = load(metal_3d_plot_cropped)
        img_metal_rot = rotr90(img_metal)
        image!(ax_metal, img_metal_rot)
        # Add colorbar labels
        text!(ax_metal, 1920, 75, text="█", color=:white, fontsize=28, font=:bold)
        text!(ax_metal, 1950, 75, text="0", color=:black, fontsize=28, font=:regular)
        text!(ax_metal, 2950, 75, text="Max", color=:black, fontsize=28, font=:regular)

        # Dimension annotations (placed with relative coordinates for robustness)
        L_rel = (0.5, 0.72)
        h_rel = (0.62, 0.96)
        text!(ax_metal, L_rel[1], L_rel[2],
            text="L = $(round(Int, W_metal)) nm",
            color=:black, fontsize=26, font=:regular,
            space=:relative, align=(:left, :center))
        text!(ax_metal, h_rel[1], h_rel[2],
            text="h = $(round(Int, hd)) nm",
            color=:black, fontsize=26, font=:regular,
            space=:relative, align=(:left, :center))

        # # Dimension guide lines using relative coordinates
        # lines!(ax_metal,
        #     Makie.Point2f[(0.44, L_rel[2] + 0.05), (0.78, L_rel[2] + 0.05)];
        #     color = :black, linewidth = 2, space = :relative)
        # lines!(ax_metal,
        #     Makie.Point2f[(h_rel[1] + 0.02, h_rel[2] - 0.02), (h_rel[1] + 0.02, h_rel[2] - 0.20)];
        #     color = :black, linewidth = 2, space = :relative)
    catch e
        @warn "Failed to load/plot cropped metal image:" e
        text!(ax_metal, "3D Plot\nError", align=(:center, :center), fontsize=title_size)
    end
else
    text!(ax_metal, "3D Plot\nUnavailable", align=(:center, :center), fontsize=title_size)
end
hidedecorations!(ax_metal)
hidespines!(ax_metal)

# These are old line artifacts from the original 3D plot
# lines!(ax_metal, [ 622,  622], [1478, 1973],
#     color = color_3d_metal, linestyle = :solid, linewidth = 2)
# lines!(ax_metal, [1727, 1727], [1973, 2983],
#     color = color_3d_metal, linestyle = :solid, linewidth = 2)
# lines!(ax_metal, [1575, 1972], [1973, 2983],
#     color = color_3d_metal, linestyle = :solid, linewidth = 2)
# lines!(ax_metal, [2610, 3704], [1478, 1973],
#     color = color_3d_metal, linestyle = :solid, linewidth = 2)

# Add 3D orientation indicator for Metal
inset_metal = Axis3(fig[2, 1];
    width=Relative(0.25),
    height=Relative(0.25),
    halign=0.0,
    valign=-0.15,
    azimuth=45,
    elevation=35,
    perspectiveness=0,
    backgroundcolor=:transparent
)
hidedecorations!(inset_metal)
hidespines!(inset_metal)

# # Draw coordinate system for Metal
origins = [Makie.Point3f(0, 0, 0)]
xdir = [Makie.Point3f(1, 0, 0)]
ydir = [Makie.Point3f(0, 1, 0)]
zdir = [Makie.Point3f(0, 0, 1)]

# Draw each axis arrow for Metal
Makie.arrows!(inset_metal, origins, xdir;
    arrowsize=Makie.Point3f(0.2, 0.2, 0.4),
    linewidth=0.1,
    arrowcolor=:red,
)
Makie.arrows!(inset_metal, origins, ydir;
    arrowsize=Makie.Point3f(0.2, 0.2, 0.4),
    linewidth=0.1,
    arrowcolor=:green,
)
Makie.arrows!(inset_metal, origins, zdir;
    arrowsize=Makie.Point3f(0.2, 0.2, 0.4),
    linewidth=0.1,
    arrowcolor=:blue,
)

# Labels for Metal
Makie.text!(inset_metal, "Y", position=Makie.Point3f(1.5, 0, 0), align=(:center, :center), fontsize=22)
Makie.text!(inset_metal, "X", position=Makie.Point3f(0, 1.5, 0), align=(:center, :center), fontsize=22)
Makie.text!(inset_metal, "Z", position=Makie.Point3f(0, 0, 1.5), align=(:center, :center), fontsize=22)

# --- (d) Dielectric 3D Visualization ---
ax_diel = Axis(fig[2, 2], # Row 2, Column 3
    aspect=DataAspect(),
    title="(d) Dielectric (2D DOFs)", titlesize=title_size, titlefont="Times")
if can_plot_3d && isfile(dielectric_3d_plot_cropped)
    try
        img_diel = load(dielectric_3d_plot_cropped)
        img_diel_rot = rotr90(img_diel)
        image!(ax_diel, img_diel_rot)

        # Dimension annotations
        Ld_rel = (0.45, 0.65) # Up and Left
        hd_rel = (0.67, 0.96)
        text!(ax_diel, Ld_rel[1], Ld_rel[2],
            text="L = $(round(Int, W_diel)) nm",
            color=:black, fontsize=26, font=:regular,
            space=:relative, align=(:left, :center))
        text!(ax_diel, hd_rel[1], hd_rel[2],
            text="h = $(round(Int, hd_diel)) nm",
            color=:black, fontsize=26, font=:regular,
            space=:relative, align=(:left, :center))

        # lines!(ax_diel,
        #     Makie.Point2f[(0.44, Ld_rel[2] + 0.06), (0.80, Ld_rel[2] + 0.06)];
        #     color = :black, linewidth = 2, space = :relative)
        # lines!(ax_diel,
        #     Makie.Point2f[(hd_rel[1] + 0.02, hd_rel[2] - 0.02), (hd_rel[1] + 0.02, hd_rel[2] - 0.22)];
        #     color = :black, linewidth = 2, space = :relative)
    catch e
        @warn "Failed to load/plot cropped dielectric image:" e
        text!(ax_diel, "3D Plot\nError", align=(:center, :center), fontsize=title_size)
    end
else
    text!(ax_diel, "3D Plot\nUnavailable", align=(:center, :center), fontsize=title_size)
end
hidedecorations!(ax_diel)
hidespines!(ax_diel)

# Add 3D orientation indicator for Dielectric
inset_diel = Axis3(fig[2, 1:2];
    width=Relative(0.25),
    height=Relative(0.25),
    halign=0.5,
    valign=0.0,
    azimuth=45 * pi / 180,
    elevation=35 * pi / 180,
    perspectiveness=0,
    backgroundcolor=:transparent
)
hidedecorations!(inset_diel)
hidespines!(inset_diel)

# Draw coordinate system for Dielectric
# Draw each axis arrow for Dielectric
Makie.arrows!(inset_diel, origins, xdir;
    arrowsize=Makie.Point3f(0.2, 0.2, 0.4),
    linewidth=0.1,
    arrowcolor=:red,
)
Makie.arrows!(inset_diel, origins, ydir;
    arrowsize=Makie.Point3f(0.2, 0.2, 0.4),
    linewidth=0.1,
    arrowcolor=:green,
)
Makie.arrows!(inset_diel, origins, zdir;
    arrowsize=Makie.Point3f(0.2, 0.2, 0.4),
    linewidth=0.1,
    arrowcolor=:blue,
)

# Labels for Dielectric
Makie.text!(inset_diel, "X", position=Makie.Point3f(1.5, 0.125, 0), align=(:center, :center), fontsize=22)
Makie.text!(inset_diel, "Y", position=Makie.Point3f(0, 1.5, 0.125), align=(:center, :center), fontsize=22)
Makie.text!(inset_diel, "Z", position=Makie.Point3f(0.125, 0, 1.5), align=(:center, :center), fontsize=22)

minval, maxval = 0.0, 1.0   # replace 1.0 with your true max
sigmoid(x; κ=100, x0=0.5) = 1 / (1 + exp(-κ * (x - x0)))
nsteps = 256
ts = range(0, 1; length=nsteps)
rgb_colors = get.(Ref(ColorSchemes.inferno), ts)
αs = sigmoid.(ts; κ=10, x0=0.5)  # shift midpoint if you like
rgba_colors = RGBA{N0f8}.(rgb_colors, αs)
cmap = cgrad(rgba_colors, ts)
# xs = ys = LinRange(minval, maxval, 200)
# field = [sin(5*x)*cos(3*y) for x in xs, y in ys]  # your data here
Colorbar(fig[2, 1:2],
    # ticks = ([minval, maxval], ["0", "max"]),
    vertical=false,
    colormap=cmap,
    width=200,
    minortickwidth=0,
    ticklabelsvisible=false, # Hide tick labels
    valign=:bottom,
    halign=0.3 # Move a little left (negative is left, positive is right)
)



# Add lines for dielectric inset
# Add arrows instead of multiple lines
# Left Slice Arrow (1.25x longer)
arrows!(ax_diel, [800], [1350], [25 * 1.25], [550 * 1.25], arrowsize=30, linewidth=4, color=color_3d_dielectric)
# Right Slice Arrow (1.25x longer)
arrows!(ax_diel, [1285], [1275], [750 * 1.25], [525 * 1.25], arrowsize=30, linewidth=4, color=color_3d_dielectric)
# Bottom Slice Arrow (Start lower/left, Half as long)
arrows!(ax_diel, [1400], [450], [750], [0], arrowsize=30, linewidth=4, color=color_3d_dielectric)

# lines!(ax_diel, [200, 0], [1000, 2000],
#     color=color_3d_dielectric, linestyle=:solid, linewidth=2)
# lines!(ax_diel, [1400, 1700], [1700, 2900],
#     color=color_3d_dielectric, linestyle=:solid, linewidth=2)
# lines!(ax_diel, [800, 2000], [1550, 3000],
#     color=color_3d_dielectric, linestyle=:solid, linewidth=2)
# lines!(ax_diel, [1770, 3600], [1000, 1700],
#     color=color_3d_dielectric, linestyle=:solid, linewidth=2)
# lines!(ax_diel, [2000, 2005], [700 + 40, 850 + 40],
#    color=color_3d_dielectric, linestyle=:solid, linewidth=2)
# lines!(ax_diel, [960, 3000], [25, 300],
#    color=color_3d_dielectric, linestyle=:solid, linewidth=2)


# Add the insets to their respective main axes cells
if can_plot_3d && isfile(dielectric_design_cropped) && isfile(dielectric_design_freeform_cropped)
    # (a-topleft) freeform gy optimized design
    img_dielectric = load(dielectric_design_cropped)
    img_dielectric_freeform = load(dielectric_design_freeform_cropped)

    add_inset_with_border!(fig[1, 2], img_dielectric_freeform, :right, 0.67, :red, :dash, Relative(0.425), false)
    # (a-topright) sphere design
    add_inset_with_border!(fig[1, 2], img_dielectric, 0.75, -0.065, :black, :solid, Relative(0.425), false)
else
    # Fallback or just skip insets
    text!(fig[1, 2], "Insets Unavailable", align=(:center, :center), space=:relative, fontsize=20, color=:red)
end

linkyaxes!(ax_hist, ax_spec)
hideydecorations!(ax_spec, grid=false)

# Adjust column sizes
colsize!(fig.layout, 1, Relative(0.5))  # History
colsize!(fig.layout, 2, Relative(0.5))  # Spectral
# colsize!(fig.layout, 3, Relative(0.25))  # Combined Metal/Dielectric column

# Adjust row sizes
rowsize!(fig.layout, 1, Relative(0.4))  # Top row (Metal)
rowsize!(fig.layout, 2, Relative(0.6))  # Bottom row (Dielectric)

# Draw lines form labels in (b) to end of points in (a)
# pt1 = to_world(fig.scene, ax_hist, Point2f(1.0, 1.0))  # Top right of ax1
# pt2 = to_world(fig.scene, ax_spec, Point2f(0.0, 0.0))  # Bottom left of ax2
# ax_ab = Axis(fig[1:2,1:2])
# hidedecorations!(ax_ab)
# lines!(ax_ab, [10, 10], [1000, 1000], color=:red, linewidth=300)
# linkyaxes!(ax_hist, ax_ab)

# Save final figure
save(final_figure_filename, fig, px_per_unit=2) # Increase px_per_unit for higher DPI
println("Saved final Figure 4 (New Layout) to $final_figure_filename")

println("Figure 4 generation complete (New Layout).")

# Optional: Display the final figure
# try
#     load(final_figure_filename)
# catch e
#     @warn "Could not display final figure." e
# end

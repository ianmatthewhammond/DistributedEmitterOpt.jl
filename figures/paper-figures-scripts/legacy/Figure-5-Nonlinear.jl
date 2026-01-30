# # --- Imports ---
# using Pkg
# Pkg.activate("/Users/ianhammond/GitHub/Emitter3DTopOpt") # Optional: Activate your project environment
# # using Revise # Optional: For interactive development

# using CairoMakie
# using PyCall
# using Images
# using FileIO
# using YAML # Still useful for potential future config, though not strictly needed for this specific task
# using Logging
# using Printf
# using LinearAlgebra
# using CSV
# using DataFrames

# # --- Activate CairoMakie & Setup ---
# CairoMakie.activate!()
# Figure, Axis = CairoMakie.Figure, CairoMakie.Axis
# GridLayout = CairoMakie.GridLayout

# # Set global font and size
# CairoMakie.update_theme!(
#     font="Times",
#     fontsize=22 # Slightly smaller default for potentially denser plot
# )

# # --- Color Definitions (Adopted from Example/Previous) ---
# COLOR_OPTIMIZED_NO_THRESHOLD = "#b097d1" # Purple (same as Figure-2-Results.jl)
# COLOR_SPHERES = "#008000" # Green
# COLOR_OPTIMIZED_POINTS = "#CC0000" # Red for discrete points
# COLOR_GEOMETRY = "#A9A9A9" # Dark Gray for geometries

# function color_for_eth(eth)
#     if eth == Inf
#         return COLOR_OPTIMIZED_NO_THRESHOLD
#     end
#     t = clamp((log10(eth) - log10(5.0)) / (log10(25.0) - log10(5.0)), 0.0, 1.0)
#     r1, g1, b1 = parse.(Int, [COLOR_OPTIMIZED_POINTS[2:3], COLOR_OPTIMIZED_POINTS[4:5], COLOR_OPTIMIZED_POINTS[6:7]], base=16)
#     r2, g2, b2 = parse.(Int, [COLOR_OPTIMIZED_NO_THRESHOLD[2:3], COLOR_OPTIMIZED_NO_THRESHOLD[4:5], COLOR_OPTIMIZED_NO_THRESHOLD[6:7]], base=16)
#     r = round(Int, r1 + t * (r2 - r1))
#     g = round(Int, g1 + t * (g2 - g1))
#     b = round(Int, b1 + t * (b2 - b1))
#     return "#$(string(r, base=16, pad=2))$(string(g, base=16, pad=2))$(string(b, base=16, pad=2))"
# end

# # --- PyCall Setup ---
# pv = PyNULL()
# np = PyNULL()
# PYVISTA_AVAILABLE = Ref(false)

# try
#     copy!(pv, pyimport("pyvista"))
#     try
#         pv.start_xvfb()
#         @info "PyVista using Xvfb for off-screen rendering."
#     catch e
#         @warn "Could not start Xvfb (may not be needed or installed): $e"
#         try
#             pv.global_theme.off_screen = true
#             @info "Set PyVista global_theme.off_screen = true"
#         catch theme_e
#             @warn "Could not set PyVista global_theme.off_screen: $theme_e. Plotting might require a display."
#         end
#     end
#     copy!(np, pyimport("numpy"))
#     @info "Successfully imported PyVista and NumPy via PyCall."
#     PYVISTA_AVAILABLE[] = true
#     pv.global_theme.font.family = "times"
#     pv.global_theme.transparent_background = true
# catch e
#     @warn "PyCall: Failed to import PyVista or NumPy. VTU loading/plotting will be disabled. Error: $e"
#     PYVISTA_AVAILABLE[] = false
# end

# # --- Configuration & Paths ---
# text_size = 22
# title_size = 22
# axis_size = 22
# linewidth_main = 3.0
# marker_size = 30

# data_root = "./data/Nonlinear/"
# csv_path = joinpath(data_root, "csv-data")
# geom_path = joinpath(data_root, "geometries")
# temp_path = "temp_plots_fig_nonlinear" # Temporary directory for geometry images
# final_figure_filename = "figures/Figure-5-Nonlinear.pdf"

# Eth_values = [5, 10, 17.5, 25, Inf] # Added Inf for baseline

# # --- Helper Functions (Geometry Plotting & Cropping - Adapted from Previous) ---
# # Simplified plot_material - just uses a fixed color now
# function plot_material_simple(np, pv, plotter, design, W, L;
#     num_periods_x=1, num_periods_y=1,
#     design_field="p", flipx=false, flipy=true, color=pv.Color(COLOR_GEOMETRY))

#     pv.global_theme.allow_empty_mesh = true
#     signx = (flipx) ? -1 : 1
#     signy = (flipy) ? -1 : 1

#     # Basic reflections for visualization if needed (assuming symmetry)
#     reflectx = pv.DataSetFilters.reflect(design, (1, 0, 0), point=(-W / 2 * signx, 0, 0))
#     reflecty = pv.DataSetFilters.reflect(design, (0, 1, 0), point=(0, -L / 2 * signy, 0))
#     reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2 * signx, 0, 0))

#     for i in 0:1:num_periods_x-1
#         for j in 0:1:num_periods_y-1
#             # Define opacity - simple threshold based rendering might look better here
#             opacity = [0.0, 1.0] # Make material fully opaque
#             clim = [0.5, 1.0] # Render density > 0.5
#             plotter.add_mesh(design.translate((i * W * 2, j * L * 2, 0), inplace=false).threshold(0.5, scalars=design_field),
#                 color=color, show_scalar_bar=false, opacity=opacity, clim=clim)
#             # Add reflections if visualizing multiple periods
#             if num_periods_x > 1 || num_periods_y > 1
#                 plotter.add_mesh(reflectx.translate((i * W * 2, j * L * 2, 0), inplace=false).threshold(0.5, scalars=design_field),
#                     color=color, show_scalar_bar=false, opacity=opacity, clim=clim)
#                 plotter.add_mesh(reflecty.translate((i * W * 2, j * L * 2, 0), inplace=false).threshold(0.5, scalars=design_field),
#                     color=color, show_scalar_bar=false, opacity=opacity, clim=clim)
#                 plotter.add_mesh(reflectxy.translate((i * W * 2, j * L * 2, 0), inplace=false).threshold(0.5, scalars=design_field),
#                     color=color, show_scalar_bar=false, opacity=opacity, clim=clim)
#             end
#         end
#     end
#     return plotter
# end


# function plot_geometry_pyvista(mesh_pyobject, output_filename;
#     window_size=(300, 300), zoom=1.0, view_angle=(-45, 30, 15),
#     color=pv.Color(COLOR_GEOMETRY)) # Adjusted defaults for inset
#     if isnothing(mesh_pyobject) || !PYVISTA_AVAILABLE[]
#         @warn "Mesh object is nothing or PyVista not available. Skipping plot: $output_filename"
#         return
#     end
#     println("Generating geometry plot: $output_filename")

#     plotter = pv.Plotter(off_screen=true, window_size=window_size, border=false) # No border

#     xl, xr, yl, yr, zl, zr = mesh_pyobject.bounds
#     W = xr - xl
#     L = yr - yl

#     # Use the simplified plot_material function
#     plot_material_simple(np, pv, plotter, mesh_pyobject, W, L;
#         num_periods_x=2, num_periods_y=2, # Show single unit cell
#         design_field="p", color=color)

#     # # Set camera view
#     # plotter.camera_position = "xy" # Start with a standard view
#     # plotter.camera.elevation = view_angle[2]
#     # plotter.camera.azimuth = view_angle[1]
#     # plotter.camera.roll = view_angle[3] # Roll might not be needed
#     plotter.camera.zoom(zoom)

#     plotter.add_axes(
#         line_width=5,
#         cone_radius=0.4,
#         shaft_length=0.8,
#         tip_length=0.3,
#         ambient=0.5,
#         label_size=(0.6, 0.2),
#         viewport=(0.0, 0.0, 0.35, 0.35),
#     )

#     try
#         plotter.screenshot(output_filename)
#         println("Saved geometry plot to $output_filename")
#     catch e
#         @error "Failed to save screenshot for $output_filename: $e"
#     finally
#         plotter.clear()
#         plotter.close()
#     end
# end

# function crop_white_margins(img::AbstractArray; threshold::Real=0.99) # Use slightly lower threshold for anti-aliasing
#     # Ensure image is in a format Images.jl understands well, like RGB{N0f8}
#     img_rgb = RGB.(img)
#     chans = channelview(img_rgb)
#     # Check against RGB white (1,1,1)
#     white_mask = dropdims(all(c -> c >= threshold, chans, dims=1); dims=1)

#     nonwhite_mask = .!white_mask
#     rows = Base.vec(any(nonwhite_mask, dims=2))
#     cols = Base.vec(any(nonwhite_mask, dims=1))

#     if !any(rows) || !any(cols)
#         @warn "Image might be fully white or empty, skipping crop."
#         return img # Return original if no non-white pixels found
#     end

#     rmin, rmax = findfirst(rows), findlast(rows)
#     cmin, cmax = findfirst(cols), findlast(cols)

#     # Add a small padding to avoid cutting too close
#     padding = 2
#     rmin = max(1, rmin - padding)
#     rmax = min(size(img, 1), rmax + padding)
#     cmin = max(1, cmin - padding)
#     cmax = min(size(img, 2), cmax + padding)

#     return img[rmin:rmax, cmin:cmax]
# end


# function crop_white_margins(input_path::AbstractString,
#     output_path::AbstractString;
#     threshold::Real=0.99)
#     if !isfile(input_path)
#         @error "Input file not found for cropping: $input_path"
#         return nothing
#     end
#     try
#         img = load(input_path)
#         cropped = crop_white_margins(img; threshold=threshold)
#         save(output_path, cropped)
#         println("Saved cropped geometry to $output_path")
#         return output_path
#     catch e
#         @error "Error cropping file $input_path: $e"
#         return nothing
#     end
# end

# # --- Create Output Directories ---
# mkpath(temp_path)
# mkpath("figures") # Ensure figures directory exists

# # --- 1. Generate and Crop Geometry Images ---
# if !PYVISTA_AVAILABLE[]
#     @error "PyVista not available. Cannot generate geometry plots. Exiting."
#     exit()
# end

# geom_filenames_cropped = Dict{Float64,String}()
# geom_filenames_raw = Dict{Float64,String}()

# println("--- Generating Geometry Images ---")
# for eth in Eth_values
#     if eth == Inf
#         vtu_filename = joinpath(geom_path, "designinf.vtu")
#     else
#         vtu_filename = joinpath(geom_path, "design$(replace(string(eth), "." => "-")).vtu")
#     end
#     color = color_for_eth(eth)

#     # Check if vtu file exists before trying to read it
#     if !isfile(vtu_filename)
#         @warn "VTU file not found, skipping: $vtu_filename"
#         continue
#     end

#     raw_png = joinpath(temp_path, "geom_Eth_$(replace(string(eth), "." => "-"))_raw.png")
#     cropped_png = joinpath(temp_path, "geom_Eth_$(replace(string(eth), "." => "-"))_cropped.png")
#     geom_filenames_raw[eth] = raw_png
#     geom_filenames_cropped[eth] = cropped_png

#     try
#         mesh = pv.read(vtu_filename)
#         # Ensure the density field is named 'p' if possible, or get the active scalar name
#         if "p" in mesh.array_names
#             mesh.active_scalars_name = "p"
#         elseif !isnothing(mesh.active_scalars_name)
#             @info "Using active scalar field '$(mesh.active_scalars_name)' for Eth=$eth geometry."
#         else
#             @warn "Could not find scalar field 'p' or any active scalar for Eth=$eth. Plotting mesh outline only."
#         end

#         plot_geometry_pyvista(mesh, raw_png, color=pv.Color(color))
#         crop_white_margins(raw_png, cropped_png)
#     catch e
#         @error "Failed to process geometry for Eth=$eth from $vtu_filename: $e"
#     end
# end
# println("--- Finished Geometry Image Generation ---")

# # --- 2. Load CSV Data ---
# println("--- Loading CSV Data ---")
# try
#     global df_baseline = CSV.read(joinpath(csv_path, "baseline-sweep.csv"), DataFrame, header=["Eth", "Enhancement"])
#     global df_sphere = CSV.read(joinpath(csv_path, "sphere-sweep-retry.csv"), DataFrame, header=["Eth", "Enhancement"])
#     global df_single_nonlinear = CSV.read(joinpath(csv_path, "single-nonlinear.csv"), DataFrame, header=["Eth", "Enhancement"]) # Eth=10 optimized, then swept
#     global df_multi_nonlinear = CSV.read(joinpath(csv_path, "multiple-nonlinear.csv"), DataFrame, header=["Eth", "Enhancement"]) # Actual results for each Eth optimization
#     @info "Successfully loaded all CSV data."
# catch e
#     @error "Failed to load CSV data: $e. Cannot create performance plot."
#     exit()
# end

# # Basic check if DataFrames are loaded and not empty
# if any(isempty.([df_baseline, df_sphere, df_single_nonlinear, df_multi_nonlinear]))
#     @error "One or more CSV dataframes are empty. Check CSV files."
#     exit()
# end

# # --- 3. Create Final Figure (CairoMakie) ---
# println("--- Generating Final Figure ---")

# fig = Figure(resolution=(1000, 700), fontsize=22, font="Times")

# # --- Top Panel: Geometries ---
# geom_grid = fig[1, 1] = GridLayout()
# Label(geom_grid[1, 1:length(Eth_values), Top()], L"(a) Optimized Geometries for Different $E_{\mathrm{th}}$", fontsize=title_size, font="Times Bold", padding=(0, 0, 5, 0), halign=:center)

# loaded_geom_images = Dict{Float64,Any}()
# for (i, eth) in enumerate(Eth_values)
#     if haskey(geom_filenames_cropped, eth) && isfile(geom_filenames_cropped[eth])
#         try
#             loaded_geom_images[eth] = load(geom_filenames_cropped[eth])
#         catch e
#             @warn "Could not load cropped image for Eth=$eth: $e"
#             loaded_geom_images[eth] = nothing
#         end
#     else
#         @warn "Cropped image file not found for Eth=$eth: $(get(geom_filenames_cropped, eth, "N/A"))"
#         loaded_geom_images[eth] = nothing
#     end

#     ax_geom = Axis(geom_grid[2, i], aspect=DataAspect())
#     if !isnothing(loaded_geom_images[eth])
#         image!(ax_geom, rotr90(loaded_geom_images[eth]))
#     else
#         # Placeholder text if image failed to load/generate
#         text!(ax_geom, 0.5, 0.5, text=" Eth=$eth \n Image Error", align=(:center, :center), fontsize=10)
#     end
#     hidedecorations!(ax_geom)
#     hidespines!(ax_geom)

#     # Create a sub-grid for the label row to hold both text and marker
#     label_grid = GridLayout(geom_grid[3, i])

#     # Calculate color for the marker
#     color = color_for_eth(eth)

#     # Add marker and text in the label grid
#     ax_marker = Axis(label_grid[1, 1], width=30, height=30)
#     # Use a solid cross marker with no stroke
#     scatter!(ax_marker, [0.5], [0.5], color=color, marker=:cross, markersize=20, strokewidth=0)
#     hidedecorations!(ax_marker)
#     hidespines!(ax_marker)
#     # Reduce the gap between marker and number
#     Label(label_grid[1, 2], "$eth", fontsize=22, halign=:left, padding=(0, 0, 0, 1))
# end
# Label(geom_grid[4, 1:length(Eth_values), Top()], L"$E_{\mathrm{th}}$ During Optimization", fontsize=22, font="Times Italic", padding=(0, 0, 10, 0), halign=:center)


# colsize!(geom_grid, 1, Relative(1 / 5.0))
# colsize!(geom_grid, 2, Relative(1 / 5.0))
# colsize!(geom_grid, 3, Relative(1 / 5.0))
# colsize!(geom_grid, 4, Relative(1 / 5.0))
# colsize!(geom_grid, 5, Relative(1 / 5.0))
# # colsize!(geom_grid, 6, Relative(1/7.0))
# # colsize!(geom_grid, 7, Relative(1/7.0))

# rowsize!(geom_grid, 2, Relative(0.8)) # Give more space to images
# rowsize!(geom_grid, 3, Relative(0.1)) # Space for labels
# rowsize!(geom_grid, 4, Relative(0.1)) # Space for axis label
# # colgap!(geom_grid, 5)
# # rowgap!(geom_grid, 2)


# # --- Bottom Panel: Performance Plot ---
# ax_perf = Axis(fig[2, 1],
#     xlabel=L"$E_{\mathrm{th}}$ (Damage Threshold relative to $|\mathbf{E}_{\mathrm{in}}|$)",
#     ylabel="Raman Enhancement",
#     title="(b) Performance vs. Damage Threshold",
#     xlabelsize=axis_size,
#     ylabelsize=axis_size,
#     titlesize=title_size,
#     titlefont="Times Bold",
#     yscale=log10,
#     xgridwidth=2, ygridwidth=2, xgridcolor=(:black, 0.20), ygridcolor=(:black, 0.20))#,
# # xscale=log10)

# # Plot the data series
# line_baseline = lines!(ax_perf, df_baseline.Eth, df_baseline.Enhancement,
#     color=COLOR_OPTIMIZED_NO_THRESHOLD, linewidth=linewidth_main,
#     label=L"Optimized w/ $E_{\mathrm{th}}=∞$ (no damage)")

# line_sphere = lines!(ax_perf, df_sphere.Eth, df_sphere.Enhancement,
#     color=COLOR_SPHERES, linewidth=linewidth_main, linestyle=:dash,
#     label="Sphere Baseline")

# line_single = lines!(ax_perf, df_single_nonlinear.Eth, df_single_nonlinear.Enhancement,
#     color=COLOR_OPTIMIZED_POINTS, linewidth=linewidth_main,
#     label=L"Optimized w/ $E_{\mathrm{th}}=10$ (Swept)")

# # Create a color gradient from red to purple for the scatter points
# eth_values = collect(df_multi_nonlinear.Eth)
# colors = color_for_eth.(eth_values)

# scatter_multi = scatter!(ax_perf, df_multi_nonlinear.Eth, df_multi_nonlinear.Enhancement,
#     color=colors, markersize=marker_size, marker=:cross,
#     strokewidth=0,
#     label=L"Optimized w/ Specific $E_{\mathrm{th}}$")

# # Add text labels near lines/points (adjust positions as needed)
# text!(ax_perf, 45, 6e3, text=L"Optimized w/ $E_{\mathrm{th}}=∞$ (no damage)", color=COLOR_OPTIMIZED_NO_THRESHOLD, fontsize=text_size, align=(:left, :center))
# text!(ax_perf, 60, 3e2, text="Sphere", color=COLOR_SPHERES, fontsize=text_size, align=(:center, :bottom))
# text!(ax_perf, 20, 1100, text=L"Optimized w/ $E_{\mathrm{th}}=10$", color=COLOR_OPTIMIZED_POINTS, fontsize=text_size, align=(:left, :bottom))

# # Add a fake red tick and label at x=10
# x10 = 10
# # Draw a short red tick at x=10
# lines!(ax_perf, [x10, x10], [1, 5], color=:red, linewidth=3)
# # Place a red label just below the tick
# text!(ax_perf, x10 + 1.5, 7.0, text="10", color=:red, fontsize=axis_size, align=(:center, :top))

# # Add two-way line and multiplier label at Eth=15
# eth_15 = 15
# # Find the enhancement values at Eth=15 for both lines
# idx_baseline = argmin(abs.(df_baseline.Eth .- eth_15))
# idx_single = argmin(abs.(df_single_nonlinear.Eth .- eth_15))
# enh_baseline = df_baseline.Enhancement[idx_baseline]
# enh_single = df_single_nonlinear.Enhancement[idx_single]

# # Calculate multiplier
# multiplier = round(enh_single / enh_baseline, digits=1)

# # Draw two-way line
# lines!(ax_perf, [eth_15, eth_15], [enh_single, enh_baseline],
#     color=:black, linestyle=:dash, linewidth=2.5)
# # Add multiplier label
# text!(ax_perf, 14, 300,
#     text="×$(multiplier)", color=:black, fontsize=22, align=(:center, :center))

# ## Add nonlinear equation in LaTeX Gₙₗ = ∫ |E|⁴ [1+exp(γ(|E|²-|Eₜₕ|²))]⁻¹dΩ
# text!(ax_perf, 40, 30,
#     text=L"G_{\mathrm{nl}} = \int \frac{|\mathbf{E}(\mathbf{x})|^4 α_0^2(\mathbf{x})}{1+e^{\gamma(|\mathbf{E}(\mathbf{x})|^2-E_{\mathrm{th}}^2)}} \mathrm{d} \Omega",
#     color=:black, fontsize=24, align=(:center, :center))

# xlims!(ax_perf, 10^0.5, 100)
# ylims!(ax_perf, 10^0.55, 10^4)

# # # Adjust layout - Give more height to the performance plot
# # rowsize!(fig.layout, 1, Relative(0.35)) # Top panel (geometries) takes 35% height
# # rowsize!(fig.layout, 2, Relative(0.65)) # Bottom panel (plot) takes 65% height
# # colsize!(fig.layout, 1, Relative(1.0))


# # --- Save Figure ---
# # try
# save(final_figure_filename, fig, px_per_unit=2) # Increase pixel density for PDF/vector output
# println("Saved final figure to $final_figure_filename")
# # catch e
# #     @error "Failed to save the final figure: $e"
# # end

# # --- Optional Cleanup ---
# rm(temp_path, recursive=true, force=true)
# println("Removed temporary directory: $temp_path")

# println("--- Figure Generation Script Finished ---")

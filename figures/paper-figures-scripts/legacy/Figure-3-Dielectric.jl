using Pkg;
Pkg.activate("/Users/ianhammond/GitHub/Emitter3DTopOpt"); # Adjust path if needed
include("load_data.jl")
using Revise
import Emitter3DTopOpt as e3 # No longer explicitly needed for plotting functions provided here
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

# Assume the `get_data` function from the previous notebook cell is defined and working
# Also assume `Figures.scp.yaml` is present and configured correctly.
# If `get_data` is not in scope, you need to copy its definition here.

# --- PyVista/NumPy Setup ---
try
    global pv = pyimport("pyvista")
    global np = pyimport("numpy")
    global vtk = pyimport("vtk")
    # Setup offscreen rendering if needed (assuming it was done in the notebook)
    # try
    #     pv.start_xvfb()
    # catch; @warn "Xvfb not started"; end
    # pv.global_theme.off_screen = true
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
temp_path = "temp_plots_fig4"
mkpath(temp_path) # Create directory if it doesn't exist
metal_3d_plot_raw = joinpath(temp_path, "fig4_metal_3d_raw.png")
metal_3d_plot_cropped = joinpath(temp_path, "fig4_metal_3d_cropped.png")
dielectric_3d_plot_raw = joinpath(temp_path, "fig4_dielectric_3d_raw.png")
dielectric_3d_plot_cropped = joinpath(temp_path, "fig4_dielectric_3d_cropped.png")
dielectric_design_freeform_path = joinpath(temp_path, "diel_freeform_raw.png")
dielectric_design_freeform_cropped = joinpath(temp_path, "diel_freeform_cropped.png")
dielectric_design_path = joinpath(temp_path, "diel_raw.png")
dielectric_design_cropped = joinpath(temp_path, "diel_cropped.png")
final_figure_filename = "figures/Figure-3-Dielectric_v2.pdf"
mkpath("figures") # Ensure final directory exists

# --- Define Plotting Functions (Adapted from Vis.jl / previous) ---

# Re-defined here for self-containment, using slightly modified logic for combined plots
function plot_material(np, pv, plotter, design, W, L; colorbar=true, title="\$\\rho(x)\$", font_size=20, title_font_size=24, num_periods_x=1, num_periods_y=1, ontop=false, design_field="p", contour=true, full=false, clim=(0.0, 1.0), opacity=nothing, reflectybool=true, flipx=false, flipy=false, color="#b097d1")
    pv.global_theme.allow_empty_mesh = true

    contours = design.contour(np.linspace(0, 1, 2), scalars=design_field)
    signx = (flipx) ? -1 : 1
    signy = (flipy) ? -1 : 1
    reflectx = pv.DataSetFilters.reflect(design, (1, 0, 0), point=(-W / 2 * signx, 0, 0))
    reflecty = pv.DataSetFilters.reflect(design, (0, 1, 0), point=(0, -L / 2 * signy, 0))
    reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2 * signx, 0, 0))
    contoursx = reflectx.contour(np.linspace(0, 1, 2), scalars=design_field)
    contoursy = reflecty.contour(np.linspace(0, 1, 2), scalars=design_field)
    contoursxy = reflectxy.contour(np.linspace(0, 1, 2), scalars=design_field)
    reflectx = pv.DataSetFilters.reflect(design, (1, 0, 0), point=(-W / 2 * signx, 0, 0))
    reflecty = pv.DataSetFilters.reflect(design, (0, 1, 0), point=(0, -L / 2 * signy, 0))
    reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2 * signx, 0, 0))

    # Plot all periods
    contour_color = pv.Color(color)
    scalargs = if ontop
        ["title" => title, "vertical" => false, "label_font_size" => font_size, "title_font_size" => title_font_size, "position_x" => 0.25, "position_y" => 0.8, "color" => "Black", "use_opacity" => false]
    else
        ["title" => title, "vertical" => true, "label_font_size" => font_size, "title_font_size" => title_font_size, "position_x" => 0.0, "position_y" => 0.35, "color" => "Black", "use_opacity" => false]
    end

    for i in 0:1:num_periods_x-1
        for j in 0:1:num_periods_y-1
            # Plot
            ps = Vector(0:0.01:1.0)
            opacity = (isnothing(opacity)) ? (x -> ifelse(x < 1.0, 0.0, x / 20)).(ps) : opacity
            plotter.add_mesh(design.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, opacity=opacity, show_scalar_bar=(colorbar & (i == j) & (j == 0)), scalar_bar_args=Dict(scalargs))
            # plotter.add_scalar_bar(Dict(scalargs)...)
            if !full
                plotter.add_mesh(reflectx.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                if reflectybool
                    plotter.add_mesh(reflecty.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                    plotter.add_mesh(reflectxy.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                end
            end
        end
    end

    return plotter
end

function plot_geometry_for_inset(mesh_pyobject, output_filename;
    scalar_field="p", # Still relevant for contour calculation
    window_size=(400, 400),
    zoom=1.2, # Adjust zoom for single cell
    group_name=nothing) # Add group_name parameter to determine color
    println("Generating geometry plot for inset using plot_material: $output_filename")

    # Ensure background is not transparent for reliable cropping later
    pv.global_theme.transparent_background = true
    plotter = pv.Plotter(off_screen=true, window_size=window_size)

    # Calculate dimensions W, L from the mesh bounds
    # Assumes the input mesh represents exactly one primitive cell centered appropriately
    # or that the bounds correctly reflect the L/2, W/2 dimensions for reflection logic *inside* plot_material if full=false.
    # Since we will set full=true, reflection logic inside plot_material is bypassed,
    # but W/L are still needed for the translate call within its loop (even if loop runs once).
    if !hasproperty(mesh_pyobject, :bounds)
        error("Input mesh object does not have 'bounds' property.")
    end
    xl, xr, yl, yr, zl, zr = mesh_pyobject.bounds
    des_high = zr
    # We need the *full* cell width/length if plot_material uses W/2, L/2 internally for reflection points
    # Let's assume xl, yl are -W/2, -L/2 and xr, yr are W/2, L/2
    W = xr - xl
    L = yr - yl
    println("  Calculated W=$W, L=$L from bounds.")
    if W <= 0 || L <= 0
        error("Calculated width or length is non-positive. Check mesh bounds: $mesh_pyobject.bounds")
    end

    # Determine color based on group name
    color = if group_name == "Freeform_Metal"
        "#b097d1" # Purple for 3D Metal
    elseif group_name == "Constrained_Metal"
        "#496cb7" # Blue for 2D Metal
    elseif group_name == "Freeform_Dielectric"
        "#ff9900" # Orange for 3D Dielectric
    elseif group_name == "Constrained_Dielectric"
        "#d62728" # Red for 2D Dielectric
    elseif group_name == "Spheres"
        "#008000" # Green for Metal Spheres
    else
        "#b097d1" # Default to purple if unknown
    end

    # Call the user-provided plot_material function with inset-specific settings
    plot_material(
        np, pv, plotter, mesh_pyobject, W, L;
        colorbar=false,       # No colorbar for inset
        num_periods_x=2,      # Only one cell
        num_periods_y=2,      # Only one cell
        contour=true,         # Keep contour shell
        design_field=scalar_field, # Field for contour calculation
        color=color,           # Pass the determined color
        flipy=true,
    )

    # Remove plotter axes decorations
    # plotter.remove_bounds_axes()

    plotter.add_axes(
        line_width=5,
        cone_radius=0.4,
        shaft_length=0.8,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.6, 0.2),
        viewport=(0.60, 0.60, 0.93, 0.93),
    )

    # Save the plot
    plotter.screenshot(output_filename)
    plotter.clear()
    plotter.close()
    println("Saved geometry plot to $output_filename")
end

function plot_field_paper(np, pv, plotter, field_mesh, W, L;
    scalar_bar_title="||E||\$^4\$", # Changed default
    font_size=28, title_font_size=28,
    field_lim=(0.0, 5.0), # Important: Set per material!
    field_cmap="coolwarm", # Or "inferno", "viridis" etc.
    field_opacity="sigmoid", # Can be float, array, "sigmoid", "geom", etc.
    num_periods_x=2, num_periods_y=2,
    full=false, flipx=false, flipy=false,
    colorbar=true, vertical_colorbar=false, # Adjust colorbar position
    color=nothing
)
    pv.global_theme.transparent_background = false
    # Simplified: Assumes field_mesh has "uhNorm", calculates E2, plots it
    # uhNorm = nothing
    # try
    #     field_data = field_mesh.point_data["uhNorm"]
    #     # Calculate |E|^2 = Ex^2 + Ey^2 + Ez^2 (assuming uhNorm has 3 components)
    #     uhNorm = np.square(field_data[:,0]) + np.square(field_data[:,1]) + np.square(field_data[:,2])
    #     field_mesh.point_data.set_scalars(uhNorm, "E2") # Add/overwrite "E2" scalars
    # catch e
    #     @error "Could not find 'uhNorm' in point data or calculate |E|^2." e
    #     return plotter # Return plotter without field if error
    # end

    signx = (flipx) ? -1 : 1
    signy = (flipy) ? -1 : 1

    # Reflect field mesh if needed
    reflectx = !full ? pv.DataSetFilters.reflect(field_mesh, (1, 0, 0), point=(-W / 2 * signx, 0, 0)) : nothing
    reflecty = !full ? pv.DataSetFilters.reflect(field_mesh, (0, 1, 0), point=(0, -L / 2 * signy, 0)) : nothing
    reflectxy = !full ? pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2 * signx, 0, 0)) : nothing

    # Colorbar Arguments
    annot = Dict(
        field_lim[1] => "",         # bottom of the bar
        field_lim[2] => ""        # top of the bar
    )
    scalar_bar_args = Dict(
        "title" => scalar_bar_title,
        "vertical" => vertical_colorbar,
        "label_font_size" => font_size * 2,
        "n_labels" => 0,              # ← no auto numeric labels
        "title_font_size" => title_font_size * 3,
        "color" => "black", # Color bar text color
        "use_opacity" => false # Usually false for colorbar text
    )
    # Adjust position based on vertical/horizontal
    if vertical_colorbar
        merge!(scalar_bar_args, Dict("position_x" => 0.05, "position_y" => 0.25))
    else
        merge!(scalar_bar_args, Dict("position_x" => 0.25, "position_y" => 0.05)) # Position below horizontal bar
    end

    for i in 0:1:num_periods_x-1
        for j in 0:1:num_periods_y-1
            show_cb = colorbar && (i == 0 && j == 0) # Show colorbar only once

            # Original quarter field plot
            plotter.add_mesh(field_mesh.translate((i * W * 2, j * L * 2, 0), inplace=false),
                scalars="E4", cmap=field_cmap, clim=field_lim,
                opacity=field_opacity, show_scalar_bar=show_cb,
                annotations=annot,
                scalar_bar_args=scalar_bar_args,
                ambient=0.7, diffuse=0.3, specular=0.25, specular_power=12)

            # # ——— right here, after creating the bar actor ———
            # if show_cb
            #     # get the actor PyVista just added
            #     sb_actor = plotter.scalar_bars[end]
            #     # move ticks & annotations to the 'succeed' side
            #     sb_actor.SetTextPositionToSucceedScalarBar()  # right side for vertical bars :contentReference[oaicite:0]{index=0}
            # end

            # Reflected fields if needed
            if !full
                if !isnothing(reflectx)
                    plotter.add_mesh(reflectx.translate((i * W * 2, j * L * 2, 0), inplace=false),
                        scalars="E4", cmap=field_cmap, clim=field_lim,
                        opacity=field_opacity, show_scalar_bar=false,
                        ambient=0.7, diffuse=0.3, specular=0.25, specular_power=12)
                end
                if !isnothing(reflecty)
                    plotter.add_mesh(reflecty.translate((i * W * 2, j * L * 2, 0), inplace=false),
                        scalars="E4", cmap=field_cmap, clim=field_lim,
                        opacity=field_opacity, show_scalar_bar=false,
                        ambient=0.7, diffuse=0.3, specular=0.25, specular_power=12)
                end
                if !isnothing(reflectxy)
                    plotter.add_mesh(reflectxy.translate((i * W * 2, j * L * 2, 0), inplace=false),
                        scalars="E4", cmap=field_cmap, clim=field_lim,
                        opacity=field_opacity, show_scalar_bar=false,
                        ambient=0.7, diffuse=0.3, specular=0.25, specular_power=12)
                end
            end
        end
    end
    return plotter
end

function hex_to_rgba(s::AbstractString)
    h = replace(s, r"^#" => "")
    if length(h) in (3, 4)
        h = join([repeat(string(c), 2) for c in h])
    elseif length(h) in (6, 8)
        # ok
    else
        throw(ArgumentError("Invalid hex color: $s"))
    end
    r = parse(Int, h[1:2], base=16)
    g = parse(Int, h[3:4], base=16)
    b = parse(Int, h[5:6], base=16)
    a = length(h) == 8 ? parse(Int, h[7:8], base=16) : 255
    return (r / 255, g / 255, b / 255, a / 255)
end

function plot_field_with_cross_sections(np, pv, field_mesh, design_mesh, W, L, hd;
    plotter_kwargs=Dict{String,Any}(),
    main_plot_kwargs=Dict{String,Any}(),
    slice_y_coord=nothing,
    slice_x_coord=nothing, # Placeholder for future X-slice
    slice_z_coord=nothing, # Placeholder for future Z-slice
    slice_plane_opacity=0.3,
    slice_plot_kwargs=Dict{String,Any}(),
    link_cameras=false,
    color=nothing
)
    pv.global_theme.transparent_background = false

    # --- Determine Plot Layout ---
    num_slices = (isnothing(slice_y_coord) ? 0 : 1) + (isnothing(slice_x_coord) ? 0 : 1)
    if num_slices == 0
        plot_shape = (1, 1) # Only the main plot
        main_plot_row, main_plot_col = 0, 0
        println("Warning: No slice coordinates provided. Showing only the main plot.")
    elseif num_slices == 1
        plot_shape = (2, 1) # Slice above main plot
        slice_row, slice_col = 0, 0
        main_plot_row, main_plot_col = 1, 0
    else # num_slices == 2
        plot_shape = (2, 3) # 2 rows, 2 columns
        x_slice_rc = (0, 0)
        y_slice_rc = (0, 1)
        main_plot_rc_span = ((1, 0), (1, 1)) # Main plot spans row 1, cols 0 and 1
        main_plot_row, main_plot_col = 1, 0 # Start at (1,0)
    end

    # Update plotter_kwargs with the calculated shape and remove borders
    plotter_kwargs_final = merge(Dict("shape" => "3/1", "border" => false), plotter_kwargs)
    plotter_kwargs_final = Dict(Symbol(k) => v for (k, v) in plotter_kwargs_final)

    # --- Create Plotter ---
    plotter = pv.Plotter(; off_screen=true, window_size=[1800 * 10, 1800 * 10], plotter_kwargs_final...)
    plotter.set_background("white")
    plotter.padding = 0
    plotter.margin = 0

    # --- Prepare Common Plotting Args ---
    common_cmap = main_plot_kwargs["field_cmap"]
    common_clim = main_plot_kwargs["field_lim"]
    common_scalars = "E4"

    pybuiltins = pyimport("builtins")
    contour_color = pv.Color(color)

    # --- Plot X-Normal Slice (if requested) ---
    if !isnothing(slice_x_coord)
        plotter.subplot(3)
        origin_x_slice = (0.0, 2 * W, 0)#(slice_x_coord, 0, 0)
        slice_mesh_x = field_mesh.slice(normal=(1, 0, 0), origin=origin_x_slice)
        slice_mesh_x_translated = slice_mesh_x.translate((0, W, 0), inplace=false)
        slice_design_x = design_mesh.slice(normal=(1, 0, 0), origin=origin_x_slice)
        if slice_mesh_x.n_points > 0
            slice_args = merge(Dict(
                    "scalars" => common_scalars,
                    "cmap" => "hot",
                    "clim" => common_clim .* 0.1,
                    "opacity" => "sigmoid",
                    "show_scalar_bar" => false
                ), slice_plot_kwargs)
            merge!(slice_args, Dict(
                "ambient" => 0.7,
                "diffuse" => 0.3,
                "specular" => 0.25,
                "specular_power" => 12,
            ))
            slice_args = Dict(Symbol(k) => v for (k, v) in slice_args)
            # Tile and reflect for 2 periods in Y
            for j in 0:1
                for reflect in (false, true)
                    mesh = slice_mesh_x
                    if reflect
                        mesh = pv.DataSetFilters.reflect(mesh, (0, 1, 0), point=(0, L / 2, 0))
                    end
                    mesh = mesh.translate((0, j * L * 2, 0), inplace=false)
                    plotter.add_mesh(mesh; slice_args...)
                    geom = slice_design_x
                    if reflect
                        geom = pv.DataSetFilters.reflect(geom, (0, 1, 0), point=(0, L / 2, 0))
                    end
                    geom = geom.translate((0, j * L * 2, 0), inplace=false)
                    plotter.add_mesh(geom; show_scalar_bar=false, color="#808080")
                    # Add substrate for each slice
                end
            end
            e3.plot_substrate(np, pv, plotter, slice_mesh_x, W, L, zl; full=true, flipx=false, flipy=true, num_periods_x=1, num_periods_y=2)
            e3.plot_substrate(np, pv, plotter, slice_mesh_x_translated, W, L, zl; full=true, flipx=false, flipy=true, num_periods_x=1, num_periods_y=2)
            # plotter.view_yz()
            plotter.set_background("lightgreen")
            plotter.camera.position = plotter.camera.position .- (0, 0, 4 * hd)
            plotter.camera.focal_point = plotter.camera.focal_point .- (0, 0, 4 * hd)
            plotter.camera.zoom(1.0)
            # plotter.remove_bounds_axes()
        else
            @warn "X-normal slice at x=$slice_x_coord resulted in an empty mesh. Check coordinate."
        end
    end

    # --- Plot Y-Normal Slice (if requested) ---
    if !isnothing(slice_y_coord)
        plotter.subplot(1)
        origin_y_slice = (W, W / 4, -hd)#(0, slice_y_coord, 0)
        slice_mesh_y = field_mesh.slice(normal=(0, 1, 0), origin=origin_y_slice)
        slice_mesh_y_translated = slice_mesh_y.translate((-W, 0, 0), inplace=false)
        slice_design_y = design_mesh.slice(normal=(0, 1, 0), origin=origin_y_slice)
        if slice_mesh_y.n_points > 0
            slice_args = merge(Dict(
                    "scalars" => common_scalars,
                    "cmap" => "hot",
                    "clim" => common_clim .* 0.1,
                    "opacity" => "sigmoid",
                    "show_scalar_bar" => false
                ), slice_plot_kwargs)
            merge!(slice_args, Dict(
                "ambient" => 0.7,
                "diffuse" => 0.3,
                "specular" => 0.25,
                "specular_power" => 12,
            ))
            slice_args = Dict(Symbol(k) => v for (k, v) in slice_args)
            # Tile and reflect for 2 periods in X
            for i in 0:1
                for reflect in (false, true)
                    mesh = slice_mesh_y
                    if reflect
                        mesh = pv.DataSetFilters.reflect(mesh, (1, 0, 0), point=(-W / 2, 0, 0))
                    end
                    mesh = mesh.translate((i * W * 2, 0, 0), inplace=false)
                    plotter.add_mesh(mesh; slice_args...)
                    geom = slice_design_y
                    if reflect
                        geom = pv.DataSetFilters.reflect(geom, (1, 0, 0), point=(-W / 2, 0, 0))
                    end
                    geom = geom.translate((i * W * 2, 0, 0), inplace=false)
                    plotter.add_mesh(geom; show_scalar_bar=false, color="#808080")
                    # Add substrate for each slice
                    # e3.plot_substrate(np, pv, plotter, mesh, W, L, zl; full=false, flipx=false, flipy=true, num_periods_x=1, num_periods_y=1)
                end
            end
            e3.plot_substrate(np, pv, plotter, slice_mesh_y, W, L, zl; full=true, flipx=false, flipy=true, num_periods_x=2, num_periods_y=1)
            e3.plot_substrate(np, pv, plotter, slice_mesh_y_translated, W, L, zl; full=true, flipx=false, flipy=true, num_periods_x=2, num_periods_y=1)
            # plotter.view_xz()
            # plotter.set_background("lightblue")
            plotter.camera.position = plotter.camera.position .- (-W, 0, 4 * hd)
            plotter.camera.focal_point = plotter.camera.focal_point .- (-W, 0, 4 * hd)
            plotter.camera.zoom(1.0)
            # plotter.remove_bounds_axes()
        else
            @warn "Y-normal slice at y=$slice_y_coord resulted in an empty mesh. Check coordinate."
        end
    end


    # --- Plot Z-Normal Slice (if requested) ---
    if !isnothing(slice_z_coord)
        plotter.subplot(2) # Assuming this is the correct subplot index for Z
        origin_z_slice = (0, 0, -280)
        slice_mesh_z = field_mesh.slice(normal=(0, 0, 1), origin=origin_z_slice)
        slice_design_z = design_mesh.slice(normal=(0, 0, 1), origin=origin_z_slice)
        if slice_mesh_z.n_points > 0
            slice_args = merge(Dict(
                    "scalars" => common_scalars,
                    "cmap" => "hot",
                    "clim" => common_clim .* 0.1,
                    "opacity" => "sigmoid",
                    "show_scalar_bar" => false
                ), slice_plot_kwargs)
            merge!(slice_args, Dict(
                "ambient" => 0.7,
                "diffuse" => 0.3,
                "specular" => 0.25,
                "specular_power" => 12,
            ))
            slice_args = Dict(Symbol(k) => v for (k, v) in slice_args)
            # Tile and reflect for 2 periods in X, with Y offset for visualization
            for i in 0:1  # X direction
                for j in 0:1  # Y direction  
                    for reflect_x in (false, true)
                        for reflect_y in (false, true)
                            mesh = slice_mesh_z
                            geom = slice_design_z

                            # Apply X reflection if needed
                            if reflect_x
                                mesh = pv.DataSetFilters.reflect(mesh, (1, 0, 0), point=(-W / 2, 0, 0))
                                geom = pv.DataSetFilters.reflect(geom, (1, 0, 0), point=(-W / 2, 0, 0))
                            end

                            # Apply Y reflection if needed
                            if reflect_y
                                mesh = pv.DataSetFilters.reflect(mesh, (0, 1, 0), point=(0, -L / 2, 0))
                                geom = pv.DataSetFilters.reflect(geom, (0, 1, 0), point=(0, -L / 2, 0))
                            end

                            # Apply translation for tiling with Y offset
                            y_offset = ifelse(j == 1 && reflect_y == false, -j * L * 2, j * L * 2)  # Shift Y visualization by half period
                            mesh = mesh.translate((i * W * 2, y_offset, 0), inplace=false)
                            geom = geom.translate((i * W * 2, y_offset, 0), inplace=false)

                            # Add meshes to plotter
                            plotter.add_mesh(mesh; slice_args...)
                            plotter.add_mesh(geom; show_scalar_bar=false, color="#808080")
                        end
                    end
                end
            end
            e3.plot_substrate(np, pv, plotter, slice_mesh_z, W, L, zl; full=false, flipx=false, flipy=false, num_periods_x=2, num_periods_y=2)
            # plotter.view_xy()  # Top-down view for Z-normal slice
            plotter.camera.position = plotter.camera.position .+ (0, 0, 2 * hd)
            plotter.camera.focal_point = plotter.camera.focal_point .+ (0, 0, 2 * hd)
            plotter.camera.zoom(1.0)
        else
            @warn "Z-normal slice at z=$slice_z_coord resulted in an empty mesh. Check coordinate."
        end
    end

    # --- Plot Main Field (Bottom Subplot) ---
    plotter.subplot(0)
    main_args = merge(Dict(
            "colorbar" => true,
            "vertical_colorbar" => false,
            "scalar_bar_title" => "",  # Remove title from scalar bar
            # "annotations" => Dict()    # Remove annotations (0 and M)
        ), main_plot_kwargs)
    main_args = Dict(Symbol(k) => v for (k, v) in main_args)
    plotter = plot_field_paper(np, pv, plotter, field_mesh, W, L; main_args...)
    plot_material(
        np, pv, plotter, design_mesh, W, L;
        colorbar=false,
        num_periods_x=2,
        num_periods_y=2,
        flipx=false,
        flipy=true,
        design_field="p",
        color="#808080"#color
    )
    e3.plot_substrate(np, pv, plotter, field_mesh, W, L, zl; full=false, flipx=false, flipy=true, num_periods_x=2, num_periods_y=2)
    # plotter.remove_bounds_axes()
    # plotter.add_scalar_bar(Dict(
    #     "title" => scalar_bar_title,
    #     "vertical" => false,
    #     "label_font_size" => font_size*2,
    #     "n_labels" => 0, # No auto numeric labels
    #     "title_font_size" => title_font_size*3,
    #     "color" => "black", # Color bar text color
    #     "use_opacity" => false, # Usually false for colorbar text
    #     "position_x" => 0.25,
    #     "position_y" => 0.05
    # )...)

    # --- Add Visual Cues (Planes) ---
    if !isnothing(slice_y_coord) && num_slices > 0
        num_periods_x = get(main_plot_kwargs, :num_periods_x, 1)
        num_periods_y = get(main_plot_kwargs, :num_periods_y, 1)
        full = get(main_plot_kwargs, :full, false)
        x_multiplier = full ? 1 : 2
        total_width = num_periods_x * W * x_multiplier * 2.0
        mesh_bounds = field_mesh.bounds
        z_depth = mesh_bounds[6] - mesh_bounds[5]
        # Y-normal plane
        plane_y = pv.Plane(center=(W * 1.0, 0, -hd - hd * 0.25), direction=(0, 1, 0), i_size=z_depth * 0.75, j_size=total_width, i_resolution=max(1, num_periods_x * 5), j_resolution=max(1, Int(round(num_periods_y / 2))))
        plotter.add_mesh(plane_y, color="#b097d1", opacity=1.0, pickable=false)
    end
    if !isnothing(slice_x_coord) && num_slices > 0
        num_periods_x = get(main_plot_kwargs, :num_periods_x, 1)
        num_periods_y = get(main_plot_kwargs, :num_periods_y, 1)
        full = get(main_plot_kwargs, :full, false)
        y_multiplier = full ? 1 : 2
        total_height = num_periods_x * W * x_multiplier * 2.0 #num_periods_y * L * y_multiplier
        mesh_bounds = field_mesh.bounds
        z_depth = mesh_bounds[6] - mesh_bounds[5]
        # X-normal plane
        plane_x = pv.Plane(center=(-W * 0.75, W * 1.75, -hd - hd * 0.25), direction=(1, 0, 0), i_size=z_depth * 0.75, j_size=total_height, i_resolution=max(1, num_periods_y * 5), j_resolution=max(1, Int(round(num_periods_x / 2))))
        plotter.add_mesh(plane_x, color="#b097d1", opacity=1.0, pickable=false)
    end
    if !isnothing(slice_z_coord) && num_slices > 0
        num_periods_x = get(main_plot_kwargs, :num_periods_x, 1)
        num_periods_y = get(main_plot_kwargs, :num_periods_y, 1)
        full = get(main_plot_kwargs, :full, false)
        y_multiplier = full ? 1 : 2
        total_height = num_periods_x * W * x_multiplier * 2.25#num_periods_y * L * y_multiplier
        mesh_bounds = field_mesh.bounds
        z_depth = mesh_bounds[6] - mesh_bounds[5]
        # X-normal plane
        plane_z = pv.Plane(center=(W * 0.5, W * 1.25, -280), direction=(0, 0, 1), i_size=total_height, j_size=total_height, i_resolution=max(1, num_periods_y * 5), j_resolution=max(1, num_periods_y * 5))
        plotter.add_mesh(plane_z, color="#b097d1", opacity=1.0, pickable=false)
    end


    plotter.background_color = "white"
    # if link_cameras
    #     plotter.link_views()
    # end
    return plotter
end

function plot_field_with_cross_sections_dielectric(np, pv, field_mesh, design_mesh, W, L, hd;
    plotter_kwargs=Dict{String,Any}(),
    main_plot_kwargs=Dict{String,Any}(),
    slice_y_coord=nothing,
    slice_x_coord=nothing,
    slice_plane_opacity=1.0,
    slice_plot_kwargs=Dict{String,Any}(),
    link_cameras=false,
    color=nothing
)
    global zl
    pv.global_theme.transparent_background = false

    # --- Determine Plot Layout ---
    num_slices = (isnothing(slice_y_coord) ? 0 : 1) + (isnothing(slice_x_coord) ? 0 : 1)
    if num_slices == 0
        plot_shape = (1, 1)
        main_plot_row, main_plot_col = 0, 0
        println("Warning: No slice coordinates provided. Showing only the main plot.")
    elseif num_slices == 1
        plot_shape = (2, 1)
        slice_row, slice_col = 0, 0
        main_plot_row, main_plot_col = 1, 0
    else
        plot_shape = (2, 2)
        x_slice_rc = (0, 0)
        y_slice_rc = (0, 1)
        main_plot_rc_span = ((1, 0), (1, 1))
        main_plot_row, main_plot_col = 1, 0
    end

    # Update plotter_kwargs with the calculated shape and remove borders
    plotter_kwargs_final = merge(Dict("shape" => "3/1", "border" => false, "window_size" => (1800 * 5, 1800 * 5)), plotter_kwargs)
    plotter_kwargs_final = Dict(Symbol(k) => v for (k, v) in plotter_kwargs_final)

    # --- Create Plotter ---
    plotter = pv.Plotter(; off_screen=true, plotter_kwargs_final...)
    plotter.set_background("white")
    plotter.padding = 0
    plotter.margin = 0

    # --- Prepare Common Plotting Args ---
    common_cmap = main_plot_kwargs["field_cmap"]
    common_clim = main_plot_kwargs["field_lim"]
    common_scalars = "E4"

    pybuiltins = pyimport("builtins")
    contour_color = pv.Color("808080")

    # --- Plot X-Normal Slice (if requested) ---
    if !isnothing(slice_x_coord)
        plotter.subplot(3)
        origin_x_slice = (W / 2, 2 * W - W / 8, 0)
        slice_mesh_x = field_mesh.slice(normal=(1, 0, 0), origin=origin_x_slice)
        slice_mesh_x_translated = slice_mesh_x.translate((0, W, 0), inplace=false)
        slice_design_x = design_mesh.slice(normal=(1, 0, 0), origin=origin_x_slice)
        if slice_mesh_x.n_points > 0
            slice_args = merge(Dict(
                    "scalars" => common_scalars,
                    "cmap" => "hot",
                    "clim" => common_clim,
                    "opacity" => "sigmoid",
                    "show_scalar_bar" => false
                ), slice_plot_kwargs)
            merge!(slice_args, Dict(
                "ambient" => 0.9,
                "diffuse" => 0.1,
                "specular" => 0.35,
                "specular_power" => 20,
            ))
            slice_args = Dict(Symbol(k) => v for (k, v) in slice_args)
            # Tile and reflect for 2 periods in Y
            for j in -1:0
                for reflect in (false, true)
                    mesh = slice_mesh_x
                    if reflect
                        mesh = pv.DataSetFilters.reflect(mesh, (0, 1, 0), point=(0, L / 2, 0))
                    end
                    jL = ifelse(j == -1 && !reflect, j * L * 2 + L + 4 * L, j * L * 2 + L)
                    mesh = mesh.translate((0, jL, 0), inplace=false)
                    plotter.add_mesh(mesh; slice_args...)
                    geom = slice_design_x
                    if reflect
                        geom = pv.DataSetFilters.reflect(geom, (0, 1, 0), point=(0, L / 2, 0))
                    end
                    geom = geom.translate((0, jL, 0), inplace=false)
                    plotter.add_mesh(geom; show_scalar_bar=false, color=contour_color)
                end
            end
            e3.plot_substrate(np, pv, plotter, slice_mesh_x, W, L, zl; full=true, flipx=true, flipy=false, num_periods_x=1, num_periods_y=2)
            e3.plot_substrate(np, pv, plotter, slice_mesh_x_translated, W, L, zl; full=true, flipx=true, flipy=false, num_periods_x=1, num_periods_y=2)
            plotter.set_background("lightgreen")
            plotter.camera.position = plotter.camera.position .- (0, 0, 2 * hd)
            plotter.camera.focal_point = plotter.camera.focal_point .- (0, 0, 2 * hd)
            plotter.camera.zoom(1.0)
        else
            @warn "X-normal slice at x=$slice_x_coord resulted in an empty mesh. Check coordinate."
        end
    end

    # --- Plot Y-Normal Slice (if requested) ---
    if !isnothing(slice_y_coord)
        plotter.subplot(1)
        # origin_y_slice = (W, W/4, -hd)
        origin_y_slice = (W, W / 2, -hd)
        slice_mesh_y = field_mesh.slice(normal=(0, 1, 0), origin=origin_y_slice)
        slice_mesh_y_translated = slice_mesh_y.translate((-W, 0, 0), inplace=false)
        slice_design_y = design_mesh.slice(normal=(0, 1, 0), origin=origin_y_slice)
        if slice_mesh_y.n_points > 0
            slice_args = merge(Dict(
                    "scalars" => common_scalars,
                    "cmap" => "hot",
                    "clim" => common_clim,
                    "opacity" => "sigmoid",
                    "show_scalar_bar" => false
                ), slice_plot_kwargs)
            merge!(slice_args, Dict(
                "ambient" => 0.9,
                "diffuse" => 0.1,
                "specular" => 0.35,
                "specular_power" => 20,
            ))
            slice_args = Dict(Symbol(k) => v for (k, v) in slice_args)
            # Tile and reflect for 2 periods in X
            for i in 0:1
                for reflect in (false, true)
                    mesh = slice_mesh_y
                    if reflect
                        mesh = pv.DataSetFilters.reflect(mesh, (1, 0, 0), point=(-W / 2, 0, 0))
                    end
                    mesh = mesh.translate((i * W * 2, 0, 0), inplace=false)
                    plotter.add_mesh(mesh; slice_args...)
                    geom = slice_design_y
                    if reflect
                        geom = pv.DataSetFilters.reflect(geom, (1, 0, 0), point=(-W / 2, 0, 0))
                    end
                    geom = geom.translate((i * W * 2, 0, 0), inplace=false)
                    plotter.add_mesh(geom; show_scalar_bar=false, color=contour_color)
                end
            end
            e3.plot_substrate(np, pv, plotter, slice_mesh_y, W, L, zl; full=true, flipx=false, flipy=true, num_periods_x=2, num_periods_y=1)
            e3.plot_substrate(np, pv, plotter, slice_mesh_y_translated, W, L, zl; full=true, flipx=false, flipy=true, num_periods_x=2, num_periods_y=1)
            plotter.camera.position = plotter.camera.position .- (0, 0, 2 * hd)
            plotter.camera.focal_point = plotter.camera.focal_point .- (0, 0, 2 * hd)
            plotter.camera.zoom(1.0)
        else
            @warn "Y-normal slice at y=$slice_y_coord resulted in an empty mesh. Check coordinate."
        end
    end

    # --- Plot Z-Normal Slice (if requested) --- ADDED SECTION
    if !isnothing(0)
        plotter.subplot(2) # Match plot_field_with_cross_sections
        origin_z_slice = (0, 0, -200) # Match plot_field_with_cross_sections
        slice_mesh_z = field_mesh.slice(normal=(0, 0, 1), origin=origin_z_slice)
        slice_design_z = design_mesh.slice(normal=(0, 0, 1), origin=origin_z_slice)
        if slice_mesh_z.n_points > 0
            slice_args = merge(Dict(
                    "scalars" => common_scalars,
                    "cmap" => "hot", # Match plot_field_with_cross_sections
                    "clim" => common_clim, # MODIFIED: Scale clim
                    "opacity" => "sigmoid", # MODIFIED: Match plot_field_with_cross_sections
                    "show_scalar_bar" => false
                ), slice_plot_kwargs)
            merge!(slice_args, Dict(
                "ambient" => 0.9,
                "diffuse" => 0.1,
                "specular" => 0.35,
                "specular_power" => 20,
            ))
            slice_args = Dict(Symbol(k) => v for (k, v) in slice_args)
            # Tile and reflect for 2x2 periods (X and Y)

            for i in 0:1  # X direction tiling
                for j in 0:1  # Y direction tiling
                    for reflect_x in (false, true)
                        for reflect_y in (false, true)
                            mesh = slice_mesh_z
                            geom = slice_design_z

                            # Apply X reflection
                            if reflect_x
                                mesh = pv.DataSetFilters.reflect(mesh, (1, 0, 0), point=(-W / 2, 0, 0))
                                geom = pv.DataSetFilters.reflect(geom, (1, 0, 0), point=(-W / 2, 0, 0))
                            end

                            # Apply Y reflection
                            if reflect_y
                                mesh = pv.DataSetFilters.reflect(mesh, (0, 1, 0), point=(0, -L / 2, 0))
                                geom = pv.DataSetFilters.reflect(geom, (0, 1, 0), point=(0, -L / 2, 0))
                            end

                            # Apply translation for tiling
                            current_x_translation = i * W * 2
                            current_y_translation = j * L * 2
                            if j == 1 && !reflect_y
                                current_y_translation = -j * L * 2 # Shift Y visualization by half period
                            end

                            mesh_translated = mesh.translate((current_x_translation, current_y_translation, 0), inplace=false)
                            geom_translated = geom.translate((current_x_translation, current_y_translation, 0), inplace=false)

                            plotter.add_mesh(mesh_translated; slice_args...)
                            plotter.add_mesh(geom_translated; show_scalar_bar=false, color="#808080") # MODIFIED color
                        end
                    end
                end
            end
            _, _, _, _, zl, _ = field_mesh.bounds
            # e3.plot_substrate(np, pv, plotter, slice_mesh_z.translate((0, y_visual_offset,0), inplace=false), W, L, zl; full=false, flipx=false, flipy=false, num_periods_x=2, num_periods_y=2)

            plotter.camera.position = plotter.camera.position .+ (L / 2, L / 2, 2 * hd) # Match plot_field_with_cross_sections
            plotter.camera.focal_point = plotter.camera.focal_point .+ (L / 2, L / 2, 2 * hd) # Match plot_field_with_cross_sections
            plotter.camera.zoom(1.0) # Match plot_field_with_cross_sections
        else
            @warn "Z-normal slice at z=$slice_z_coord resulted in an empty mesh. Check coordinate."
        end
    end

    # --- Plot Main Field (Bottom Subplot) ---
    plotter.subplot(0)
    main_args = merge(Dict(
            "colorbar" => true,
            "vertical_colorbar" => false,
            "scalar_bar_title" => "",
        ), main_plot_kwargs)
    main_args = Dict(Symbol(k) => v for (k, v) in main_args)
    plotter = plot_field_paper(np, pv, plotter, field_mesh, W, L; main_args...)

    plot_material(
        np, pv, plotter, design_mesh, W, L;
        colorbar=false,
        num_periods_x=2,
        num_periods_y=2,
        flipx=false,
        flipy=true,
        design_field="p",
        color=color
    )
    # e3.plot_substrate(np, pv, plotter, field_mesh, W, L, zl; full=false, flipx=false, flipy=true, num_periods_x=2, num_periods_y=2)

    # --- Add Visual Cues (Planes) ---
    if !isnothing(slice_y_coord) && num_slices > 0
        num_periods_x = get(main_plot_kwargs, :num_periods_x, 1)
        num_periods_y = get(main_plot_kwargs, :num_periods_y, 1)
        full = get(main_plot_kwargs, :full, false)
        x_multiplier = full ? 1 : 2
        total_width = num_periods_x * W * x_multiplier
        mesh_bounds = field_mesh.bounds
        z_depth = mesh_bounds[6] - mesh_bounds[5]
        # Y-normal plane
        plane_y = pv.Plane(center=(W / 2, W / 2 + W / 4, -hd / 2), direction=(0, 1, 0), i_size=total_width * 2, j_size=z_depth * 1.25, i_resolution=max(1, num_periods_x * 5), j_resolution=max(1, Int(round(num_periods_y / 2))))
        plotter.add_mesh(plane_y, color="#d62728", opacity=slice_plane_opacity, pickable=false)
    end
    if !isnothing(slice_x_coord) && num_slices > 0
        num_periods_x = get(main_plot_kwargs, :num_periods_x, 1)
        num_periods_y = get(main_plot_kwargs, :num_periods_y, 1)
        full = get(main_plot_kwargs, :full, false)
        y_multiplier = full ? 1 : 2
        total_height = num_periods_x * W * x_multiplier * 7 / 8
        mesh_bounds = field_mesh.bounds
        z_depth = mesh_bounds[6] - mesh_bounds[5]
        # X-normal plane
        plane_x = pv.Plane(center=(-W / 2 + W / 4, 1.5 * W, -hd / 2), direction=(1, 0, 0), i_size=total_height * 2, j_size=z_depth * 1.25, i_resolution=max(1, num_periods_y * 5), j_resolution=max(1, Int(round(num_periods_x / 2))))
        plotter.add_mesh(plane_x, color="#d62728", opacity=slice_plane_opacity, pickable=false)
    end
    if !isnothing(0)
        plane_z_center = (W * 0.5, W * 1.5, -200)
        plane_z_isize = get(main_plot_kwargs, :num_periods_x, 2) * W * (get(main_plot_kwargs, :full, false) ? 1 : 2) * 1.25
        plane_z_jsize = get(main_plot_kwargs, :num_periods_y, 2) * L * (get(main_plot_kwargs, :full, false) ? 1 : 2) * 1.25

        plane_z = pv.Plane(center=plane_z_center, direction=(0, 0, 1),
            i_size=plane_z_isize,
            j_size=plane_z_jsize,
            i_resolution=max(1, get(main_plot_kwargs, :num_periods_x, 2) * 5),
            j_resolution=max(1, get(main_plot_kwargs, :num_periods_y, 2) * 5))
        plotter.add_mesh(plane_z, color="#d62728", opacity=1.0, pickable=false)
    end

    plotter.background_color = "white"
    return plotter
end

# Cropping Function (reuse from previous)
function crop_white_margins(img::AbstractArray; threshold::Real=1.0)
    chans = channelview(img)
    white_mask = dropdims(all(chans .>= threshold, dims=1); dims=1)
    nonwhite_mask = .!white_mask
    rows = Base.vec(any(nonwhite_mask, dims=2))
    cols = Base.vec(any(nonwhite_mask, dims=1))
    if !any(rows) || !any(cols)
        @warn "Image seems to be all white, not cropping."
        return img
    end
    rmin, rmax = findfirst(rows), findlast(rows)
    cmin, cmax = findfirst(cols), findlast(cols)
    return img[rmin:rmax, cmin:cmax]
end

function padleft(img::AbstractMatrix{RGB{N0f8}}, n::Integer)
    h, _ = size(img)
    padcol = fill(RGB{N0f8}(1, 1, 1), h, n)
    return hcat(padcol, img)
end
function padright(img::AbstractMatrix{RGB{N0f8}}, n::Integer)
    h, _ = size(img)
    padcol = fill(RGB{N0f8}(1, 1, 1), h, n)
    return hcat(img, padcol)
end
function padtop(img::AbstractMatrix{RGB{N0f8}}, n::Integer)
    _, w = size(img)
    padrow = fill(RGB{N0f8}(1, 1, 1), n, w)
    return vcat(padrow, img)
end
function padbottom(img::AbstractMatrix{RGB{N0f8}}, n::Integer)
    _, w = size(img)
    padrow = fill(RGB{N0f8}(1, 1, 1), n, w)
    return vcat(img, padrow)
end

function crop_white_margins(input_path::AbstractString, output_path::AbstractString; threshold::Real=0.98, more=0)
    if !isfile(input_path)
        @error "Input file not found for cropping: $input_path"
        return nothing
    end
    try
        img = load(input_path)
        cropped = crop_white_margins(img; threshold=threshold)
        if more == 1 # Crop out middle rows of image (guard small images)
            h, w = size(cropped)
            if h < 800 || w < 1500
                @warn "Image too small for advanced crop (more=1); using basic crop." (h=h, w=w)
                save(output_path, cropped)
                return output_path
            end

            r1 = 1:min(h, 600)
            c1 = 1:min(w, 1200)
            left = crop_white_margins(cropped[r1, c1])

            rmid = 1:min(h, 1500)
            cmid = 1301:min(w, 2500)
            middle = crop_white_margins(cropped[rmid, cmid])

            r2 = 1:min(h, 700)
            c2_start = min(2500, w)
            right = crop_white_margins(cropped[r2, c2_start:w])
            rbot_start = min(2001, h)
            bot = crop_white_margins(cropped[rbot_start:h, 1:end])

            h, t = size(bot)
            height = 1000 * h / t
            fig = Figure(; size=(2700, 3000))
            ax_bottom = Axis(fig[2, 1], width=1000, height=height, valign=:bottom)
            hidedecorations!(ax_bottom)
            hidespines!(ax_bottom)
            image!(ax_bottom, rotr90(bot))

            h, t = size(left)
            height = 800 * h / t
            ax_top_left = Axis(fig[1, 1], width=800, height=height, halign=:left)
            image!(ax_top_left, rotr90(left))
            hidedecorations!(ax_top_left)
            hidespines!(ax_top_left)

            h, t = size(right)
            height = 800 * h / t
            ax_top_right = Axis(fig[1, 2], width=800, height=height, halign=:right)
            image!(ax_top_right, rotr90(right))
            hidedecorations!(ax_top_right)
            hidespines!(ax_top_right)

            h, t = size(middle)
            height = 800 * h / t
            ax_top_middle = Axis(fig[2, 2], width=800, height=height, halign=:center)
            image!(ax_top_middle, rotr90(middle))
            hidedecorations!(ax_top_middle)
            hidespines!(ax_top_middle)

            # colsize!(fig.layout, 1, Relative(0.25))
            # colsize!(fig.layout, 2, Relative(0.25))
            # colsize!(fig.layout, 3, Relative(0.25))
            # colsize!(fig.layout, 4, Relative(0.25))

            save(output_path * "temp.png", fig)
            img = load(output_path * "temp.png")
            img = crop_white_margins(img; threshold=0.98)
            cropped = img

            # Patch to remove artifact at bottom right (using transparent pixels)
            if size(cropped, 1) >= 2706 && size(cropped, 2) >= 2249
                # Lower right corner: (2249, 2706)
                # Size: 274 (W) x 303 (H) -> TL: (1975, 2403)
                cropped[2403:2706, 1975:2249] .= RGBA(1.0, 1.0, 1.0, 0.0)
            end
        elseif more == 2 # Crop out middle rows of image (guard small images)
            h, w = size(cropped)
            if h < 2600 || w < 3000
                @warn "Image too small for advanced crop (more=2); using basic crop." (h=h, w=w)
                save(output_path, cropped)
                return output_path
            end

            left = crop_white_margins(cropped[1:min(h, 2500), 600:min(w, 2500)])
            middle = crop_white_margins(cropped[1:min(h, 2500), 2500:min(w, 5500)])
            right_start = min(6500, w)
            right = crop_white_margins(crop_white_margins(cropped[1:min(h, 2500), right_start:w]))
            bot_start = min(2201, h)
            bot = crop_white_margins(crop_white_margins(cropped[bot_start:h, 1:end]))

            h, t = size(bot)
            height = 1000 * h / t
            fig = Figure(; size=(2700, 3000))
            ax_bottom = Axis(fig[2, 1], width=1000, height=height, valign=:bottom)
            hidedecorations!(ax_bottom)
            hidespines!(ax_bottom)
            image!(ax_bottom, rotr90(bot))

            h, t = size(left)
            height = 800 * h / t
            ax_top_left = Axis(fig[1, 1], width=800, height=height, halign=:left)
            image!(ax_top_left, rotr90(left))
            hidedecorations!(ax_top_left)
            hidespines!(ax_top_left)

            h, t = size(right)
            height = 800 * h / t
            ax_top_right = Axis(fig[1, 2], width=800, height=height, halign=:right)
            image!(ax_top_right, rotr90(right))
            hidedecorations!(ax_top_right)
            hidespines!(ax_top_right)

            h, t = size(middle)
            height = 800 * h / t
            ax_top_middle = Axis(fig[2, 2], width=800, height=height, halign=:center)
            image!(ax_top_middle, rotr90(middle))
            hidedecorations!(ax_top_middle)
            hidespines!(ax_top_middle)

            save(output_path * "temp.png", fig)
            img = load(output_path * "temp.png")
            img = crop_white_margins(img; threshold=0.98)
            cropped = img
        end
        save(output_path, cropped)
        @info "Cropped image saved to $output_path"
        return output_path
    catch e
        @error "Error cropping image $input_path:" e
        return nothing
    end
end

# Lighten saved RGBA images without changing transparency
clamp01(x) = max(0.0, min(1.0, x))
function brighten_image!(path::AbstractString; factor::Real=1.18)
    if !isfile(path)
        @warn "Cannot brighten missing image $path"
        return nothing
    end
    try
        img = load(path)
        boosted = map(img) do c
            rgba = RGBA(c) # Promotes RGB → RGBA if needed
            RGBA(clamp01(rgba.r * factor), clamp01(rgba.g * factor), clamp01(rgba.b * factor), alpha(rgba))
        end
        save(path, boosted)
        @info "Brightened image $path by factor=$(round(factor; digits=2))"
        return path
    catch e
        @warn "Failed to brighten image $path" e
        return nothing
    end
end


# --- 1. Load Data ---
println("--- Loading Data for Figure 4 ---")
metal_results = get_data(metal_group, nominal_variation, "results")
metal_spectral = get_data(metal_group, nominal_variation, "spectral")
metal_design = PYVISTA_AVAILABLE ? get_data(metal_group, nominal_variation, "design_y") : nothing
metal_fields = PYVISTA_AVAILABLE ? get_data(metal_group, nominal_variation, "field_y") : nothing

dielectric_results = get_data(dielectric_group, nominal_variation, "results")
dielectric_spectral = get_data(dielectric_group, nominal_variation, "spectral")
dielectric_spectral_freeform = get_data("Freeform_Dielectric", nominal_variation, "spectral")
dielectric_design_freeform = PYVISTA_AVAILABLE ? get_data("Freeform_Dielectric", nominal_variation, "design_y") : nothing
dielectric_design = PYVISTA_AVAILABLE ? get_data(dielectric_group, nominal_variation, "design_y") : nothing
dielectric_fields = PYVISTA_AVAILABLE ? get_data(dielectric_group, nominal_variation, "field_y") : nothing

# Load Freeform Metal and Freeform Dielectric results for 3D DOFs
freeform_metal_results = get_data("Freeform_Metal", nominal_variation, "results")
freeform_dielectric_results = get_data("Freeform_Dielectric", nominal_variation, "results")

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

# Generate insets
plot_geometry_for_inset(dielectric_design_freeform, dielectric_design_freeform_path, group_name="Freeform_Dielectric")
plot_geometry_for_inset(dielectric_design, dielectric_design_path, group_name="Constrained_Dielectric")

# Crop the insets
crop_white_margins(dielectric_design_freeform_path, dielectric_design_freeform_cropped)
crop_white_margins(dielectric_design_path, dielectric_design_cropped)

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
    plotter_combined = plot_field_with_cross_sections(
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

    plotter_combined_diel = plot_field_with_cross_sections_dielectric(
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
l_3d_diel_spec = lines!(ax_spec, dielectric_wl, dielectric_gy_freeform, linewidth=3, color=color_3d_dielectric)
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
# (a-topleft) freeform gy optimized design with an orange dashed boarder
img_dielectric = load(dielectric_design_cropped)
img_dielectric_freeform = load(dielectric_design_freeform_cropped)
add_inset_with_border!(fig[1, 2], img_dielectric_freeform, :right, 0.67, :red, :dash, Relative(0.425), false)
# (a-topright) sphere design with a black boarder
add_inset_with_border!(fig[1, 2], img_dielectric, 0.75, -0.065, :black, :solid, Relative(0.425), false)

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

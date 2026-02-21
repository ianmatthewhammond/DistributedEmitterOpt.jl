using DistributedEmitterOpt
using PyCall, PyPlot, CairoMakie, LaTeXStrings
const Viz = Base.get_extension(DistributedEmitterOpt, :VisualizationExt); using .Viz
# using Gridap, GridapGmsh, GridapMakie # Unused for plotting results
using CSV, DataFrames
using Images, JLD2, Random
using ColorSchemes, Colors
using Images
using FileIO

# Set Makie font to Times
CairoMakie.activate!()
Figure, Axis = CairoMakie.Figure, CairoMakie.Axis
CairoMakie.update_theme!(
    font="Times",
    fontsize=34
)

vtk = pyimport("vtk")
np = pyimport("numpy")
pv = pyimport("pyvista")
pv.global_theme.allow_empty_mesh = true

# Global font sizes for Makie plots
makie_text_size = text_size = 36
makie_title_size = title_size = 36
makie_axis_label_size = axis_label_size = 36

pv.global_theme.font.family = "times"
root = "./"
np.random.seed(0)

# Functions
function crop_white_margins(img::AbstractArray; threshold::Real=1.0)
    chans = channelview(img)
    white_mask = dropdims(all(chans .>= threshold, dims=1); dims=1)
    nonwhite_mask = .!white_mask
    rows = Base.vec(any(nonwhite_mask, dims=2))
    cols = Base.vec(any(nonwhite_mask, dims=1))
    if !any(rows) || !any(cols)
        return img
    end
    rmin, rmax = findfirst(rows), findlast(rows)
    cmin, cmax = findfirst(cols), findlast(cols)
    return img[rmin:rmax, cmin:cmax]
end

function crop_white_margins(input_path::AbstractString, output_path::AbstractString; threshold::Real=1.0)
    img = load(input_path)
    cropped = crop_white_margins(img; threshold=threshold)
    save(output_path, cropped)
    return output_path
end

# --- Configuration ---
constrained_group = "Constrained_Metal"
benchmark_group = "Spheres"
mono_opt_variation = "nominal"
bi_opt_variation = "polarization_Bi"
benchmark_variation = "polarization_Bi"
wavelength_target = 532.0
wavelength_min = wavelength_target - 50.0
wavelength_max = wavelength_target + 50.0

# --- Define Plotting Styles ---
constrained_color_mono = :purple # Monopolarized 2D is PURPLE
constrained_color_bi = :blue   # Bipolarized 2D is BLUE
sphere_color = :green
mono_style = :solid
bi_style = :dashdot
gx_style = :dot
common_linewidth = 6.0 # Made lines thicker

# --- Output Paths ---
temp_path = "figures/paper-figures-scripts/temporary/fig2"
mkpath(temp_path)
geom_con_mono_raw = joinpath(temp_path, "geom_con_mono_raw.png")
geom_sphere_raw = joinpath(temp_path, "geom_sphere_raw.png")
geom_con_bi_raw = joinpath(temp_path, "geom_con_bi_raw.png")
geom_con_mono_cropped = joinpath(temp_path, "geom_con_mono_cropped.png")
geom_sphere_cropped = joinpath(temp_path, "geom_sphere_cropped.png")
geom_con_bi_cropped = joinpath(temp_path, "geom_con_bi_cropped.png")
final_figure_filename = "figures/paper-figures-scripts/figures/Figure-2-Results.pdf"
mkpath("figures/paper-figures-scripts/figures")

# --- 1. Load Data ---
println("Loading spectral data...")
spec_con_mono = get_figure_data(constrained_group, mono_opt_variation, "spectral")
spec_con_bi = get_figure_data(constrained_group, bi_opt_variation, "spectral")
spec_sphere = get_figure_data(benchmark_group, benchmark_variation, "spectral")
println("Loading geometry data...")
mesh_con_mono = get_figure_data(constrained_group, mono_opt_variation, "design_y")
mesh_con_bi = get_figure_data(constrained_group, bi_opt_variation, "design_y")
mesh_sphere = get_figure_data(benchmark_group, benchmark_variation, "design_y")

if any(isnothing, [spec_con_mono, spec_con_bi, spec_sphere, mesh_con_mono, mesh_con_bi, mesh_sphere])
    error("Failed to load required data.")
end

# --- 2. Create Geometry Plots for Insets ---
function plot_geometry_for_inset(mesh_pyobject, output_filename; scalar_field="p", window_size=(400, 400), color_hex_str::String, flipy=false)
    println("Generating geometry plot for inset: $output_filename with color $color_hex_str")
    pv.global_theme.transparent_background = true
    plotter = pv.Plotter(off_screen=true, window_size=window_size)
    if !hasproperty(mesh_pyobject, :bounds)
        error("Mesh object missing 'bounds'.")
    end
    xl, xr, yl, yr, _, _ = mesh_pyobject.bounds
    W = xr - xl
    L = yr - yl
    if W <= 0 || L <= 0
        error("Invalid W or L from bounds.")
    end
    plot_material(np, pv, plotter, mesh_pyobject, W, L; colorbar=false, num_periods_x=2, num_periods_y=2, contour=true, design_field=scalar_field, contour_color=color_hex_str, flipy=flipy)
    # plotter.remove_bounds_axes()

    plotter.add_axes(
        line_width=5,
        cone_radius=0.4,
        shaft_length=0.8,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.6, 0.2),
        viewport=(0.66, 0.0, 1.0, 0.34), # Moved axes to bottom right
    )

    plotter.screenshot(output_filename)
    plotter.clear()
    plotter.close()
    println("Saved geometry plot to $output_filename")
end

# Define hex colors for PyVista
color_hex_con_mono = "#b097d1" # Purple
color_hex_con_bi = "#496cb7"   # Blue
color_hex_sphere = "#008000"   # Green

plot_geometry_for_inset(mesh_con_mono, geom_con_mono_raw, color_hex_str=color_hex_con_mono, flipy=true)
plot_geometry_for_inset(mesh_sphere, geom_sphere_raw, color_hex_str=color_hex_sphere, flipy=false)
plot_geometry_for_inset(mesh_con_bi, geom_con_bi_raw, color_hex_str=color_hex_con_bi, flipy=true)

# --- 3. Crop Geometry Plots ---
println("Cropping geometry plots...")
crop_threshold = 0.99
crop_white_margins(geom_con_mono_raw, geom_con_mono_cropped, threshold=crop_threshold)
crop_white_margins(geom_sphere_raw, geom_sphere_cropped, threshold=crop_threshold)
crop_white_margins(geom_con_bi_raw, geom_con_bi_cropped, threshold=crop_threshold)

let
    img_con_mono = load(geom_con_mono_cropped)
    img_sphere = load(geom_sphere_cropped)
    img_con_bi = load(geom_con_bi_cropped)

    fig = Figure(resolution=(1600, 800), fontsize=22, fonts=(; regular="Times", weird="Times")) # Adjusted resolution for 1x2 layout

    # Determine y-axis tick range for log scale (powers of 10)
    all_g_values = Float64[]
    for spec_data in [spec_con_mono, spec_con_bi, spec_sphere]
        if !isnothing(spec_data)
            !isnothing(get(spec_data, "g_y", nothing)) && append!(all_g_values, spec_data["g_y"])
            !isnothing(get(spec_data, "g_x", nothing)) && append!(all_g_values, spec_data["g_x"])
            !isnothing(get(spec_data, "g_combined", nothing)) && append!(all_g_values, spec_data["g_combined"])
        end
    end
    # Filter out zeros or negative values before log, and handle empty case
    all_g_values = filter(x -> x > 0, all_g_values)
    min_g = isempty(all_g_values) ? 1.0 : minimum(all_g_values)
    max_g = isempty(all_g_values) ? 100.0 : maximum(all_g_values)

    # Calculate y-axis range based on spheres data
    sphere_min = minimum(spec_sphere["g_combined"])
    y_min_power = floor(Int, log10(sphere_min))
    y_max_power = ceil(Int, log10(max_g))
    log_yticks = ([10.0^i for i in y_min_power:y_max_power], [L"10^{%$i}" for i in y_min_power:y_max_power])

    # --- Left Plot (Comparison) ---
    ax_left = Axis(fig[1, 1],
        xlabel="Emission Wavelength λₑ (nm)", ylabel="Enhancement Factor",
        yscale=log10, title="(a) Performance Comparison",
        xlabelsize=axis_label_size, ylabelsize=axis_label_size,
        xticklabelsize=axis_label_size, yticklabelsize=axis_label_size,
        titlesize=axis_label_size,
        yticks=log_yticks,
        yminorticksvisible=false, titlefont="Times",
        xgridwidth=2, ygridwidth=2, xgridcolor=(:black, 0.20), ygridcolor=(:black, 0.20))

    # Add spectral lines for (a)
    !isnothing(spec_con_mono) && lines!(ax_left, spec_con_mono["wavelengths"], spec_con_mono["g_y"], linewidth=common_linewidth, color=constrained_color_mono, linestyle=mono_style)
    !isnothing(spec_con_bi) && lines!(ax_left, spec_con_bi["wavelengths"], spec_con_bi["g_combined"], linewidth=common_linewidth, color=constrained_color_bi, linestyle=bi_style)
    !isnothing(spec_sphere) && lines!(ax_left, spec_sphere["wavelengths"], spec_sphere["g_combined"], linewidth=common_linewidth, color=sphere_color, linestyle=:dashdot)

    # Add vertical line and label for 532nm in (a)
    vlines!(ax_left, [wavelength_target], color=:black, linestyle=:dashdot, linewidth=4.0)
    text!(ax_left, wavelength_target + 2, 10^(y_min_power + 0.5 * (y_max_power - y_min_power) - 1.5), text="Pump λₚ=532nm", color=:black, fontsize=text_size, align=(:left, :center))

    if !isnothing(spec_con_mono)
        idx_532 = argmin(abs.(spec_con_mono["wavelengths"] .- 532))
        y_val_mono = spec_con_mono["g_y"][idx_532]

        # Find y-value for 2D monopolarized at 510nm
        idx_510_mono = argmin(abs.(spec_con_mono["wavelengths"] .- 510))
        y_val_mono_510 = spec_con_mono["g_y"][idx_510_mono]
        # Adjusted: label anchor down by 10^0.25 and right by 15nm
        label_x = 0.25 * (wavelength_max - wavelength_min) + wavelength_min + 10
        label_y = 10^(y_min_power + 0.935 * (y_max_power - y_min_power))
        lines!(ax_left, [label_x, 510], [label_y, y_val_mono_510], color=constrained_color_mono, linestyle=:solid, linewidth=2)
    end
    if !isnothing(spec_sphere)
        idx_532 = argmin(abs.(spec_sphere["wavelengths"] .- 532))
        y_val_sphere = spec_sphere["g_combined"][idx_532]
        # Add line to label (Spheres at 0.05, 0.28)
        lines!(ax_left, [532, 503], [y_val_sphere, 10^2.35],
            color=sphere_color, linestyle=:solid, linewidth=2)
    end
    if !isnothing(spec_con_bi)
        idx_532 = argmin(abs.(spec_con_bi["wavelengths"] .- 532))
        y_val_bi = spec_con_bi["g_combined"][idx_532]
        # Add line to label (2D Bipolarized at 0.65, 0.925)
        lines!(ax_left, [532, 545], [y_val_bi, 10^4.8],
            color=constrained_color_bi, linestyle=:solid, linewidth=2)
    end

    xlims!(ax_left, wavelength_min, wavelength_max)
    ylims!(ax_left, 10^y_min_power, 10^y_max_power)

    # --- Right Plot (2D Bipolarized Breakdown) ---
    ax_right = Axis(fig[1, 2],
        xlabel="Emission Wavelength λₑ (nm)",
        yscale=log10, title="(b) Bipolarized Enhancement Contributions",
        xlabelsize=axis_label_size, ylabelsize=axis_label_size,
        xticklabelsize=axis_label_size, yticklabelsize=axis_label_size,
        titlesize=axis_label_size,
        yticks=log_yticks,
        yminorticksvisible=false, titlefont="Times",
        xgridwidth=2, ygridwidth=2, xgridcolor=(:black, 0.20), ygridcolor=(:black, 0.20))
    !isnothing(spec_con_bi) && lines!(ax_right, spec_con_bi["wavelengths"], spec_con_bi["g_y"], linewidth=common_linewidth, color=constrained_color_bi, linestyle=mono_style)
    !isnothing(spec_con_bi) && lines!(ax_right, spec_con_bi["wavelengths"], spec_con_bi["g_x"], linewidth=common_linewidth, color=constrained_color_bi, linestyle=gx_style)
    !isnothing(spec_con_bi) && lines!(ax_right, spec_con_bi["wavelengths"], spec_con_bi["g_combined"], linewidth=common_linewidth, color=constrained_color_bi, linestyle=bi_style)
    vlines!(ax_right, [wavelength_target], color=:black, linestyle=:dashdot, linewidth=4.0)
    text!(ax_right, wavelength_target - 20, 10^y_min_power * 2.0, text="Pump λₚ=532nm", color=:black, fontsize=text_size, align=(:center, :center))

    if !isnothing(spec_con_bi)
        idx = argmin(abs.(spec_con_bi["wavelengths"] .- 510))
        text!(ax_right, 490, spec_con_bi["g_x"][idx] * 0.27, text=L"g_y", color=constrained_color_bi, fontsize=text_size)
        text!(ax_right, 490, spec_con_bi["g_x"][idx] * 0.08, text=L"g_x", color=constrained_color_bi, fontsize=text_size)
        text!(ax_right, 497, spec_con_bi["g_combined"][idx] * 1.1, text=L"g_x+g_y", color=constrained_color_bi, fontsize=text_size)

        # Add line style indicators
        # g_y indicator (solid line)
        lines!(ax_right, [488 + 1, 492 + 1], [spec_con_bi["g_x"][idx] * 0.26, spec_con_bi["g_x"][idx] * 0.26],
            color=constrained_color_bi, linestyle=mono_style, linewidth=common_linewidth)

        # g_x indicator (dotted line)
        lines!(ax_right, [488 + 2, 492 + 2], [spec_con_bi["g_x"][idx] * 0.075, spec_con_bi["g_x"][idx] * 0.075],
            color=constrained_color_bi, linestyle=gx_style, linewidth=common_linewidth)

        # g_x+g_y indicator (dashdot line)
        lines!(ax_right, [492 + 5, 502 + 5], [spec_con_bi["g_combined"][idx] * 1.05, spec_con_bi["g_combined"][idx] * 1.05],
            color=constrained_color_bi, linestyle=bi_style, linewidth=common_linewidth)
    end

    xlims!(ax_right, wavelength_min, wavelength_max)
    ylims!(ax_right, 10^y_min_power, 10^y_max_power)
    hideydecorations!(ax_right, grid=false, ticks=false)

    # Text labels above insets in panel (a)
    text!(ax_left, 0.05, 0.935, text="2D Monopolarized", color=constrained_color_mono, fontsize=text_size, space=:relative)
    text!(ax_left, 0.65, 0.925, text="2D Bipolarized", color=constrained_color_bi, fontsize=text_size, space=:relative)
    text!(ax_left, 0.05, 0.28, text="Spheres", color=sphere_color, fontsize=text_size, space=:relative)

    # --- Add Insets with Borders ---
    inset_padding = 5 # Reduced padding
    border_width = 2

    function add_inset_with_border!(fig_pos, img, halign, valign, border_color, border_style, inset_size=Relative(0.3), border=true) # Default smaller inset
        current_padding = border ? (inset_padding - border_width, inset_padding - border_width, inset_padding - border_width, inset_padding - border_width) : (inset_padding, inset_padding, inset_padding, inset_padding)

        ax_inset = Axis(fig_pos, width=inset_size, height=inset_size,
            halign=halign, valign=valign, aspect=DataAspect(),
            backgroundcolor=(:white, 0.0))
        image!(ax_inset, rotr90(img))
        hidedecorations!(ax_inset)
        hidespines!(ax_inset)

        if border
            Box(fig_pos, width=inset_size, height=inset_size,
                halign=halign, valign=valign,
                color=(:white, 0.0),
                strokecolor=border_color,
                strokewidth=border_width,
                linestyle=border_style)
        end
    end

    # Add diagonal line from label in (a) to edge of (a)
    lines!(ax_left, [0.96, 1.0], [0.945, 0.94],
        color=constrained_color_bi, linestyle=:solid, linewidth=2, space=:relative)

    # Add diagonal line from edge of (b) to scatter point
    lines!(ax_right, [0.0, 0.5], [0.94, 0.9],
        color=constrained_color_bi, linestyle=:solid, linewidth=2, space=:relative)

    # Insets for Left Panel (a)
    add_inset_with_border!(fig[1, 1], img_con_mono, :left, :top, constrained_color_mono, :solid, Relative(0.3), false)
    add_inset_with_border!(fig[1, 1], img_con_bi, :right, :top, constrained_color_bi, :dashdot, Relative(0.3), false)
    # Repositioned spheres inset: centered around x=510nm, y corresponding to its label's original relative height
    # Calculate approximate relative x for 510nm
    relative_x_510 = (510 - wavelength_min) / (wavelength_max - wavelength_min)
    # Move sphere inset down by factor of 10 in logy: 0.28/10 = 0.028
    add_inset_with_border!(fig[1, 1], img_sphere, relative_x_510, 0.028, sphere_color, :solid, Relative(0.35), false)

    # Inset for Right Panel (b) - 2D Bipolarized again
    add_inset_with_border!(fig[1, 2], img_con_bi, :right, :bottom, constrained_color_bi, :dashdot, Relative(0.6), false) # Larger inset

    # --- Layout Adjustments ---
    linkaxes!(ax_left, ax_right) # Link all axes
    colgap!(fig.layout, 10)
    # rowgap!(fig.layout, 10) # Not needed for 1 row

    save(final_figure_filename, fig)
    println("Saved final PDF to $final_figure_filename")
end

println("Figure generation complete.")
using DistributedEmitterOpt
using PyCall, CairoMakie
const Viz = Base.get_extension(DistributedEmitterOpt, :VisualizationExt); using .Viz
using CSV, DataFrames
using Images, JLD2, Random
using ColorSchemes, Colors
using Images
using FileIO # For loading saved images
using StatsBase: percentile

Figure, Axis = CairoMakie.Figure, CairoMakie.Axis
CairoMakie.activate!()

# Set global font to Times
CairoMakie.update_theme!(
    font="Times",
    fontsize=20
)

# Colors imported from Visualization

# --- PyCall Setup ---
# (Keep the PyCall setup block from the original script)
SCRIPT_DIR = @__DIR__
YAML_FILE = joinpath(SCRIPT_DIR, "Figures.scp.yaml") # Assuming this file exists
try
    global pv = pyimport("pyvista")
    try
        pv.start_xvfb()
        @info "PyVista using Xvfb for off-screen rendering."
    catch e
        @warn "Could not start Xvfb (may not be needed or installed): $e"
        try
            pv.global_theme.off_screen = true
            @info "Set PyVista global_theme.off_screen = true"
        catch theme_e
            @warn "Could not set PyVista global_theme.off_screen: $theme_e. Plotting might require a display."
        end
    end
    global np = pyimport("numpy")
    @info "Successfully imported PyVista and NumPy via PyCall."
    global PYVISTA_AVAILABLE = true
catch e
    @warn "PyCall: Failed to import PyVista or NumPy. VTU loading will be disabled. Error: $e"
    global PYVISTA_AVAILABLE = false
end

function load_spectral_from_path(path::AbstractString)
    if !isfile(path)
        @error "Spectral file not found: $path"
        return nothing
    end
    try
        loaded_data = JLD2.load_object(path)
        if isa(loaded_data, Dict) && all(k -> haskey(loaded_data, k), ["wavelengths", "g_y", "g_x", "g_combined"])
            @info "Loaded spectral data from $path as Dict."
            return loaded_data
        elseif isa(loaded_data, Tuple) && length(loaded_data) == 4 && all(isa.(loaded_data, AbstractVector))
            @info "Loaded spectral data from $path as Tuple. Converting to Dict."
            return Dict(
                "wavelengths" => loaded_data[1],
                "g_y" => loaded_data[2],
                "g_x" => loaded_data[3],
                "g_combined" => loaded_data[4]
            )
        else
            @warn "Spectral data at $path has unexpected format: $(typeof(loaded_data)). Returning raw data."
            return loaded_data
        end
    catch e
        @error "Error loading spectral file $path: $e"
        return nothing
    end
end

# --- Configuration ---
text_size = 34 # Standardized font size
title_size = 34 # Standardized font size
# fig_width = 2800 # This might be too large for a paper figure, consider reducing
if PYVISTA_AVAILABLE
    pv.global_theme.font.family = "times"
end
root = "./" # Assuming data is relative to the script location

# --- Legacy helpers removed ---
# Data loading now handled by DistributedEmitterOpt.Visualization.get_figure_data


# --- Helper Functions (Geometry Plotting & Cropping - Unchanged) ---
# [ Functions plot_material, plot_geometry_pyvista, crop_white_margins unchanged ]
function get_material_color(material_type, case)
    if material_type == "metal"
        if case == "isotropic"
            return pv.Color(COLOR_METAL_ISOTROPIC)
        elseif case == "anisotropic"
            return pv.Color(COLOR_METAL_ANISOTROPIC)
        elseif case == "iso2ani"
            return pv.Color(COLOR_METAL_ISO2ANI)
        elseif case == "inelastic"
            return pv.Color(COLOR_METAL_INELASTIC)
        else
            return pv.Color(COLOR_METAL_ISOTROPIC)
        end
    else  # dielectric
        if case == "isotropic"
            return pv.Color(COLOR_DIEL_ISOTROPIC)
        elseif case == "anisotropic"
            return pv.Color(COLOR_DIEL_ANISOTROPIC)
        elseif case == "iso2ani"
            return pv.Color(COLOR_DIEL_ISO2ANI)
        elseif case == "inelastic"
            return pv.Color(COLOR_DIEL_INELASTIC)
        else
            return pv.Color(COLOR_DIEL_ISOTROPIC)
        end
    end
end


function plot_geometry_pyvista(mesh_pyobject, output_filename;
    window_size=(400, 400), zoom=1.8, view_angle=(-45, 30, 15), material_type="metal", case="isotropic")

    if isnothing(mesh_pyobject)
        return
    end

    if !PYVISTA_AVAILABLE
        @warn "PyVista not available, skipping plot."
        return
    end

    xl, xr, yl, yr, _, _ = mesh_pyobject.bounds
    W = xr - xl
    L = yr - yl
    color = get_material_color(material_type, case)

    save_geometry_snapshot(np, pv, mesh_pyobject, output_filename, W, L;
        color=color,
        window_size=window_size,
        zoom=zoom,
        axes_viewport=(0.0, 0.0, 0.33, 0.33),
        full=false, contour=true, num_periods_x=2, num_periods_y=2, flipx=false, flipy=true
    )
end

# crop_white_margins imported from Visualization


# --- Configuration for Figure 5 (Anisotropy + Splitting) ---
metal_group_key = "Constrained_Metal"
diel_group_key = "Constrained_Dielectric"

# Anisotropy Variations
variation_nominal = "nominal"
variation_anisotropic = "isotropy_Anisotropic"
variation_ani2iso = "isotropy_Ani2Iso"
variation_iso2ani = "isotropy_Iso2Ani"

# Splitting Variations
variation_splitting = "frequency_Shifted" # New variation key

# Wavelengths
wavelength_target_nominal = 532.0 # nm
wavelength_target_pump_split = 532.0 # nm
wavelength_target_emission_split = 549.0 # nm

# --- Define output paths ---
temp_path = "figures/paper-figures-scripts/temporary/fig4"
mkpath(temp_path)

# Anisotropy Geometry Filenames (Metal & Dielectric)
geom_metal_nominal_filename_raw = joinpath(temp_path, "fig5_geom_metal_nominal_raw.png")
geom_metal_anisotropic_filename_raw = joinpath(temp_path, "fig5_geom_metal_anisotropic_raw.png")
geom_metal_nominal_filename_cropped = joinpath(temp_path, "fig5_geom_metal_nominal_cropped.png")
geom_metal_anisotropic_filename_cropped = joinpath(temp_path, "fig5_geom_metal_anisotropic_cropped.png")

geom_diel_nominal_filename_raw = joinpath(temp_path, "fig5_geom_diel_nominal_raw.png")
geom_diel_anisotropic_filename_raw = joinpath(temp_path, "fig5_geom_diel_anisotropic_raw.png")
geom_diel_nominal_filename_cropped = joinpath(temp_path, "fig5_geom_diel_nominal_cropped.png")
geom_diel_anisotropic_filename_cropped = joinpath(temp_path, "fig5_geom_diel_anisotropic_cropped.png")

# Splitting Geometry Filenames (Metal & Dielectric) - NEW
geom_metal_splitting_filename_raw = joinpath(temp_path, "fig5_geom_metal_splitting_raw.png")
geom_metal_splitting_filename_cropped = joinpath(temp_path, "fig5_geom_metal_splitting_cropped.png")
geom_diel_splitting_filename_raw = joinpath(temp_path, "fig5_geom_diel_splitting_raw.png")
geom_diel_splitting_filename_cropped = joinpath(temp_path, "fig5_geom_diel_splitting_cropped.png")

# Starting Point Geometry Filenames (Metal & Dielectric) - NEW
geom_metal_init_filename_raw = joinpath(temp_path, "fig4_geom_metal_init_raw.png")
geom_metal_init_filename_cropped = joinpath(temp_path, "fig4_geom_metal_init_cropped.png")
geom_diel_init_filename_raw = joinpath(temp_path, "fig4_geom_diel_init_raw.png")
geom_diel_init_filename_cropped = joinpath(temp_path, "fig4_geom_diel_init_cropped.png")

# Final Figure Filename
final_figure_filename = "figures/paper-figures-scripts/figures/Figure-4-Anisotropy_Combined.pdf"
mkpath("figures/paper-figures-scripts/figures")
mkpath("figures")

# Paths to updated metal spectral sweeps (post-anisotropy bonus, spectral)
bonus_spectral_dir = "data/post-anisotropy-bonus"
bonus_metal_iso_path = joinpath(bonus_spectral_dir, "iso-spectral.jld2")
bonus_metal_aniso_path = joinpath(bonus_spectral_dir, "ani-spectral.jld2")

# --- 1. Load Data ---
# --- 1. Load Data ---
pv.global_theme.allow_empty_mesh = true

println("--- Loading Anisotropy Data (Metal) ---")
spectral_metal_nominal = get_figure_data(metal_group_key, variation_nominal, "spectral")
spectral_metal_anisotropic = get_figure_data(metal_group_key, variation_anisotropic, "spectral")
# spectral_metal_ani2iso = get_figure_data(metal_group_key, variation_ani2iso, "spectral")
spectral_metal_iso2ani = get_figure_data(metal_group_key, variation_iso2ani, "spectral")
bonus_spectral_metal_iso = get_figure_data("Bonus", "iso-spectral.jld2", "spectral")
bonus_spectral_metal_aniso = get_figure_data("Bonus", "ani-spectral.jld2", "spectral")
mesh_metal_nominal = get_figure_data(metal_group_key, variation_nominal, "design_y")
mesh_metal_anisotropic = get_figure_data(metal_group_key, variation_anisotropic, "design_y")

println("--- Loading Anisotropy Data (Dielectric) ---")
spectral_diel_nominal = get_figure_data(diel_group_key, variation_nominal, "spectral")
spectral_diel_anisotropic = get_figure_data(diel_group_key, variation_anisotropic, "spectral")
# spectral_diel_ani2iso = get_figure_data(diel_group_key, variation_ani2iso, "spectral")
spectral_diel_iso2ani = get_figure_data(diel_group_key, variation_iso2ani, "spectral")
if isnothing(spectral_diel_iso2ani)
    @warn "Missing spectral_diel_iso2ani. Using nominal as fallback."
    spectral_diel_iso2ani = spectral_diel_nominal
end
mesh_diel_nominal = get_figure_data(diel_group_key, variation_nominal, "design_y")
mesh_diel_anisotropic = get_figure_data(diel_group_key, variation_anisotropic, "design_y")

println("--- Loading Splitting Data (Metal) ---") # NEW
spectral_metal_splitting = get_figure_data(metal_group_key, variation_splitting, "spectral")
mesh_metal_splitting = get_figure_data(metal_group_key, variation_splitting, "design_y")

println("--- Loading Splitting Data (Dielectric) ---") # NEW
spectral_diel_splitting = get_figure_data(diel_group_key, variation_splitting, "spectral")
mesh_diel_splitting = get_figure_data(diel_group_key, variation_splitting, "design_y")

println("--- Loading Starting Point (Init) Geometries ---") # NEW
starting_metal_vtu = "figures/paper-figures-scripts/data/data_starting/metal/y_design.vtu"
starting_diel_vtu = "figures/paper-figures-scripts/data/data_starting/dielectric/y_design.vtu"

mesh_metal_init = PYVISTA_AVAILABLE ? pyimport("pyvista").read(starting_metal_vtu) : nothing
mesh_diel_init = PYVISTA_AVAILABLE ? pyimport("pyvista").read(starting_diel_vtu) : nothing

# --- Basic Checks ---
# Combine all checks
all_data_ok = !any(isnothing, [
    spectral_metal_nominal, spectral_metal_anisotropic, bonus_spectral_metal_iso, bonus_spectral_metal_aniso,
    mesh_metal_nominal, mesh_metal_anisotropic,
    spectral_diel_nominal, spectral_diel_anisotropic, mesh_diel_nominal, mesh_diel_anisotropic,
    spectral_metal_splitting, mesh_metal_splitting, spectral_diel_splitting, mesh_diel_splitting,
    mesh_metal_init, mesh_diel_init
])

if !all_data_ok || !PYVISTA_AVAILABLE
    error("Failed to load required data for Anisotropy/Splitting or PyVista is not available. Aborting figure generation.")
end

# --- 2. Create ALL Geometry Plots ---
# Anisotropy Geometries
plot_geometry_pyvista(mesh_metal_nominal, geom_metal_nominal_filename_raw, material_type="metal", case="isotropic")
plot_geometry_pyvista(mesh_metal_anisotropic, geom_metal_anisotropic_filename_raw, material_type="metal", case="anisotropic")
crop_white_margins(geom_metal_nominal_filename_raw, geom_metal_nominal_filename_cropped)
crop_white_margins(geom_metal_anisotropic_filename_raw, geom_metal_anisotropic_filename_cropped)
plot_geometry_pyvista(mesh_diel_nominal, geom_diel_nominal_filename_raw, material_type="dielectric", case="isotropic")
plot_geometry_pyvista(mesh_diel_anisotropic, geom_diel_anisotropic_filename_raw, material_type="dielectric", case="anisotropic")
crop_white_margins(geom_diel_nominal_filename_raw, geom_diel_nominal_filename_cropped)
crop_white_margins(geom_diel_anisotropic_filename_raw, geom_diel_anisotropic_filename_cropped)

# Splitting Geometries - NEW
plot_geometry_pyvista(mesh_metal_splitting, geom_metal_splitting_filename_raw, material_type="metal", case="inelastic")
crop_white_margins(geom_metal_splitting_filename_raw, geom_metal_splitting_filename_cropped)
plot_geometry_pyvista(mesh_diel_splitting, geom_diel_splitting_filename_raw, material_type="dielectric", case="inelastic")
crop_white_margins(geom_diel_splitting_filename_raw, geom_diel_splitting_filename_cropped)

# Starting Point Geometries - NEW
plot_geometry_pyvista(mesh_metal_init, geom_metal_init_filename_raw, material_type="metal", case="isotropic")
crop_white_margins(geom_metal_init_filename_raw, geom_metal_init_filename_cropped)
plot_geometry_pyvista(mesh_diel_init, geom_diel_init_filename_raw, material_type="dielectric", case="isotropic")
crop_white_margins(geom_diel_init_filename_raw, geom_diel_init_filename_cropped)

# --- 3. Create Final Figure (CairoMakie) ---
println("Generating final Figure...")
try
    # Load ALL cropped images
    img_metal_nominal_geom = isfile(geom_metal_nominal_filename_cropped) ? load(geom_metal_nominal_filename_cropped) : nothing
    img_metal_anisotropic_geom = isfile(geom_metal_anisotropic_filename_cropped) ? load(geom_metal_anisotropic_filename_cropped) : nothing
    img_diel_nominal_geom = isfile(geom_diel_nominal_filename_cropped) ? load(geom_diel_nominal_filename_cropped) : nothing
    img_diel_anisotropic_geom = isfile(geom_diel_anisotropic_filename_cropped) ? load(geom_diel_anisotropic_filename_cropped) : nothing
    # NEW Splitting Images
    img_metal_splitting_geom = isfile(geom_metal_splitting_filename_cropped) ? load(geom_metal_splitting_filename_cropped) : nothing
    img_diel_splitting_geom = isfile(geom_diel_splitting_filename_cropped) ? load(geom_diel_splitting_filename_cropped) : nothing
    # NEW Starting Point Images
    img_metal_init_geom = isfile(geom_metal_init_filename_cropped) ? load(geom_metal_init_filename_cropped) : nothing
    img_diel_init_geom = isfile(geom_diel_init_filename_cropped) ? load(geom_diel_init_filename_cropped) : nothing

    if any(isnothing, [img_metal_nominal_geom, img_metal_anisotropic_geom, img_diel_nominal_geom, img_diel_anisotropic_geom, img_metal_splitting_geom, img_diel_splitting_geom, img_metal_init_geom, img_diel_init_geom])
        error("Failed to load one or more cropped geometry images.")
    end

    # Define color and linestyle mappings
    # Purple shades for metal (based on #b097d1)
    color_3d_metal = "#b097d1"  # Base purple
    color_2d_metal = "#b097d1"  # Base purple for isotropic
    color_metal_anisotropic = "#8b7db1"  # More blue-purple for anisotropic
    color_metal_iso2ani = "#2a1b3d"  # Much darker purple for iso2ani (almost black)
    color_metal_inelastic = "#9db1a8"  # More green-purple for inelastic

    # Red shades for dielectric (based on #D7263D)
    color_3d_dielectric = "#D7263D"  # Base red
    color_2d_dielectric = "#D7263D"  # Base red for isotropic
    color_diel_anisotropic = "#FF6B35"  # More orange-red for anisotropic
    color_diel_iso2ani = "#1a0f1a"  # Very dark purple-red for iso2ani
    color_diel_inelastic = "#B5179E"  # More purple-red for inelastic

    ls_isotropic = :solid
    ls_elastic = :solid
    ls_monopolarized = :solid
    ls_bipolarized = :dashdot
    ls_anisotropic = :dash
    ls_inelastic = :dot
    ls_dashdotdot = :dashdotdot  # New linestyle for ani2iso

    # Define parameters for line thicknesses and font sizes
    linewidth_main = 6.0  # Increased from 5.0 by 20%
    linewidth_secondary = 4.8  # Increased from 4.0 by 20%
    linewidth_indicator = 4.8  # Increased from 4.0 by 20%
    linewidth_pump = 6.0  # Increased from 5.0 by 20%
    linewidth_vertical = 3.6  # Increased from 3.0 by 20%

    fontsize_labels = 24 # Standardized font size
    fontsize_axis = 24 # Standardized font size
    fontsize_title = 24 # Standardized font size

    # --- Figure Setup (2 columns: left = 2 stacked plots, right = 3x2 grid) ---
    fig = Figure(resolution=(1200, 800), fontsize=fontsize_axis, font="Times")

    # --- Left Column: Spectral Plots stacked vertically ---
    ax_spectral_anisotropy = Axis(fig[1, 1],
        xlabel="Emission Wavelength λₑ (nm)",
        ylabel="Raman Enhancement",
        yscale=log10,
        title="(a) Anisotropy Effects",
        xlabelsize=fontsize_axis,
        ylabelsize=fontsize_axis,
        titlesize=fontsize_title,
        titlefont="Times",
        xgridwidth=2, ygridwidth=2, xgridcolor=(:black, 0.20), ygridcolor=(:black, 0.20))

    # Plot Metal Anisotropy Data (labels updated)
    # Plot main cases first (isotropic and anisotropic)
    v1 = lines!(ax_spectral_anisotropy, bonus_spectral_metal_iso["wavelengths"], bonus_spectral_metal_iso["g_y"], label="Metal: Isotropic", linewidth=linewidth_main, color=COLOR_METAL_ISOTROPIC, linestyle=ls_isotropic)
    # v4 = lines!(ax_spectral_anisotropy, bonus_spectral_metal_iso["wavelengths"], bonus_spectral_metal_iso["g_y"], label="Metal: Iso Geom->Ani Phys", linewidth=linewidth_main, color=COLOR_METAL_ISO2ANI, linestyle=ls_bipolarized)
    v2 = lines!(ax_spectral_anisotropy, bonus_spectral_metal_aniso["wavelengths"], bonus_spectral_metal_aniso["g_y"], label="Metal: Anisotropic", linewidth=linewidth_main, color=COLOR_METAL_ANISOTROPIC, linestyle=ls_anisotropic)

    # Plot Dielectric Anisotropy Data (labels updated)
    # Plot main cases first (isotropic and anisotropic)
    lines!(ax_spectral_anisotropy, spectral_diel_iso2ani["wavelengths"], spectral_diel_iso2ani["g_y"], label="Dielectric: Isotropic", linewidth=linewidth_secondary, color=COLOR_DIEL_ISOTROPIC, linestyle=ls_isotropic)
    # lines!(ax_spectral_anisotropy, spectral_diel_iso2ani["wavelengths"], spectral_diel_iso2ani["g_y"], label="Dielectric: Iso Geom->Ani Phys", linewidth=linewidth_secondary, color=COLOR_DIEL_ISO2ANI, linestyle=ls_bipolarized)
    lines!(ax_spectral_anisotropy, spectral_diel_anisotropic["wavelengths"], spectral_diel_anisotropic["g_y"], label="Dielectric: Anisotropic", linewidth=linewidth_secondary, color=COLOR_DIEL_ANISOTROPIC, linestyle=ls_anisotropic)

    # Add labels next to lines in the left subplot (a)
    # Metal labels
    text!(ax_spectral_anisotropy, 495 - 2, 10^4.2, text="Metal: Isotropic", color=COLOR_METAL_ISOTROPIC, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    lines!(ax_spectral_anisotropy, [485 - 2, 500 - 3], [0.6 * 10^4.2, 0.6 * 10^4.2], color=COLOR_METAL_ISOTROPIC, linestyle=ls_isotropic, linewidth=linewidth_indicator, alpha=1.0)

    # text!(ax_spectral_anisotropy, 570+2, 10^4.2, text="Metal: Iso→Ani", color=COLOR_METAL_ISO2ANI, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    # lines!(ax_spectral_anisotropy, [560+2, 580-2], [0.6*10^4.2, 0.6*10^4.2], color=COLOR_METAL_ISO2ANI, linestyle=ls_bipolarized, linewidth=linewidth_indicator, alpha=1.0)

    text!(ax_spectral_anisotropy, 555, 10^3.0, text="Metal: Anisotropic", color=COLOR_METAL_ANISOTROPIC, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    lines!(ax_spectral_anisotropy, [545, 565], [0.6 * 10^3.0, 0.6 * 10^3.0], color=COLOR_METAL_ANISOTROPIC, linestyle=ls_anisotropic, linewidth=linewidth_indicator, alpha=1.0)

    # Dielectric labels
    text!(ax_spectral_anisotropy, 510 - 3, 10^2.6, text="Dielectric: Isotropic", color=COLOR_DIEL_ISOTROPIC, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    lines!(ax_spectral_anisotropy, [500 - 3, 520 - 3], [0.6 * 10^2.6, 0.6 * 10^2.6], color=COLOR_DIEL_ISOTROPIC, linestyle=ls_isotropic, linewidth=linewidth_indicator, alpha=1.0)

    text!(ax_spectral_anisotropy, 570 - 4, 10^1.8, text="Dielectric: Anisotropic", color=COLOR_DIEL_ANISOTROPIC, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    lines!(ax_spectral_anisotropy, [560 - 3, 580 - 3], [0.6 * 10^1.8, 0.6 * 10^1.8], color=COLOR_DIEL_ANISOTROPIC, linestyle=ls_anisotropic, linewidth=linewidth_indicator, alpha=1.0)

    # text!(ax_spectral_anisotropy, 497, 10^1.5, text="Dielectric: Iso→Ani", color=COLOR_DIEL_ISO2ANI, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    # lines!(ax_spectral_anisotropy, [488-5, 500-5], [0.6*10^1.5, 0.6*10^1.5], color=COLOR_DIEL_ISO2ANI, linestyle=ls_bipolarized, linewidth=linewidth_indicator, alpha=1.0)

    # Add pump wavelength label with thicker line
    text!(ax_spectral_anisotropy, 548, 10^0.7, text="Pump λₚ=532nm", color=COLOR_PUMP, fontsize=fontsize_labels, align=(:center, :center), font="Times", alpha=1.0)
    lines!(ax_spectral_anisotropy, [535, 545], [0.6 * 10^0.7, 0.6 * 10^0.7], color=COLOR_PUMP, linestyle=:dash, linewidth=linewidth_pump, alpha=1.0)

    v5 = vlines!(ax_spectral_anisotropy, [wavelength_target_nominal], color=COLOR_PUMP, linestyle=:dash, linewidth=linewidth_vertical)

    # Add circles at 532nm for each type
    idx_532_metal_iso = argmin(abs.(bonus_spectral_metal_iso["wavelengths"] .- 532))
    idx_532_metal_aniso = argmin(abs.(bonus_spectral_metal_aniso["wavelengths"] .- 532))
    idx_532_dielectric = argmin(abs.(spectral_diel_nominal["wavelengths"] .- 532))

    # Metal Isotropic
    # scatter!(ax_spectral_anisotropy, [532], [spectral_metal_nominal["g_y"][idx_532]], 
    #     color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=COLOR_METAL_ISOTROPIC)
    lines!(ax_spectral_anisotropy, [532, 507], [bonus_spectral_metal_iso["g_y"][idx_532_metal_iso], 10^4.3],
        color=COLOR_METAL_ISOTROPIC, linestyle=:dash, linewidth=2)

    # Metal Anisotropic
    # scatter!(ax_spectral_anisotropy, [532], [spectral_metal_anisotropic["g_y"][idx_532]], 
    #     color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=COLOR_METAL_ANISOTROPIC)
    lines!(ax_spectral_anisotropy, [532, 542], [bonus_spectral_metal_aniso["g_y"][idx_532_metal_aniso], 10^3.2],
        color=COLOR_METAL_ANISOTROPIC, linestyle=:dash, linewidth=2)

    # # Metal Iso2Ani
    # scatter!(ax_spectral_anisotropy, [532], [spectral_metal_iso2ani["g_y"][idx_532]], 
    #     color=:white, markersize=12, marker=:circle, strokewidth=2, strokecolor=COLOR_METAL_ISO2ANI)
    # lines!(ax_spectral_anisotropy, [532, 557], [spectral_metal_iso2ani["g_y"][idx_532], 10^4.2], 
    #     color=COLOR_METAL_ISO2ANI, linestyle=:dash, linewidth=2)

    # Dielectric Isotropic
    # scatter!(ax_spectral_anisotropy, [532], [spectral_diel_nominal["g_y"][idx_532]], 
    #     color=(:white,0.5), markersize=30, marker=:circle, strokewidth=1, strokecolor=COLOR_DIEL_ISOTROPIC)
    lines!(ax_spectral_anisotropy, [532, 525], [spectral_diel_nominal["g_y"][idx_532_dielectric], 10^2.6],
        color=COLOR_DIEL_ISOTROPIC, linestyle=:dash, linewidth=2)

    # Dielectric Anisotropic
    # scatter!(ax_spectral_anisotropy, [532], [spectral_diel_anisotropic["g_y"][idx_532]], 
    #     color=(:white,0.5), markersize=20, marker=:circle, strokewidth=1, strokecolor=COLOR_DIEL_ANISOTROPIC)
    lines!(ax_spectral_anisotropy, [532, 552 - 7], [spectral_diel_anisotropic["g_y"][idx_532_dielectric], 10^1.9],
        color=COLOR_DIEL_ANISOTROPIC, linestyle=:dash, linewidth=2)

    # # Dielectric Iso2Ani
    # scatter!(ax_spectral_anisotropy, [532], [spectral_diel_iso2ani["g_y"][idx_532]], 
    #     color=(:white,0.5), markersize=12, marker=:circle, strokewidth=1, strokecolor=COLOR_DIEL_ISO2ANI)
    # lines!(ax_spectral_anisotropy, [532, 515], [spectral_diel_iso2ani["g_y"][idx_532], 10^1.5], 
    #     color=COLOR_DIEL_ISO2ANI, linestyle=:dash, linewidth=2)

    # --- Bottom Left: Inelastic (Splitting) Spectral Plot ---
    ax_spectral_inelastic = Axis(fig[2, 1],
        xlabel="Emission Wavelength λₑ (nm)",
        ylabel="Raman Enhancement",
        yscale=log10,
        title="(b) Inelastic Effects",
        xlabelsize=fontsize_axis,
        ylabelsize=fontsize_axis,
        titlesize=fontsize_title,
        titlefont="Times",
        xgridwidth=2, ygridwidth=2, xgridcolor=(:black, 0.20), ygridcolor=(:black, 0.20))

    # Plot Metal Inelastic Data (labels updated)
    v11 = lines!(ax_spectral_inelastic, spectral_metal_nominal["wavelengths"], spectral_metal_nominal["g_y"], label="Metal: Isotropic", linewidth=linewidth_main, color=color_2d_metal, linestyle=ls_isotropic)
    v12 = lines!(ax_spectral_inelastic, spectral_metal_splitting["wavelengths"], spectral_metal_splitting["g_y"], label="Metal: Inelastic", linewidth=linewidth_main, color=color_metal_inelastic, linestyle=ls_inelastic)

    # Plot Dielectric Inelastic Data (labels updated)
    lines!(ax_spectral_inelastic, spectral_diel_nominal["wavelengths"], spectral_diel_nominal["g_y"], label="Dielectric: Isotropic", linewidth=linewidth_secondary, color=color_2d_dielectric, linestyle=ls_isotropic)
    lines!(ax_spectral_inelastic, spectral_diel_splitting["wavelengths"], spectral_diel_splitting["g_y"], label="Dielectric: Inelastic", linewidth=linewidth_secondary, color=color_diel_inelastic, linestyle=ls_inelastic)

    # Add labels next to lines in the inelastic plot (b)
    # Metal labels
    text!(ax_spectral_inelastic, 495, 10^4.2, text="Elastic Metal", color=color_2d_metal, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    lines!(ax_spectral_inelastic, [485, 500], [0.6 * 10^4.2, 0.6 * 10^4.2], color=color_2d_metal, linestyle=ls_isotropic, linewidth=linewidth_indicator, alpha=1.0)

    text!(ax_spectral_inelastic, 575, 10^4.2, text="Inelastic Metal", color=color_metal_inelastic, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    lines!(ax_spectral_inelastic, [565, 585], [0.6 * 10^4.2, 0.6 * 10^4.2], color=color_metal_inelastic, linestyle=ls_inelastic, linewidth=linewidth_indicator, alpha=1.0)

    # Dielectric labels
    text!(ax_spectral_inelastic, 510, 10^1.8, text="Elastic Dielectric", color=color_2d_dielectric, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    lines!(ax_spectral_inelastic, [500, 520], [0.6 * 10^1.8, 0.6 * 10^1.8], color=color_2d_dielectric, linestyle=ls_isotropic, linewidth=linewidth_indicator, alpha=1.0)

    text!(ax_spectral_inelastic, 570, 10^1.8, text="Inelastic Dielectric", color=color_diel_inelastic, fontsize=fontsize_labels, align=(:center, :center), font="Times")
    lines!(ax_spectral_inelastic, [560, 580], [0.6 * 10^1.8, 0.6 * 10^1.8], color=color_diel_inelastic, linestyle=ls_inelastic, linewidth=linewidth_indicator, alpha=1.0)

    # Add pump and emission wavelength labels with thicker lines
    text!(ax_spectral_inelastic, 520 - 5, 10^-0.3, text="Pump λₚ=532nm", color="#1B3B6F", fontsize=fontsize_labels, align=(:center, :center), font="Times", alpha=1.0)
    lines!(ax_spectral_inelastic, [510 - 4, 520 - 4], [0.6 * 10^-0.4, 0.6 * 10^-0.4], color="#1B3B6F", linestyle=:dash, linewidth=linewidth_pump, alpha=1.0)

    text!(ax_spectral_inelastic, 565 + 4, 10^-0.3, text="Emission λₑ=549nm", color=:green, fontsize=fontsize_labels, align=(:center, :center), font="Times", alpha=1.0)
    lines!(ax_spectral_inelastic, [555 + 3, 565 + 3], [0.6 * 10^-0.4, 0.6 * 10^-0.4], color=:green, linestyle=:dash, linewidth=linewidth_pump, alpha=1.0)

    v13 = vlines!(ax_spectral_inelastic, [wavelength_target_pump_split], color="#1B3B6F", linestyle=:dash, linewidth=linewidth_vertical)
    v14 = vlines!(ax_spectral_inelastic, [wavelength_target_emission_split], color=:green, linestyle=:dash, linewidth=linewidth_vertical)

    # Add circles at 532nm and 549nm for both elastic and inelastic cases
    idx_532 = argmin(abs.(spectral_metal_nominal["wavelengths"] .- 532))
    idx_549 = argmin(abs.(spectral_metal_nominal["wavelengths"] .- 549))

    # Elastic Metal (at 532nm)
    # scatter!(ax_spectral_inelastic, [532], [spectral_metal_nominal["g_y"][idx_532]], 
    #     color=(:white,0.5), markersize=22, marker=:circle, strokewidth=2, strokecolor=color_2d_metal)
    lines!(ax_spectral_inelastic, [532, 507], [spectral_metal_nominal["g_y"][idx_532], 10^4.2],
        color=color_2d_metal, linestyle=:dash, linewidth=2)

    # Inelastic Metal (at 549nm)
    # scatter!(ax_spectral_inelastic, [549], [spectral_metal_splitting["g_y"][idx_549]], 
    #     color=(:white,0.5), markersize=22, marker=:circle, strokewidth=2, strokecolor=color_metal_inelastic)
    lines!(ax_spectral_inelastic, [549, 562], [spectral_metal_splitting["g_y"][idx_549], 10^4.2],
        color=color_metal_inelastic, linestyle=:dash, linewidth=2)

    # Elastic Dielectric (at 532nm)
    # scatter!(ax_spectral_inelastic, [532], [spectral_diel_nominal["g_y"][idx_532]], 
    #     color=(:white,0.5), markersize=22, marker=:circle, strokewidth=2, strokecolor=color_2d_dielectric)
    lines!(ax_spectral_inelastic, [532, 525], [spectral_diel_nominal["g_y"][idx_532], 10^2.1],
        color=color_2d_dielectric, linestyle=:dash, linewidth=2)

    # Inelastic Dielectric (at 549nm)
    # scatter!(ax_spectral_inelastic, [549], [spectral_diel_splitting["g_y"][idx_549]], 
    #     color=(:white,0.5), markersize=22, marker=:circle, strokewidth=2, strokecolor=color_diel_inelastic)
    lines!(ax_spectral_inelastic, [549, 553], [spectral_diel_splitting["g_y"][idx_549], 10^1.8],
        color=color_diel_inelastic, linestyle=:dash, linewidth=2)

    # --- Right Column: 3x3 grid of geometries (3 rows, 3 columns) ---
    geom_grid = fig[1:2, 2] = GridLayout(4, 2)
    geom_aspect = DataAspect()

    # Row 1: Anisotropic
    # ax_geom_left1 = Axis(geom_grid[1, 1], aspect=geom_aspect); image!(ax_geom_left1, rotr90(img_metal_nominal_geom)); hidedecorations!(ax_geom_left1); hidespines!(ax_geom_left1)
    ax_geom_m_ani = Axis(geom_grid[1, 1], aspect=geom_aspect)
    image!(ax_geom_m_ani, rotr90(img_metal_anisotropic_geom))
    hidedecorations!(ax_geom_m_ani)
    hidespines!(ax_geom_m_ani)
    xlims!(ax_geom_m_ani, 0, 390)
    ylims!(ax_geom_m_ani, 0, 390)
    ax_geom_d_ani = Axis(geom_grid[1, 2], aspect=geom_aspect)
    image!(ax_geom_d_ani, rotr90(img_diel_anisotropic_geom))
    hidedecorations!(ax_geom_d_ani)
    hidespines!(ax_geom_d_ani)
    xlims!(ax_geom_d_ani, 0, 410)
    ylims!(ax_geom_d_ani, 0, 410)
    # Row 2: Isotropic
    ax_geom_m_iso = Axis(geom_grid[2, 1], aspect=geom_aspect)
    image!(ax_geom_m_iso, rotr90(img_metal_nominal_geom))
    hidedecorations!(ax_geom_m_iso)
    hidespines!(ax_geom_m_iso)
    ax_geom_d_iso = Axis(geom_grid[3, 2], aspect=geom_aspect)
    image!(ax_geom_d_iso, rotr90(img_diel_nominal_geom))
    hidedecorations!(ax_geom_d_iso)
    hidespines!(ax_geom_d_iso)
    ax_geom_m_iso_ = Axis(geom_grid[2, 2], aspect=geom_aspect)
    image!(ax_geom_m_iso_, rotr90(img_metal_init_geom))
    hidedecorations!(ax_geom_m_iso_)
    hidespines!(ax_geom_m_iso_)
    xlims!(ax_geom_m_iso_, 0, 390)
    ylims!(ax_geom_m_iso_, 0, 390)
    ax_geom_d_iso_ = Axis(geom_grid[3, 1], aspect=geom_aspect)
    image!(ax_geom_d_iso_, rotr90(img_diel_init_geom))
    hidedecorations!(ax_geom_d_iso_)
    hidespines!(ax_geom_d_iso_)
    xlims!(ax_geom_d_iso_, 0, 410)
    ylims!(ax_geom_d_iso_, 0, 410)
    # Row 3: Inelastic
    # Leave leftmost cell empty for now (or could add another placeholder if desired)
    ax_geom_m_inel = Axis(geom_grid[4, 1], aspect=geom_aspect)
    image!(ax_geom_m_inel, rotr90(img_metal_splitting_geom))
    hidedecorations!(ax_geom_m_inel)
    hidespines!(ax_geom_m_inel)
    xlims!(ax_geom_m_inel, 0, 390)
    ylims!(ax_geom_m_inel, 0, 390)
    ax_geom_d_inel = Axis(geom_grid[4, 2], aspect=geom_aspect)
    image!(ax_geom_d_inel, rotr90(img_diel_splitting_geom))
    hidedecorations!(ax_geom_d_inel)
    hidespines!(ax_geom_d_inel)
    xlims!(ax_geom_d_inel, 0, 410)
    ylims!(ax_geom_d_inel, 0, 410)

    # --- Add overlay Axis for arrows ---
    # This axis will use normalized coordinates (0-1 for x and y) spanning fig[1:2, 2]
    ax_overlay = Axis(fig[1:2, 2]) # Create in the same position as geom_grid
    hidedecorations!(ax_overlay)   # Hide ticks, labels, etc.
    hidespines!(ax_overlay)       # Hide the border spines
    xlims!(ax_overlay, 0, 1) # Set x-limits to [0, 1]
    ylims!(ax_overlay, 0, 1) # Set y-limits to [0, 1]

    # Define arrow properties
    arrow_size = 15 # Adjust as needed, can be a single number or Point2f for x/y sizes
    line_width = 2  # Adjust as needed

    # Normalized coordinates for cell centers within the 3x3 geom_grid.
    # The geom_grid has N_row=3 rows and N_col=3 columns.
    # x_center(col_idx) = (col_idx - 0.5) / N_col
    # y_center(row_idx) = (N_row - row_idx + 0.5) / N_row (where row_idx 1 is top, N_row is bottom)
    N_col = 2
    N_row = 4

    # --- Purple arrows (metal column) ---
    # Source: metal isotropic, geom_grid[row=2, col=2]
    metal_source_col_idx = 2
    metal_source_row_idx = 2
    metal_start_point = Point2f((metal_source_col_idx - 0.5) / (N_col + 0.6), (N_row - metal_source_row_idx + 0.5) / N_row)

    # Destinations for metal arrows (all in column metal_source_col_idx):
    # 1. metal anisotropic: geom_grid[row=1, col=1]
    # 2. metal isotropic:   geom_grid[row=2, col=1] (self)
    # 3. metal inelastic:   geom_grid[row=4, col=1]
    metal_target_row_indices = [1, 2, 4]
    metal_target_points = [
        Point2f((metal_source_col_idx - 0.5) / (N_col + 1.5), (N_row - r_idx + 0.5) / N_row)
        for r_idx in metal_target_row_indices
    ]

    metal_origins = [metal_start_point for _ in 1:length(metal_target_points)]
    metal_directions = [target - metal_start_point for target in metal_target_points]

    arrows!(ax_overlay, metal_origins, metal_directions,
        color=:purple, arrowsize=arrow_size, linewidth=line_width)


    metal_target_row_indices = [1, 3, 4]
    metal_start_point = Point2f((metal_source_col_idx - 0.5) / (N_col + 1.5), (N_row - metal_source_row_idx - 1 + 0.125) / (N_row - 1))
    metal_target_points = [
        Point2f((metal_source_col_idx - 0.5) / (N_col + 0.6), (N_row - r_idx + 0.5) / N_row)
        for r_idx in metal_target_row_indices
    ]

    diel_origins = [metal_start_point for _ in 1:length(metal_target_points)]
    diel_directions = [target - metal_start_point for target in metal_target_points]

    # White lines underneath arrows
    arrows!(ax_overlay, diel_origins, diel_directions,
        color=:white, arrowsize=arrow_size * 2, linewidth=line_width * 5)

    arrows!(ax_overlay, diel_origins, diel_directions,
        color=COLOR_DIEL_ISOTROPIC, arrowsize=arrow_size, linewidth=line_width)

    # Add direct labels to each geometry (panel c) with underlines matching color and linestyle
    # Anisotropic Metal
    text!(ax_geom_m_ani, 0.05, 1.0, text="Elastic Anisotropic\n Metal", color=color_metal_anisotropic, fontsize=19, space=:relative, align=(:left, :top), font="Times")
    lines!(ax_geom_m_ani, [0.05, 0.25], [0.72, 0.72], color=color_metal_anisotropic, linestyle=ls_anisotropic, linewidth=4, space=:relative)
    # Anisotropic Dielectric
    text!(ax_geom_d_ani, 0.05, 1.0, text="Elastic Anisotropic\n Dielectric", color=color_diel_anisotropic, fontsize=19, space=:relative, align=(:left, :top), font="Times")
    lines!(ax_geom_d_ani, [0.05, 0.25], [0.72, 0.72], color=color_diel_anisotropic, linestyle=ls_anisotropic, linewidth=4, space=:relative)
    # Isotropic Metal
    text!(ax_geom_m_iso, 0.05, 1.0, text="Isotropic Elastic\n Metal", color=color_2d_metal, fontsize=19, space=:relative, align=(:left, :top), font="Times")
    lines!(ax_geom_m_iso, [0.05, 0.25], [0.72, 0.72], color=color_2d_metal, linestyle=ls_isotropic, linewidth=4, space=:relative)
    # Isotropic Dielectric
    text!(ax_geom_d_iso, 0.05, 1.0, text="Isotropic Elastic\n Dielectric", color=color_2d_dielectric, fontsize=19, space=:relative, align=(:left, :top), font="Times")
    lines!(ax_geom_d_iso, [0.05, 0.25], [0.72, 0.72], color=color_2d_dielectric, linestyle=ls_isotropic, linewidth=4, space=:relative)
    # Inelastic Metal
    text!(ax_geom_m_inel, 0.05, 1.0, text="Isotropic Inelastic\n Metal", color=color_metal_inelastic, fontsize=19, space=:relative, align=(:left, :top), font="Times")
    lines!(ax_geom_m_inel, [0.05, 0.25], [0.72, 0.72], color=color_metal_inelastic, linestyle=ls_inelastic, linewidth=4, space=:relative)
    # Inelastic Dielectric
    text!(ax_geom_d_inel, 0.05, 1.0, text="Isotropic Inelastic\n Dielectric", color=color_diel_inelastic, fontsize=19, space=:relative, align=(:left, :top), font="Times")
    lines!(ax_geom_d_inel, [0.05, 0.25], [0.72, 0.72], color=color_diel_inelastic, linestyle=ls_inelastic, linewidth=4, space=:relative)

    # Add labels above the initial geometries in the geometry grid
    text!(ax_geom_d_iso_, 0.05, 1.0, text="Dielectric \nEpoch #1-2", color=color_2d_dielectric, fontsize=19, space=:relative, align=(:left, :top), font="Times")
    text!(ax_geom_m_iso_, 0.05, 1.0, text="Metal Epoch \n#1-2", color=color_2d_metal, fontsize=19, space=:relative, align=(:left, :top), font="Times")

    # Add label along the arrow from dielectric base (geom_grid[3,3]) to the topmost dielectric (geom_grid[1,3])
    # Compute relative coordinates for the arrow direction (vertical, rightmost column)
    # Place the label just above the arrow tip
    text!(ax_overlay, 0.53, 0.5, text="Epochs #3-4", color=:black, fontsize=22, font="Times", align=(:center, :bottom), rotation=pi / 2 * 0.85)

    # Draw arrows from the isotropic (starting point) to the other geometries
    # Get axis centers in normalized coordinates (space=:relative)
    arrow_color_metal = color_2d_metal
    arrow_color_diel = color_2d_dielectric
    arrow_lw = 4
    arrow_headsize = 18

    # ax_arrows = Axis(geom_grid[1:2, 2], aspect=geom_aspect)
    # hidedecorations!(ax_arrows)
    # strt = ax_geom_m_iso_.scene.viewport[].origin
    # nd = ax_geom_m_iso.scene.viewport[].origin
    # # arrows!(ax_arrows, [strt[1]], [strt[2]], [nd[1] - strt[1]], [nd[2] - strt[2]],
    #     arrowsize=arrow_headsize, color=arrow_color_metal, linewidth=arrow_lw, space=:relative, lengthscale=0.25)

    # # Helper to draw an arrow between axes in the grid
    # function draw_arrow_between_axes!(ax_from, ax_to; color, lw, headsize)
    #     # Start at center right of ax_from, end at center left of ax_to
    #     arrows!(ax_from, [1.0], [0.5], [1.0], [0.0],
    #         arrowsize=headsize, color=color, linewidth=lw, space=:relative, lengthscale=0.25, axis_to=ax_to)
    # end

    # # Metal arrows (purple): from ax_geom_m_iso to anisotropic and inelastic
    # draw_arrow_between_axes!(ax_geom_m_iso, ax_geom_m_ani; color=arrow_color_metal, lw=arrow_lw, headsize=arrow_headsize)
    # draw_arrow_between_axes!(ax_geom_m_iso, ax_geom_m_iso; color=arrow_color_metal, lw=arrow_lw, headsize=arrow_headsize) # self-arrow (optional, can remove)
    # draw_arrow_between_axes!(ax_geom_m_iso, ax_geom_m_inel; color=arrow_color_metal, lw=arrow_lw, headsize=arrow_headsize)

    # # Dielectric arrows (red): from ax_geom_d_iso to anisotropic and inelastic
    # draw_arrow_between_axes!(ax_geom_d_iso, ax_geom_d_ani; color=arrow_color_diel, lw=arrow_lw, headsize=arrow_headsize)
    # draw_arrow_between_axes!(ax_geom_d_iso, ax_geom_d_iso; color=arrow_color_diel, lw=arrow_lw, headsize=arrow_headsize) # self-arrow (optional, can remove)
    # draw_arrow_between_axes!(ax_geom_d_iso, ax_geom_d_inel; color=arrow_color_diel, lw=arrow_lw, headsize=arrow_headsize)

    # colgap!(geom_grid, 5)
    # rowgap!(geom_grid, 5)

    # --- Comment out the legends ---
    # Legend(fig[2, 1:2], [v1, v2, v3, v4, v5], ["Isotropic", "Full Ani", "Ani Geom->Iso Phys", "Iso Geom->Ani Phys", "λₙ = $(Int(wavelength_target_nominal))nm"], orientation=:vertical, tellheight=true, nbanks=1)
    # legend_elements_inelastic = [v11, v12, v13, v14]
    # legend_labels_inelastic = ["Isotropic", "Inelastic", "λₚ = $(Int(wavelength_target_pump_split))nm", "λₑ = $(Int(wavelength_target_emission_split))nm"]
    # Legend(fig[2, 5:6], legend_elements_inelastic, legend_labels_inelastic, orientation=:vertical, tellheight=true, nbanks=1)

    # Adjust column and row sizes for new layout
    colsize!(fig.layout, 2, Relative(0.45))
    # colsize!(fig.layout, 2, Relative(0.55))
    # rowsize!(fig.layout, 1, Relative(0.33))
    # rowsize!(fig.layout, 2, Relative(0.33))
    # rowsize!(fig.layout, 3, Relative(0.34))

    # Save the final figure
    save(final_figure_filename, fig, px_per_unit=2)
    println("Saved final combined Figure to $final_figure_filename")

catch e
    @error "Failed to generate Figure: $e"
    showerror(stdout, e, catch_backtrace()) # Print stack trace
end

println("Figure generation attempt complete.")

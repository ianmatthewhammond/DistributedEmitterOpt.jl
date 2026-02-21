module VisualizationExt

# ---------------------------------------------------------------------------
# Extension trigger deps (weak deps)
# ---------------------------------------------------------------------------
using PyCall
using Images
using Colors
using ColorSchemes
using FileIO
using CSV
using DataFrames
using LaTeXStrings

# ---------------------------------------------------------------------------
# Core deps (already in [deps])
# ---------------------------------------------------------------------------
using Gridap
using JLD2
using Statistics
using CairoMakie

# ---------------------------------------------------------------------------
# Parent package
# ---------------------------------------------------------------------------
using DistributedEmitterOpt

export vperiodicdesign, visualize, visualizepost, visualize_new,
    plot_material, plot_field, plot_substrate, get_figure_data,
    combine_figures, add_text!, load_bid_parameters,
    plot_directionals, plot_geometrics, plot_geometrics_one_only, plot_tolerance,
    crop_white_margins, brighten_image!, hex_to_rgba, save_geometry_snapshot, plot_field_slices,
    get_spectral_data

# ============================================================================
# FigureLoader
# ============================================================================

# Hardcoded base path relative to the package root
const DATA_REL_PATH = joinpath("figures", "paper-figures-scripts", "data")

function _get_data_root()
    joinpath(pkgdir(DistributedEmitterOpt), DATA_REL_PATH)
end

"""
    get_figure_data(group_key::String, variation_key::String, file_key::String)

Loads data for figure generation. Replaces the legacy `get_data` and YAML config.
"""
function get_figure_data(group_key::String, variation_key::String, file_key::String)
    data_root = _get_data_root()

    local_path = ""

    if group_key == "Nonlinear"
        if file_key == "csv"
            local_path = joinpath(data_root, "Nonlinear", "csv-data", variation_key)
        elseif file_key == "geometry"
            local_path = joinpath(data_root, "Nonlinear", "geometries", variation_key)
        end
    elseif group_key == "Starting"
        local_path = joinpath(data_root, "data_starting", variation_key, "y_design.vtu")
    elseif group_key == "Bonus"
        local_path = joinpath(data_root, "post-anisotropy-bonus", variation_key)
    else
        filename = ""
        if file_key == "spectral"
            filename = "spectral.jld2"
        elseif file_key == "design_y"
            filename = "y_design.vtu"
        elseif file_key == "results"
            filename = "results.jld2"
        elseif file_key == "output"
            filename = "output.txt"
        elseif file_key == "fields_y"
            filename = "y_fields.vtu"
        elseif file_key == "fields_x"
            filename = "x_fields.vtu"
        else
            error("Unknown file_key: $file_key")
        end

        local_path = joinpath(data_root, group_key, variation_key, filename)
    end

    if !isfile(local_path)
        if group_key == "Constrained_Metal" && variation_key == "nominal" && file_key == "spectral"
            alt_path = joinpath(data_root, "post-anisotropy-bonus", "iso-spectral.jld2")
            if isfile(alt_path)
                println("Warning: Main file missing. Using fallback: $alt_path")
                local_path = alt_path
            else
                println("Warning: File not found at $local_path and fallback failed.")
                return nothing
            end
        else
            println("Warning: File not found at $local_path")
            return nothing
        end
    end

    # Load based on extension/type
    if endswith(local_path, ".jld2")
        data = load(local_path)

        if file_key == "spectral" && haskey(data, "single_stored_object")
            val = data["single_stored_object"]
            if isa(val, Tuple) && length(val) == 4
                return Dict("wavelengths" => val[1], "g_y" => val[2], "g_x" => val[3], "g_combined" => val[4])
            end
            return val
        elseif file_key == "results" && haskey(data, "single_stored_object")
            val = data["single_stored_object"]
            if isa(val, Tuple)
                res = Dict{String,Any}()
                if length(val) >= 2
                    res["design"] = val[2]
                end
                if length(val) >= 3
                    res["gradient"] = val[3]
                end
                if length(val) >= 4
                    res["g_array"] = val[4]
                end
                if length(val) >= 5
                    res["g_final"] = val[5]
                end
                if length(val) >= 6
                    res["p_biggest"] = val[6]
                end
                return res
            end
            return val
        end

        return data
    elseif endswith(local_path, ".vtu")
        try
            pv = pyimport("pyvista")
            return pv.read(local_path)
        catch e
            println("Error loading VTU with PyVista: $e")
            return nothing
        end
    elseif endswith(local_path, ".csv")
        return CSV.read(local_path, DataFrame)
    elseif endswith(local_path, ".txt")
        return readlines(local_path)
    end

    return nothing
end

# ============================================================================
# Visualization helpers
# ============================================================================

# Helper for E^4 objective visualization
sumabs4_uh2(uh²) = uh² * uh²

"""
Saves a png of a periodic design with both symmetries and with dimensions L, W and cell size M, N.
"""
function vperiodicdesign(; kwargs...)
    pyfile = joinpath(pkgdir(DistributedEmitterOpt), "src", "python", "periodic_design.py")
    keys = ["--$key" for (key, val) in kwargs]
    vals = [string(val) for (key, val) in kwargs]
    args = collect(Iterators.flatten(zip(keys, vals)))
    run(`/Applications/ParaView-5.11.0.app/Contents/bin/pvbatch $pyfile $args`)
end

function visualize(Ω, Ω_d, E², ptfe, filebase, geom_list, L, W, symmetry=true)

    # Init figure
    filebase = filebase * ".vtu"
    fig_width = 1600
    fig_height = 2400
    font_size = 40
    title_font_size = 50
    pv = pyimport("pyvista")
    plotter = pv.Plotter(shape=(1, 3), off_screen=true, window_size=(fig_width * 3, fig_height))

    # Write the entire region vtk
    writevtk(Ω, filebase;
        cellfields=[
            "uhabs4" => sumabs4_uh2 ∘ E²
        ])

    # Process mesh
    grid = pv.read(filebase)
    plotter.subplot(0, 0)
    plotter.add_mesh(grid, scalars="cell", style="wireframe", cmap="Greys", show_scalar_bar=false)
    plotter.show_axes()
    plotter.camera.zoom(1.5)
    region_names = ["Substrate", "Design", "Fluid", "Source", "Fluid"]
    geom_list = geom_list ./ geom_list[end]
    geom_list = geom_list[1:5]
    for (height, region_name) in zip(geom_list, region_names)
        plotter.add_text(region_name, position=(0.75 * fig_width, height * fig_height), font_size=font_size, color="black")
    end

    # Process fields
    plotter.subplot(0, 1)
    min, max = 0.0, 1.0
    for (key, val) in grid.point_data.items()
        if key == "uhabs4"
            max = percentile(val, 97.5)
        end
    end
    plotter.add_mesh(grid, scalars="uhabs4", cmap="coolwarm", clim=(min, max), scalar_bar_args=Dict("title" => "|E|^4", "vertical" => true, "position_x" => 0.05, "position_y" => 0.5, "label_font_size" => font_size, "title_font_size" => title_font_size), opacity="sigmoid")
    plotter.show_axes()
    plotter.camera.zoom(1.0)

    # Reflect
    if symmetry
        reflectx = pv.DataSetFilters.reflect(grid, (1, 0, 0), point=(-W / 2, 0, 0))
        reflecty = pv.DataSetFilters.reflect(grid, (0, 1, 0), point=(0, -L / 2, 0))
        reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2, 0, 0))
        plotter.add_mesh(reflectx, scalars="uhabs4", cmap="coolwarm", clim=(min, max), show_scalar_bar=false, opacity="sigmoid")
        plotter.add_mesh(reflecty, scalars="uhabs4", cmap="coolwarm", clim=(min, max), show_scalar_bar=false, opacity="sigmoid")
        plotter.add_mesh(reflectxy, scalars="uhabs4", cmap="coolwarm", clim=(min, max), show_scalar_bar=false, opacity="sigmoid")
    end

    # Write the design
    writevtk(Ω_d, filebase;
        cellfields=[
            "p" => ptfe,
        ])

    # Process the design
    grid = pv.read(filebase)
    plotter.add_mesh(grid, scalars="p", cmap="Greys", clim=(0.0, 1.0), scalar_bar_args=Dict("title" => "params", "vertical" => true, "position_x" => 0.85, "position_y" => 0.5, "label_font_size" => font_size, "title_font_size" => title_font_size), opacity="sigmoid")

    # Reflect
    if symmetry
        reflectx = pv.DataSetFilters.reflect(grid, (1, 0, 0), point=(-W / 2, 0, 0))
        reflecty = pv.DataSetFilters.reflect(grid, (0, 1, 0), point=(0, -L / 2, 0))
        reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2, 0, 0))
        plotter.add_mesh(reflectx, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
        plotter.add_mesh(reflecty, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
        plotter.add_mesh(reflectxy, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
    end

    # Process fields
    plotter.subplot(0, 2)
    plotter.add_mesh(grid, scalars="p", cmap="Greys", clim=(0.0, 1.0), scalar_bar_args=Dict("title" => "params", "vertical" => true, "position_x" => 0.85, "position_y" => 0.5), opacity="sigmoid")
    plotter.show_axes()
    plotter.camera.zoom(1)

    # Reflect
    reflectx = pv.DataSetFilters.reflect(grid, (1, 0, 0), point=(-W / 2, 0, 0))
    reflecty = pv.DataSetFilters.reflect(grid, (0, 1, 0), point=(0, -L / 2, 0))
    reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2, 0, 0))
    plotter.add_mesh(reflectx, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
    plotter.add_mesh(reflecty, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
    plotter.add_mesh(reflectxy, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")

    # Save image
    plotter.show(screenshot=replace(filebase, ".vtu" => ".png"))
    rm(filebase)
    replace(filebase, ".vtu" => ".png")
end

function visualizepost(root, L, W, symmetry=true)

    # Init figure
    fig_width = 1600
    fig_height = 2400
    font_size = 40
    title_font_size = 50
    pv = pyimport("pyvista")
    pv.start_xvfb()
    plotter = pv.Plotter(shape=(1, 3), off_screen=true, window_size=(fig_width * 3, fig_height))

    # Process mesh
    grid = pv.read(root * "fields.vtu")
    plotter.subplot(0, 0)
    plotter.add_mesh(grid, scalars="cell", style="wireframe", cmap="Greys", show_scalar_bar=false)
    plotter.show_axes()
    plotter.camera.zoom(1.5)

    # Process fields
    plotter.subplot(0, 1)
    min, max = 0.0, 1.0
    for (key, val) in grid.point_data.items()
        if key == "E4"
            max = percentile(val, 97.5)
        end
    end
    plotter.add_mesh(grid, scalars="E4", cmap="coolwarm", clim=(min, max), scalar_bar_args=Dict("title" => "|E|^4", "vertical" => true, "position_x" => 0.05, "position_y" => 0.5, "label_font_size" => font_size, "title_font_size" => title_font_size), opacity="sigmoid")
    plotter.show_axes()
    plotter.camera.zoom(1.0)

    # Reflect
    if symmetry
        reflectx = pv.DataSetFilters.reflect(grid, (1, 0, 0), point=(-W / 2, 0, 0))
        reflecty = pv.DataSetFilters.reflect(grid, (0, 1, 0), point=(0, -L / 2, 0))
        reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2, 0, 0))
        plotter.add_mesh(reflectx, scalars="E4", cmap="coolwarm", clim=(min, max), show_scalar_bar=false, opacity="sigmoid")
        plotter.add_mesh(reflecty, scalars="E4", cmap="coolwarm", clim=(min, max), show_scalar_bar=false, opacity="sigmoid")
        plotter.add_mesh(reflectxy, scalars="E4", cmap="coolwarm", clim=(min, max), show_scalar_bar=false, opacity="sigmoid")
    end

    # Process the design
    grid = pv.read(root * "design.vtu")
    plotter.add_mesh(grid, scalars="p", cmap="Greys", clim=(0.0, 1.0), scalar_bar_args=Dict("title" => "params", "vertical" => true, "position_x" => 0.85, "position_y" => 0.5, "label_font_size" => font_size, "title_font_size" => title_font_size), opacity="sigmoid")

    # Reflect
    if symmetry
        reflectx = pv.DataSetFilters.reflect(grid, (1, 0, 0), point=(-W / 2, 0, 0))
        reflecty = pv.DataSetFilters.reflect(grid, (0, 1, 0), point=(0, -L / 2, 0))
        reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2, 0, 0))
        plotter.add_mesh(reflectx, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
        plotter.add_mesh(reflecty, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
        plotter.add_mesh(reflectxy, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
    end

    # Process fields
    plotter.subplot(0, 2)
    plotter.add_mesh(grid, scalars="p", cmap="Greys", clim=(0.0, 1.0), scalar_bar_args=Dict("title" => "params", "vertical" => true, "position_x" => 0.85, "position_y" => 0.5), opacity="sigmoid")
    plotter.show_axes()
    plotter.camera.zoom(0.5)

    # Reflect
    reflectx = pv.DataSetFilters.reflect(grid, (1, 0, 0), point=(-W / 2, 0, 0))
    reflecty = pv.DataSetFilters.reflect(grid, (0, 1, 0), point=(0, -L / 2, 0))
    reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2, 0, 0))
    plotter.add_mesh(reflectx, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
    plotter.add_mesh(reflecty, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")
    plotter.add_mesh(reflectxy, scalars="p", cmap="Greys", clim=(0.0, 1.0), show_scalar_bar=false, opacity="sigmoid")

    # Save image
    plotter.show(screenshot=root * "post-three.png")
    root * "post-three.png"
end


function plot_material(np, pv, plotter, design, W, L; colorbar=true, title="\$\\rho(x)\$", font_size=20, title_font_size=24, num_periods_x=1, num_periods_y=1, ontop=false, design_field="p", contour=true, full=false, clim=(0.0, 1.0), opacity=nothing, reflectybool=true, flipx=false, flipy=false, contour_color="#AD97D8")
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
    contour_color = pv.Color(contour_color)
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
            if !full
                plotter.add_mesh(reflectx.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                if reflectybool
                    plotter.add_mesh(reflecty.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                    plotter.add_mesh(reflectxy.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity)
                end
            end
            if contour
                try
                    plotter.add_mesh(contours.translate((i * W * 2, j * L * 2, 0), inplace=false), opacity=1.0, clim=clim, color=contour_color, smooth_shading=true, specular=1.0, specular_power=128, ambient=0.0)
                    if !full
                        plotter.add_mesh(contoursx.translate((i * W * 2, j * L * 2, 0), inplace=false), opacity=1.0, clim=clim, color=contour_color, smooth_shading=true, specular=1.0, specular_power=128, ambient=0.0)
                        if reflectybool
                            plotter.add_mesh(contoursy.translate((i * W * 2, j * L * 2, 0), inplace=false), opacity=1.0, clim=clim, color=contour_color, smooth_shading=true, specular=1.0, specular_power=128, ambient=0.0)
                            plotter.add_mesh(contoursxy.translate((i * W * 2, j * L * 2, 0), inplace=false), opacity=1.0, clim=clim, color=contour_color, smooth_shading=true, specular=1.0, specular_power=128, ambient=0.0)
                        end
                    end
                catch e
                    println("Warning: Failed to plot contour for period ($i, $j): $e")
                end
            end
        end
    end

    return plotter
end


function plot_field(np, pv, plotter, field, W, L, Z_below; colorbar=true, title="\$|E|^2_y(x)\$", font_size=20, title_font_size=24, lim=5.0, num_periods_x=1, num_periods_y=1, vertical=false, full=false)
    plotter.show_axes()
    uhNorm = Any[nothing]
    for (key, val) in field.point_data.items()
        if key == "uhNorm"
            uhNorm[1] = np.square(val[1:end, 1]) + np.square(val[1:end, 2]) + np.square(val[1:end, 3])
            break
        end
    end
    if isnothing(uhNorm[1])
        throw(ArgumentError("Field data not found"))
    end

    field.point_data.set_scalars(uhNorm[1], "E2")
    reflectx = pv.DataSetFilters.reflect(field, (1, 0, 0), point=(-W / 2, 0, 0))
    reflecty = pv.DataSetFilters.reflect(field, (0, 1, 0), point=(0, -L / 2, 0))
    reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2, 0, 0))

    # Plot all periods
    for i in 0:1:num_periods_x-1
        for j in 0:1:num_periods_y-1
            # Field
            if (i == j) & (j == 0)
                position = (vertical) ? ["position_x" => 0.0, "position_y" => 0.35] : ["position_x" => 0.25, "position_y" => 0.8]
                plotter.add_mesh(field.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="E2", cmap="seismic", clim=(0.0, lim), scalar_bar_args=Dict(["title" => title, "vertical" => vertical, "label_font_size" => font_size, "title_font_size" => title_font_size, position...]), opacity="sigmoid", show_scalar_bar=colorbar)
            else
                plotter.add_mesh(field.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="E2", cmap="seismic", clim=(0.0, lim), opacity="sigmoid", show_scalar_bar=false)
            end
            if !full
                plotter.add_mesh(reflectx.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="E2", cmap="seismic", clim=(0.0, lim), show_scalar_bar=false, opacity="sigmoid")
                plotter.add_mesh(reflecty.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="E2", cmap="seismic", clim=(0.0, lim), show_scalar_bar=false, opacity="sigmoid")
                plotter.add_mesh(reflectxy.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="E2", cmap="seismic", clim=(0.0, lim), show_scalar_bar=false, opacity="sigmoid")
            end
        end
    end

    return plotter
end

function plot_substrate(np, pv, plotter, field, W, L, Z_below; num_periods_x=1, num_periods_y=1, full=false, flipx=false, flipy=false)
    plotter.show_axes()
    sub = Any[nothing]
    for (key, val) in field.point_data.items()
        if key == "E4"
            sub[1] = ones(size(val))
            break
        end
    end
    if isnothing(sub[1])
        throw(ArgumentError("Field data not found"))
    end
    field.point_data.set_scalars(sub[1], "sub")

    clipped = field.clip(origin=(0, 0, Z_below), normal="z")
    signx = (flipx) ? -1 : 1
    signy = (flipy) ? -1 : 1
    reflectxsub = pv.DataSetFilters.reflect(clipped, (1, 0, 0), point=(-W / 2 * signx, 0, 0))
    reflectysub = pv.DataSetFilters.reflect(clipped, (0, 1, 0), point=(0, -L / 2 * signy, 0))
    reflectxysub = pv.DataSetFilters.reflect(reflectysub, (1, 0, 0), point=(-W / 2 * signx, 0, 0))

    # Plot all periods
    for i in 0:1:num_periods_x-1
        for j in 0:1:num_periods_y-1
            plotter.add_mesh(clipped.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="sub", cmap="Greys", clim=(0.0, 1.35), opacity="sigmoid", show_scalar_bar=false)
            if !full
                plotter.add_mesh(reflectxsub.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="sub", cmap="Greys", clim=(0.0, 1.35), show_scalar_bar=false, opacity="sigmoid")
                plotter.add_mesh(reflectysub.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="sub", cmap="Greys", clim=(0.0, 1.35), show_scalar_bar=false, opacity="sigmoid")
                plotter.add_mesh(reflectxysub.translate((i * W * 2, j * L * 2, 0), inplace=false), scalars="sub", cmap="Greys", clim=(0.0, 1.35), show_scalar_bar=false, opacity="sigmoid")
            end
        end
    end

    return plotter
end


function visualize_new(root, L, W, des_low, des_high, foundry)

    vtk = pyimport("vtk")
    np = pyimport("numpy")
    pv = pyimport("pyvista")
    pv.start_xvfb()

    fig_width = 2600 * 2
    fig_height = 600 * 2
    font_size = 40
    title_font_size = 50
    plotter = pv.Plotter(shape=(1, 4), off_screen=true, window_size=(fig_width, fig_height))

    design = pv.read(root * "y_design.vtu")
    y_fields = pv.read(root * "y_fields.vtu")
    x_fields = pv.read(root * "x_fields.vtu")

    plotter.subplot(0, 0)
    plot_material(np, pv, plotter, design, W, L; colorbar=true, title="\$\\rho(x)\$\n(top)", font_size=font_size, title_font_size=title_font_size)
    plotter.camera.position = (-W / 2, -L / 2.75, des_high + 800 * L / (2 * 92.1437880268))
    plotter.camera.up = (1, 0, 0)
    plotter.camera.focal_point = (-W / 2, -L / 2.75, des_low - 100 * L / (2 * 92.1437880268))

    plotter.subplot(0, 1)
    if foundry
        plot_material(np, pv, plotter, design, W, L; colorbar=true, title="\$\\rho(x)\$\n", font_size=font_size, title_font_size=title_font_size, num_periods_x=2, num_periods_y=2, ontop=true)
    else
        plot_material(np, pv, plotter, design, W, L; colorbar=true, title="\$\\rho(x)\$\n(below)", font_size=font_size, title_font_size=title_font_size)
        plotter.camera.position = (-W / 2, -L / 1.45, des_low - 800 * L / (2 * 92.1437880268))
        plotter.camera.up = (1, 0, 0)
        plotter.camera.focal_point = (-W / 2, -L / 1.45, des_high + 100 * L / (2 * 92.1437880268))
    end

    plotter.subplot(0, 2)
    plot_material(np, pv, plotter, design, W, L; colorbar=false, title="blank1", font_size=font_size, title_font_size=title_font_size, num_periods_x=2, num_periods_y=2)
    up, position, focal_point = plotter.camera.up, plotter.camera.position, plotter.camera.focal_point
    plot_field(np, pv, plotter, y_fields, W, L, des_low; colorbar=true, title="\$\\mid E \\mid^2_{\\hat{y}}\$", font_size=font_size, title_font_size=title_font_size, num_periods_x=2, num_periods_y=2)
    plotter.camera.up, plotter.camera.position, plotter.camera.focal_point = up, position, focal_point

    plotter.subplot(0, 3)
    plot_material(np, pv, plotter, design, W, L; colorbar=false, title="blank2", font_size=font_size, title_font_size=title_font_size, num_periods_x=2, num_periods_y=2)
    up, position, focal_point = plotter.camera.up, plotter.camera.position, plotter.camera.focal_point
    plot_field(np, pv, plotter, x_fields, W, L, des_low; colorbar=true, title="\$\\mid E \\mid^2_{\\hat{x}}\$", font_size=font_size, title_font_size=title_font_size, num_periods_x=2, num_periods_y=2)
    plotter.camera.up, plotter.camera.position, plotter.camera.focal_point = up, position, focal_point

    plotter.show(screenshot=root * "full-visual.png")
end

function combine_figures(image_paths::Vector{Vector{String}})
    combined_rows = []
    for row in 1:size(image_paths, 1)
        push!(combined_rows, hcat(load.(image_paths[row])...))
    end
    vcat(combined_rows...)
end


function add_text!(image_path::String, labels, fontsize::Int)
    img = load(image_path)
    img = rotr90(img, 1)

    fig = CairoMakie.Figure(resolution=size(img), font="Dejavu Sans", frame=(linewidth=0,), backgroundcolor=(:white, 0))
    ax = CairoMakie.Axis(fig[1, 1], title="", backgroundcolor=:transparent)
    CairoMakie.image!(ax, img)

    for (label, position) in labels
        CairoMakie.text!(position[1], position[2], text=label, color=:black, textsize=fontsize)
    end

    CairoMakie.hidexdecorations!(ax)
    CairoMakie.hideydecorations!(ax)
    CairoMakie.save(image_path, fig)
end

function load_bid_parameters(pv, np, path_bid::String, path_mono::String)

    # Load bidirectional
    output_file = read(path_bid * "output.txt", String)
    geo_split = split(output_file, r"geometry = (?:DistributedEmitterOpt|Emitter3DTopOpt)\.SymmetricGeometry\(")
    if length(geo_split) < 2
        error("Could not find SymmetricGeometry in output file")
    end
    L = W = parse(Float64, split(geo_split[2], ",")[1])

    phys_split = split(output_file, r"phys_1 = (?:DistributedEmitterOpt|Emitter3DTopOpt)\.PhysicalParameters?\(")
    if length(phys_split) < 2
        error("Could not find PhysicalParameters in output file")
    end
    des_low, des_high = parse.(Float64, split(split(phys_split[2], ")")[1][1:end-1], ",")[7:8])
    design = pv.read(path_bid * "y_design.vtu")
    y_fields = pv.read(path_bid * "y_fields.vtu")
    x_fields = pv.read(path_bid * "x_fields.vtu")
    _, _, _, g_ar, _, _ = JLD2.load_object(path_bid * "results.jld2")

    if isfile(path_bid * "spectral.jld2")
        λ_s, gys, gxs, ggs = JLD2.load_object(path_bid * "spectral.jld2")
    else
        @warn "spectral.jld2 not found in $path_bid. Parsing output.txt for spectral data."
        λ_s = Float64[]
        gys = Float64[]
        gxs = Float64[]
        ggs = Float64[]
        for line in split(output_file, "\n")
            m = match(r"\(λ_, gy, gx, gg, norm_xy\) = \(([\d\.]+), ([\d\.]+), ([\d\.]+), ([\d\.]+),", line)
            if !isnothing(m)
                push!(λ_s, parse(Float64, m[1]))
                push!(gys, parse(Float64, m[2]))
                push!(gxs, parse(Float64, m[3]))
                push!(ggs, parse(Float64, m[4]))
            end
        end
        if isempty(λ_s)
            @warn "Could not parse spectral data from output.txt"
        end
    end
    bid_data = Dict(
        [:L, :W, :des_low, :des_high, :design, :y_fields, :x_fields, :g_ar, :λ_s, :gys, :gxs, :ggs] .=>
            [L, W, des_low, des_high, design, y_fields, x_fields, g_ar, λ_s, gys, gxs, ggs]
    )

    # Load monodirectional
    output_file = read(path_mono * "output.txt", String)
    geo_split = split(output_file, r"geometry = (?:DistributedEmitterOpt|Emitter3DTopOpt)\.SymmetricGeometry\(")
    if length(geo_split) < 2
        error("Could not find SymmetricGeometry in mono output file")
    end
    L = W = parse(Float64, split(geo_split[2], ",")[1])

    phys_split = split(output_file, r"phys_1 = (?:DistributedEmitterOpt|Emitter3DTopOpt)\.PhysicalParameters?\(")
    if length(phys_split) < 2
        error("Could not find PhysicalParameters in mono output file")
    end
    des_low, des_high = parse.(Float64, split(split(phys_split[2], ")")[1][1:end-1], ",")[7:8])

    design = pv.read(path_mono * "y_design.vtu")
    y_fields = pv.read(path_mono * "y_fields.vtu")
    _, _, _, g_ar, _, _ = JLD2.load_object(path_mono * "results.jld2")

    if isfile(path_mono * "spectral.jld2")
        λ_s, gys, gxs, ggs = JLD2.load_object(path_mono * "spectral.jld2")
    else
        @warn "spectral.jld2 not found in $path_mono. Parsing output.txt for spectral data."
        λ_s = Float64[]
        gys = Float64[]
        gxs = Float64[]
        ggs = Float64[]
        for line in split(output_file, "\n")
            m = match(r"\(λ_, gy, gx, gg, norm_xy\) = \(([\d\.]+), ([\d\.]+), ([\d\.]+), ([\d\.]+),", line)
            if !isnothing(m)
                push!(λ_s, parse(Float64, m[1]))
                push!(gys, parse(Float64, m[2]))
                push!(gxs, parse(Float64, m[3]))
                push!(ggs, parse(Float64, m[4]))
            end
        end
        if isempty(λ_s)
            @warn "Could not parse spectral data from mono output.txt"
        end
    end
    mono_data = Dict{Symbol,Any}(
        [:L, :W, :des_low, :des_high, :design, :y_fields, :x_fields, :g_ar, :λ_s, :gys, :gxs, :ggs] .=>
            [L, W, des_low, des_high, design, y_fields, x_fields, g_ar, λ_s, gys, gxs, ggs]
    )

    return bid_data, mono_data
end


function plot_directionals(pv, np, savepath::String, fig_width::Integer, text_size::Integer, title_size::Integer; mono_data::Dict{Symbol,Any}, bid_data::Dict{Symbol,Any}, lim_bid::Float64=1.0, lim_mon::Float64=1.0, diel::Bool=false, λ2::Float64=-1.0, flip1x::Bool=false, flip2x::Bool=false, flip1y::Bool=false, flip2y::Bool=false)

    eff_fig_height = 576
    eff_fig_width = round(Int, fig_width / 24)

    L_bid, W_bid, des_low_bid, des_high_bid, design_bid, y_fields_bid, x_fields_bid, g_ar_bid, λ_s_bid, gys_bid, gxs_bid, ggs_bid = (
        bid_data[:L], bid_data[:W], bid_data[:des_low], bid_data[:des_high], bid_data[:design], bid_data[:y_fields],
        bid_data[:x_fields], bid_data[:g_ar], bid_data[:λ_s], bid_data[:gys], bid_data[:gxs], bid_data[:ggs]
    )
    L_mon, W_mon, des_low_mon, des_high_mon, design_mon, y_fields_mon, g_ar_mon, λ_s_mon, gys_mon, gxs_mon, ggs_mon = (
        mono_data[:L], mono_data[:W], mono_data[:des_low], mono_data[:des_high], mono_data[:design], mono_data[:y_fields],
        mono_data[:g_ar], mono_data[:λ_s], mono_data[:gys], mono_data[:gxs], mono_data[:ggs]
    )

    sign1x = (flip1x) ? -1 : 1
    sign1y = (flip1y) ? -1 : 1
    plotter = pv.Plotter(shape=(1, 2), off_screen=true, window_size=(10 * eff_fig_width, Int(3 * eff_fig_height / 2)), border=false)
    plotter.subplot(0, 0)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_bid, W_bid, L_bid; colorbar=false, font_size=text_size, title_font_size=title_size, flipx=flip1x, flipy=flip1y)
    plotter.camera.position = (-W_bid / 2 * sign1x, -L_bid / 2.55 * sign1y, des_high_bid + 1200 * L_bid / (2 * 92.1437880268))
    plotter.camera.up = (1, 0, 0)
    plotter.camera.focal_point = (-W_bid / 2 * sign1x, -L_bid / 2.55 * sign1y, des_low_bid - 500 * L_bid / (2 * 92.1437880268))
    plotter.subplot(0, 1)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_bid, W_bid, L_bid; colorbar=true, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n", font_size=text_size, title_font_size=title_size, flipx=flip1x, flipy=flip1y)
    plot_substrate(np, pv, plotter, y_fields_bid, W_bid, L_bid, des_low_bid, flipx=flip1x, flipy=flip1y)
    plotter.camera.position = (plotter.camera.position .* 0.875 .+ (-W_bid / 1.0 * sign1x, -L_bid / 10.0 * sign1y, des_high_bid + 1000 * L_bid / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.focal_point = (plotter.camera.focal_point .* 0.875 .+ (-W_bid / 2 * sign1x, -L_bid / 2.55 * sign1y, des_low_bid - 300 * L_bid / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.zoom(0.825)
    plotter.camera.position = plotter.camera.position .+ [W_bid / 4.5, -L_bid / 4.5, 0]
    plotter.camera.focal_point = plotter.camera.focal_point .+ [W_bid / 4.5, -L_bid / 4.5, 0]
    plotter.show(screenshot=savepath * "-11-11.png")
    plotter.clear()

    sign2x = (flip2x) ? -1 : 1
    sign2y = (flip2y) ? -1 : 1
    plotter = pv.Plotter(shape=(1, 2), off_screen=true, window_size=(10 * eff_fig_width, Int(3 * eff_fig_height / 2)), border=false)
    plotter.subplot(0, 0)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_mon, W_mon, L_mon; colorbar=false, font_size=text_size, title_font_size=title_size, flipx=flip2x, flipy=flip2y)
    plotter.camera.position = (-W_mon / 2 * sign2x, -L_mon / 2.55 * sign2y, des_high_mon + 1200 * L_mon / (2 * 92.1437880268))
    plotter.camera.up = (1, 0, 0)
    plotter.camera.focal_point = (-W_mon / 2 * sign2x, -L_mon / 2.55 * sign2y, des_low_mon - 500 * L_mon / (2 * 92.1437880268))
    plotter.subplot(0, 1)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_mon, W_mon, L_mon; colorbar=true, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n", font_size=text_size, title_font_size=title_size, flipx=flip2x, flipy=flip2y)
    plot_substrate(np, pv, plotter, y_fields_mon, W_mon, L_mon, des_low_mon, flipx=flip2x, flipy=flip2y)
    plotter.camera.position = (plotter.camera.position .* 0.875 .+ (-W_mon / 1.0 * sign2x, -L_mon / 10.0 * sign2y, des_high_mon + 1000 * L_mon / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.focal_point = (plotter.camera.focal_point .* 0.875 .+ (-W_mon / 2 * sign2x, -L_mon / 2.55 * sign2y, des_low_mon - 300 * L_mon / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.zoom(0.825)
    plotter.camera.position = plotter.camera.position .+ [W_mon / 4.5, -L_mon / 4.5, 0]
    plotter.camera.focal_point = plotter.camera.focal_point .+ [W_mon / 4.5, -L_mon / 4.5, 0]
    plotter.show(screenshot=savepath * "-11-12.png")
    plotter.clear()

    figure_matrix = [[savepath * "-11-11.png"], [savepath * "-11-12.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * "-11.png", figure)

    plotter = pv.Plotter(shape=(1, 2), off_screen=true, window_size=(14 * eff_fig_width, eff_fig_height), border=false)
    plotter.subplot(0, 0)
    plot_material(np, pv, plotter, design_bid, W_bid, L_bid; colorbar=false, title="blank1", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2)
    up, position, focal_point = plotter.camera.up, plotter.camera.position, plotter.camera.focal_point
    plot_field(np, pv, plotter, y_fields_bid, W_bid, L_bid, des_low_bid; colorbar=true, title="\$\\mid \\boldsymbol{E} \\mid^2_{\\hat{y}}\$", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, lim=lim_bid)
    if diel
        plotter.camera.zoom(0.75)
        plotter.camera.position = plotter.camera.position .+ [-150, -150, 150]
        plotter.camera.focal_point = plotter.camera.focal_point .+ [-300, -300, 100]
    else
        plotter.camera.up, plotter.camera.position, plotter.camera.focal_point = up, position, focal_point
    end
    plotter.subplot(0, 1)
    plot_material(np, pv, plotter, design_bid, W_bid, L_bid; colorbar=false, title="blank2", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2)
    up, position, focal_point = plotter.camera.up, plotter.camera.position, plotter.camera.focal_point
    plot_field(np, pv, plotter, x_fields_bid, W_bid, L_bid, des_low_bid; colorbar=true, title="\$\\mid \\boldsymbol{E} \\mid^2_{\\hat{x}}\$", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, lim=lim_bid)
    if diel
        plotter.camera.zoom(0.75)
        plotter.camera.position = plotter.camera.position .+ [-150, -150, 150]
        plotter.camera.focal_point = plotter.camera.focal_point .+ [-300, -300, 100]
    else
        plotter.camera.up, plotter.camera.position, plotter.camera.focal_point = up, position, focal_point
    end
    plotter.show(screenshot=savepath * "-12-11.png")
    plotter.clear()

    fig = CairoMakie.Figure(resolution=(eff_fig_width * 14, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(fig[1, 1], yscale=log10, title="spectral response", xlabel="wavelength (nm)", ylabel=L"$g=\int |E_p^* E_e|^2$")
    CairoMakie.lines!(ax, λ_s_bid, gys_bid, label=L"opt$(g_\hat{y}+g_\hat{x}), g_\hat{y}$", color=:orange, linewidth=4)
    CairoMakie.lines!(ax, λ_s_bid, gxs_bid, label=L"opt$(g_\hat{y}+g_\hat{x}), g_\hat{x}$", color=:green, linewidth=4)
    CairoMakie.lines!(ax, λ_s_bid, ggs_bid, label=L"opt$(g_\hat{y}+g_\hat{x}), g_\hat{x}+g_\hat{y}$", color=:blue, linewidth=4)
    CairoMakie.lines!(ax, λ_s_mon, gys_mon, label=L"opt$(g_\hat{y}), g_\hat{y}$", color=:red, linewidth=4)
    if λ2 < 0
        CairoMakie.vlines!(ax, [532], color=:black, linestyle=:dash, linewidth=2, label=L"\lambda_0")
    else
        CairoMakie.vlines!(ax, [532], color=:black, linestyle=:dash, linewidth=2, label=L"\lambda_p")
        CairoMakie.vlines!(ax, [λ2], color=:black, linestyle=:dash, linewidth=2, label=L"\lambda_e")
    end
    CairoMakie.axislegend(ax, position=:rt)
    save(savepath * "-12-21.png", fig)

    plotter = pv.Plotter(shape=(1, 1), off_screen=true, window_size=(7 * eff_fig_width, eff_fig_height), border=false)
    plot_material(np, pv, plotter, design_mon, W_mon, L_mon; colorbar=false, title="blank1", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2)
    up, position, focal_point = plotter.camera.up, plotter.camera.position, plotter.camera.focal_point
    plot_field(np, pv, plotter, y_fields_mon, W_mon, L_mon, des_low_mon; colorbar=true, title="\$\\mid \\boldsymbol{E} \\mid^2_{\\hat{y}}\$", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, lim=lim_mon)
    if diel
        plotter.camera.zoom(0.75)
        plotter.camera.position = plotter.camera.position .+ [-150, -150, 150]
        plotter.camera.focal_point = plotter.camera.focal_point .+ [-300, -300, 100]
    else
        plotter.camera.up, plotter.camera.position, plotter.camera.focal_point = up, position, focal_point
    end
    plotter.show(screenshot=savepath * "-12-31.png")
    plotter.clear()

    figure = CairoMakie.Figure(resolution=(7 * eff_fig_width, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(figure[1, 1], yscale=log10, title="objective history", xlabel="Iteration", ylabel=L"g = \int |E_p^* E_e|^2")
    CairoMakie.lines!(ax, [i for i in 1:length(g_ar_bid)], [i for i in g_ar_bid], label="both directions", color=:blue)
    CairoMakie.lines!(ax, [i for i in 1:length(g_ar_mon)], [i for i in g_ar_mon], label="only y-direction", color=:red)
    CairoMakie.axislegend(ax, position=:lt)
    save(savepath * "-12-32.png", figure)

    figure_matrix = [[savepath * "-12-11.png"], [savepath * "-12-21.png"], [savepath * "-12-31.png", savepath * "-12-32.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * "-12.png", figure)

    figure_matrix = [[savepath * "-11.png", savepath * "-12.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * ".png", figure)

    add_text!(savepath * ".png", [
            ("(a)", (50.0, eff_fig_height * 3.0 - 50)),
            ("(b)", (10 * eff_fig_width + 50.0, eff_fig_height * 3.0 - 50)),
            ("(c)", (10 * eff_fig_width + 50.0, eff_fig_height * 2.0 - 50)),
            ("(d)", (50.0, eff_fig_height * 1.5 - 50)),
            ("(e)", (10 * eff_fig_width + 50.0, eff_fig_height * 1.0 - 50)),
            ("(f)", (17 * eff_fig_width + 50.0, eff_fig_height * 1.0 - 50)),
        ], title_size)

    return savepath * ".png"
end

function plot_geometrics(pv, np, savepath::String, fig_width::Integer, text_size::Integer, title_size::Integer; con_data::Dict{Symbol,Any}, uns_data::Dict{Symbol,Any}, lim_uns::Float64=1.0, lim_con::Float64=1.0, diel::Bool=false, full::Bool=false, flip1x::Bool=false, flip2x::Bool=false, flip1y::Bool=false, flip2y::Bool=false)

    eff_fig_height = 700
    eff_fig_width = round(Int, fig_width / 6)

    L_uns, W_uns, des_low_uns, des_high_uns, design_uns, y_fields_uns, x_fields_uns, g_ar_uns, λ_s_uns, gys_uns = (
        uns_data[:L], uns_data[:W], uns_data[:des_low], uns_data[:des_high], uns_data[:design], uns_data[:y_fields],
        uns_data[:x_fields], uns_data[:g_ar], uns_data[:λ_s], uns_data[:gys]
    )
    L_con, W_con, des_low_con, des_high_con, design_con, y_fields_con, g_ar_con, λ_s_con, gys_con = (
        con_data[:L], con_data[:W], con_data[:des_low], con_data[:des_high], con_data[:design], con_data[:y_fields],
        con_data[:g_ar], con_data[:λ_s], con_data[:gys]
    )

    δxy = ifelse(full, 100, 0)

    sign1x = (flip1x) ? -1 : 1
    sign1y = (flip1y) ? -1 : 1
    plotter = pv.Plotter(shape=(1, 2), off_screen=true, window_size=(4 * eff_fig_width, eff_fig_height), border=false)
    plotter.subplot(0, 0)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_uns, W_uns, L_uns; colorbar=false, font_size=text_size, title_font_size=title_size, full, flipx=flip1x, flipy=flip1y)
    plotter.camera.position = (-W_uns / 2 * sign1x + δxy, -L_uns / 2.55 * sign1y + δxy, des_high_uns + 1000 * L_uns / (2 * 92.1437880268))
    plotter.camera.up = (1, 0, 0)
    plotter.camera.focal_point = (-W_uns / 2 * sign1x + δxy, -L_uns / 2.55 * sign1y + δxy, des_low_uns - 300 * L_uns / (2 * 92.1437880268))
    plotter.camera.zoom(1.25)
    plotter.subplot(0, 1)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_uns, W_uns, L_uns; colorbar=true, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n", font_size=text_size, title_font_size=title_size, full, flipx=flip1x, flipy=flip1y)
    plot_substrate(np, pv, plotter, y_fields_uns, W_uns, L_uns, des_low_uns; full, flipx=flip1x, flipy=flip1y)
    plotter.camera.position = (plotter.camera.position .* 0.875 .+ (-W_uns / 1.0 * sign1x, -L_uns / 10.0 * sign1y, des_high_uns + 1000 * L_uns / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.focal_point = (plotter.camera.focal_point .* 0.875 .+ (-W_uns / 2 * sign1x, -L_uns / 2.55 * sign1y, des_low_uns - 300 * L_uns / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.zoom(0.825)
    plotter.camera.position = plotter.camera.position .+ [W_uns / 4.5, -L_uns / 4.5, 0]
    plotter.camera.focal_point = plotter.camera.focal_point .+ [W_uns / 4.5, -L_uns / 4.5, 0]
    plotter.camera.zoom(1.25)
    plotter.show(screenshot=savepath * "-1-11.png")
    plotter.clear()

    plotter = pv.Plotter(shape=(1, 1), off_screen=true, window_size=(2 * eff_fig_width, eff_fig_height), border=false)
    plotter.subplot(0, 0)
    plot_material(np, pv, plotter, design_uns, W_uns, L_uns; colorbar=false, title="blank1", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, full)
    up, position, focal_point = plotter.camera.up, plotter.camera.position, plotter.camera.focal_point
    plot_field(np, pv, plotter, y_fields_uns, W_uns, L_uns, des_low_uns; colorbar=true, title="\$\\mid \\boldsymbol{E} \\mid^2_{\\hat{y}}\$", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, lim=lim_uns, vertical=true, full)
    if diel
        plotter.camera.zoom(0.95)
        plotter.camera.position = plotter.camera.position .+ [175, -250, 125]
        plotter.camera.focal_point = plotter.camera.focal_point .+ [-150, -175, 100]
    else
        plotter.camera.up, plotter.camera.position, plotter.camera.focal_point = up, position, focal_point
    end
    plotter.show(screenshot=savepath * "-1-12.png")
    plotter.clear()

    figure_matrix = [[savepath * "-1-11.png", savepath * "-1-12.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * "-1.png", figure)

    figure = CairoMakie.Figure(resolution=(eff_fig_width * 3, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(figure[1, 1], yscale=log10, title="objective history", xlabel="Iteration", ylabel=L"g = \int |E_p^* E_e|^2")
    CairoMakie.lines!(ax, [i for i in 1:length(g_ar_uns)], [i for i in g_ar_uns], label="freeform", color=:blue)
    CairoMakie.lines!(ax, [i for i in 1:length(g_ar_con)], [i for i in g_ar_con], label="constrained", color=:red)
    CairoMakie.axislegend(ax, position=:lt)
    save(savepath * "-2-11.png", figure)

    fig = CairoMakie.Figure(resolution=(eff_fig_width * 3, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(fig[1, 1], yscale=log10, title="spectral response", xlabel="wavelength (nm)", ylabel=L"$g=\int |E_p^* E_e|^2$")
    CairoMakie.lines!(ax, λ_s_uns, gys_uns, label=L"opt(freeform)$, g_\hat{y}$", color=:blue, linewidth=4)
    CairoMakie.lines!(ax, λ_s_con, gys_con, label=L"opt(constrained)$, g_\hat{y}$", color=:red, linewidth=4)
    CairoMakie.vlines!(ax, [532], color=:black, linestyle=:dash, linewidth=2, label=L"\lambda_0")
    CairoMakie.axislegend(ax, position=:rt)
    save(savepath * "-2-12.png", fig)

    figure_matrix = [[savepath * "-2-11.png", savepath * "-2-12.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * "-2.png", figure)

    sign2x = (flip2x) ? -1 : 1
    sign2y = (flip2y) ? -1 : 1
    plotter = pv.Plotter(shape=(1, 2), off_screen=true, window_size=(4 * eff_fig_width, eff_fig_height), border=false)
    plotter.subplot(0, 0)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_con, W_con, L_con; colorbar=false, font_size=text_size, title_font_size=title_size, full, flipx=flip2x, flipy=flip2y)
    plotter.camera.position = (-W_con / 2 * sign2x + δxy, -L_con / 2.55 * sign2y + δxy, des_high_con + 1000 * L_con / (2 * 92.1437880268))
    plotter.camera.up = (1, 0, 0)
    plotter.camera.focal_point = (-W_con / 2 * sign2x + δxy, -L_con / 2.55 * sign2y + δxy, des_low_con - 300 * L_con / (2 * 92.1437880268))
    plotter.camera.zoom(1.25)
    plotter.subplot(0, 1)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_con, W_con, L_con; colorbar=true, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n", font_size=text_size, title_font_size=title_size, full, flipx=flip2x, flipy=flip2y)
    plot_substrate(np, pv, plotter, y_fields_con, W_con, L_con, des_low_con; full, flipx=flip2x, flipy=flip2y)
    plotter.camera.position = (plotter.camera.position .* 0.875 .+ (-W_con / 1.0 * sign2x, -L_con / 10.0 * sign2y, des_high_con + 1000 * L_con / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.focal_point = (plotter.camera.focal_point .* 0.875 .+ (-W_con / 2 * sign2x, -L_con / 2.55 * sign2y, des_low_con - 300 * L_con / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.zoom(0.825)
    plotter.camera.position = plotter.camera.position .+ [W_con / 4.5, -L_con / 4.5, 0]
    plotter.camera.focal_point = plotter.camera.focal_point .+ [W_con / 4.5, -L_con / 4.5, 0]
    plotter.camera.zoom(1.25)
    plotter.show(screenshot=savepath * "-3-11.png")
    plotter.clear()

    plotter = pv.Plotter(shape=(1, 1), off_screen=true, window_size=(2 * eff_fig_width, eff_fig_height), border=false)
    plotter.subplot(0, 0)
    plot_material(np, pv, plotter, design_con, W_con, L_con; colorbar=false, title="blank1", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, full)
    up, position, focal_point = plotter.camera.up, plotter.camera.position, plotter.camera.focal_point
    plot_field(np, pv, plotter, y_fields_con, W_con, L_con, des_low_con; colorbar=true, title="\$\\mid \\boldsymbol{E} \\mid^2_{\\hat{y}}\$", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, lim=lim_con, vertical=true, full)
    if diel
        plotter.camera.zoom(0.95)
        plotter.camera.position = plotter.camera.position .+ [175, -250, 125]
        plotter.camera.focal_point = plotter.camera.focal_point .+ [-150, -175, 100]
    else
        plotter.camera.up, plotter.camera.position, plotter.camera.focal_point = up, position, focal_point
    end
    plotter.show(screenshot=savepath * "-3-12.png")

    figure_matrix = [[savepath * "-3-11.png", savepath * "-3-12.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * "-3.png", figure)

    figure_matrix = [[savepath * "-1.png"], [savepath * "-2.png"], [savepath * "-3.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * ".png", figure)

    add_text!(savepath * ".png", [
            ("(a)", (50.0, eff_fig_height * 3.0 - 50)),
            ("(b)", (4 * eff_fig_width + 50.0, eff_fig_height * 3.0 - 50)),
            ("(c)", (50.0, eff_fig_height * 2.0 - 50)),
            ("(d)", (3 * eff_fig_width + 50.0, eff_fig_height * 2.0 - 50)),
            ("(e)", (50.0, eff_fig_height * 1.0 - 50)),
            ("(f)", (4 * eff_fig_width + 50.0, eff_fig_height * 1.0 - 50)),
        ], title_size)

    return savepath * ".png"
end


function plot_geometrics_one_only(pv, np, savepath::String, fig_width::Integer, text_size::Integer, title_size::Integer; uns_data::Dict{Symbol,Any}, lim_uns::Float64=1.0, diel::Bool=false, full::Bool=false)

    eff_fig_height = 700
    eff_fig_width = round(Int, fig_width / 6)

    L_uns, W_uns, des_low_uns, des_high_uns, design_uns, y_fields_uns, x_fields_uns, g_ar_uns, λ_s_uns, gys_uns = (
        uns_data[:L], uns_data[:W], uns_data[:des_low], uns_data[:des_high], uns_data[:design], uns_data[:y_fields],
        uns_data[:x_fields], uns_data[:g_ar], uns_data[:λ_s], uns_data[:gys]
    )

    δxy = ifelse(full, 100, 0)

    plotter = pv.Plotter(shape=(1, 2), off_screen=true, window_size=(4 * eff_fig_width, eff_fig_height), border=false)
    plotter.subplot(0, 0)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_uns, W_uns, L_uns; colorbar=false, font_size=text_size, title_font_size=title_size, full)
    plotter.camera.position = (-W_uns / 2 + δxy, -L_uns / 2.55 + δxy, des_high_uns + 1000 * L_uns / (2 * 92.1437880268))
    plotter.camera.up = (1, 0, 0)
    plotter.camera.focal_point = (-W_uns / 2 + δxy, -L_uns / 2.55 + δxy, des_low_uns - 300 * L_uns / (2 * 92.1437880268))
    plotter.camera.zoom(1.25)
    plotter.subplot(0, 1)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_uns, W_uns, L_uns; colorbar=true, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n", font_size=text_size, title_font_size=title_size, full)
    plot_substrate(np, pv, plotter, y_fields_uns, W_uns, L_uns, des_low_uns; full)
    plotter.camera.position = (plotter.camera.position .* 0.875 .+ (-W_uns / 1.0, -L_uns / 10.0, des_high_uns + 1000 * L_uns / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.focal_point = (plotter.camera.focal_point .* 0.875 .+ (-W_uns / 2, -L_uns / 2.55, des_low_uns - 300 * L_uns / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.zoom(0.825)
    plotter.camera.position = plotter.camera.position .+ [W_uns / 4.5, -L_uns / 4.5, 0]
    plotter.camera.focal_point = plotter.camera.focal_point .+ [W_uns / 4.5, -L_uns / 4.5, 0]
    plotter.camera.zoom(1.25)
    plotter.show(screenshot=savepath * "-1-11.png")
    plotter.clear()

    plotter = pv.Plotter(shape=(1, 1), off_screen=true, window_size=(2 * eff_fig_width, eff_fig_height), border=false)
    plotter.subplot(0, 0)
    plot_material(np, pv, plotter, design_uns, W_uns, L_uns; colorbar=false, title="blank1", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, full)
    up, position, focal_point = plotter.camera.up, plotter.camera.position, plotter.camera.focal_point
    plot_field(np, pv, plotter, y_fields_uns, W_uns, L_uns, des_low_uns; colorbar=true, title="\$\\mid \\boldsymbol{E} \\mid^2_{\\hat{y}}\$", font_size=text_size, title_font_size=title_size, num_periods_x=2, num_periods_y=2, lim=lim_uns, vertical=true, full)
    if diel
        plotter.camera.zoom(0.95)
        plotter.camera.position = plotter.camera.position .+ [175, -250, 125]
        plotter.camera.focal_point = plotter.camera.focal_point .+ [-150, -175, 100]
    else
        plotter.camera.up, plotter.camera.position, plotter.camera.focal_point = up, position, focal_point
    end
    plotter.show(screenshot=savepath * "-1-12.png")
    plotter.clear()

    figure_matrix = [[savepath * "-1-11.png", savepath * "-1-12.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * "-1.png", figure)

    figure = CairoMakie.Figure(resolution=(eff_fig_width * 3, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(figure[1, 1], yscale=log10, title="objective history", xlabel="Iteration", ylabel=L"g = \int |E_p^* E_e|^2")
    CairoMakie.lines!(ax, [i for i in 1:length(g_ar_uns)], [i for i in g_ar_uns], label="freeform", color=:blue)
    CairoMakie.axislegend(ax, position=:lt)
    save(savepath * "-2-11.png", figure)

    fig = CairoMakie.Figure(resolution=(eff_fig_width * 3, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(fig[1, 1], yscale=log10, title="spectral response", xlabel="wavelength (nm)", ylabel=L"$g=\int |E_p^* E_e|^2$")
    CairoMakie.lines!(ax, λ_s_uns, gys_uns, label=L"opt(freeform)$, g_\hat{y}$", color=:blue, linewidth=4)
    CairoMakie.vlines!(ax, [532], color=:black, linestyle=:dash, linewidth=2, label=L"\lambda_0")
    CairoMakie.axislegend(ax, position=:rt)
    save(savepath * "-2-12.png", fig)

    figure_matrix = [[savepath * "-2-11.png", savepath * "-2-12.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * "-2.png", figure)

    figure_matrix = [[savepath * "-1.png"], [savepath * "-2.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * ".png", figure)

    add_text!(savepath * ".png", [
            ("(a)", (50.0, eff_fig_height * 2.0 - 50)),
            ("(b)", (4 * eff_fig_width + 50.0, eff_fig_height * 2.0 - 50)),
            ("(c)", (50.0, eff_fig_height * 1.0 - 50)),
            ("(d)", (3 * eff_fig_width + 50.0, eff_fig_height * 1.0 - 50)),
        ], title_size)

    return savepath * ".png"
end



function plot_tolerance(pv, np, savepath::String, fig_width::Integer, text_size::Integer, title_size::Integer; path_sil::String, path_nit::String)

    eff_fig_height = 700
    eff_fig_width = round(Int, fig_width / 3)

    # Load silver
    output_file = read(path_sil * "output.txt", String)
    L_sil = W_sil = parse(Float64, split(split(output_file, "geometry = DistributedEmitterOpt.SymmetricGeometry(")[2], ",")[1])
    des_low_sil, des_high_sil = parse.(Float64, split(split(split(output_file, "phys_1 = DistributedEmitterOpt.PhysicalParams(")[2], ")")[1][1:end-1], ",")[7:8])
    design_sil = pv.read(path_sil * "y_design.vtu")
    δs_p_sil, gys_p_sil, _, _ = JLD2.load_object(path_sil * "perturbed.jld2")
    δs_n_sil, gys_n_sil, _, _ = JLD2.load_object(path_sil * "noise.jld2")

    # Load nitride
    output_file = read(path_nit * "output.txt", String)
    L_nit = W_nit = parse(Float64, split(split(output_file, "geometry = DistributedEmitterOpt.SymmetricGeometry(")[2], ",")[1])
    des_low_nit, des_high_nit = parse.(Float64, split(split(split(output_file, "phys_1 = DistributedEmitterOpt.PhysicalParams(")[2], ")")[1][1:end-1], ",")[7:8])
    design_nit = pv.read(path_nit * "y_design.vtu")
    δs_p_nit, gys_p_nit, _, _ = JLD2.load_object(path_nit * "perturbed.jld2")
    δs_n_nit, gys_n_nit, _, _ = JLD2.load_object(path_nit * "noise.jld2")

    # Load field data for substrate plots
    y_fields_sil = pv.read(path_sil * "y_fields.vtu")
    y_fields_nit = pv.read(path_nit * "y_fields.vtu")

    # Plot silver design
    plotter = pv.Plotter(shape=(1, 1), off_screen=true, window_size=(eff_fig_width, eff_fig_height), border=false)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_sil, W_sil, L_sil; colorbar=true, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n", font_size=text_size, title_font_size=title_size, full=true)
    plot_substrate(np, pv, plotter, y_fields_sil, W_sil, L_sil, des_low_sil; full=true)

    plotter.camera.position = (plotter.camera.position .* 0.875 .+ (-W_sil / 1.0, -L_sil / 10.0, des_high_sil + 1000 * L_sil / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.focal_point = (plotter.camera.focal_point .* 0.875 .+ (-W_sil / 2, -L_sil / 2.55, des_low_sil - 300 * L_sil / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.zoom(0.825)
    plotter.camera.position = plotter.camera.position .+ [W_sil / 4.5, -L_sil / 4.5, 0]
    plotter.camera.focal_point = plotter.camera.focal_point .+ [W_sil / 4.5, -L_sil / 4.5, 0]
    plotter.camera.zoom(1.25)
    plotter.show(screenshot=savepath * "-11.png")
    plotter.clear()

    fig = CairoMakie.Figure(resolution=(eff_fig_width, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(fig[1, 1], yscale=log10, title="filter radius response", xlabel=L"$\Delta R_f$ (nm)", ylabel=L"$g=\int |E_p^* E_e|^2$")
    CairoMakie.lines!(ax, δs_p_sil, gys_p_sil, label="silver", color=:blue, linewidth=4)
    CairoMakie.axislegend(ax, position=:rt)
    save(savepath * "-12.png", fig)

    fig = CairoMakie.Figure(resolution=(eff_fig_width, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(fig[1, 1], yscale=log10, title="perturbed design", xlabel=L"\Delta \rho", ylabel=L"$g=\int |E_p^* E_e|^2$")
    CairoMakie.lines!(ax, δs_n_sil, gys_n_sil, label="silver", color=:blue, linewidth=4)
    CairoMakie.axislegend(ax, position=:rt)
    save(savepath * "-13.png", fig)

    plotter = pv.Plotter(shape=(1, 1), off_screen=true, window_size=(eff_fig_width, eff_fig_height), border=false)
    plotter.show_axes()
    plot_material(np, pv, plotter, design_nit, W_nit, L_nit; colorbar=true, title=" \$\\bar{\\rho}(\\boldsymbol{r})\$\n", font_size=text_size, title_font_size=title_size, full=true)
    plot_substrate(np, pv, plotter, y_fields_nit, W_nit, L_nit, des_low_nit; full=true)
    plotter.camera.position = (plotter.camera.position .* 0.875 .+ (-W_nit / 1.0, -L_nit / 10.0, des_high_nit + 1000 * L_nit / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.focal_point = (plotter.camera.focal_point .* 0.875 .+ (-W_nit / 2, -L_nit / 2.55, des_low_nit - 300 * L_nit / (2 * 92.1437880268)) .* 0.125)
    plotter.camera.zoom(0.825)
    plotter.camera.position = plotter.camera.position .+ [W_nit / 4.5, -L_nit / 4.5, 0]
    plotter.camera.focal_point = plotter.camera.focal_point .+ [W_nit / 4.5, -L_nit / 4.5, 0]
    plotter.camera.zoom(1.25)
    plotter.show(screenshot=savepath * "-21.png")
    plotter.clear()

    fig = CairoMakie.Figure(resolution=(eff_fig_width, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(fig[1, 1], yscale=log10, title="filter radius response", xlabel=L"$\Delta R_f$ (nm)", ylabel=L"$g=\int |E_p^* E_e|^2$")
    CairoMakie.lines!(ax, δs_p_nit, gys_p_nit, label="nitride", color=:blue, linewidth=4)
    CairoMakie.axislegend(ax, position=:rt)
    save(savepath * "-22.png", fig)

    fig = CairoMakie.Figure(resolution=(eff_fig_width, eff_fig_height), fontsize=text_size, titlefontsize=title_size)
    ax = CairoMakie.Axis(fig[1, 1], yscale=log10, title="perturbed design", xlabel=L"\Delta \rho", ylabel=L"$g=\int |E_p^* E_e|^2$")
    CairoMakie.lines!(ax, δs_n_nit, gys_n_nit, label="nitride", color=:blue, linewidth=4)
    CairoMakie.axislegend(ax, position=:rt)
    save(savepath * "-23.png", fig)

    figure_matrix = [[savepath * "-11.png", savepath * "-12.png", savepath * "-13.png"], [savepath * "-21.png", savepath * "-22.png", savepath * "-23.png"]]
    figure = combine_figures(figure_matrix)
    save(savepath * ".png", figure)

    add_text!(savepath * ".png", [
            ("(a)", (50.0, eff_fig_height * 2.0 - 50)),
            ("(b)", (eff_fig_width + 50.0, eff_fig_height * 2.0 - 50)),
            ("(c)", (2 * eff_fig_width + 50.0, eff_fig_height * 2.0 - 50)),
            ("(d)", (50.0, eff_fig_height * 1.0 - 50)),
            ("(e)", (eff_fig_width + 50.0, eff_fig_height * 1.0 - 50)),
            ("(f)", (2 * eff_fig_width + 50.0, eff_fig_height * 1.0 - 50)),
        ], title_size)

    return savepath * ".png"
end


# --- Shared Helper Functions ---

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

function crop_white_margins(input_path::AbstractString, output_path::AbstractString; threshold::Real=0.98, more=0)
    if !isfile(input_path)
        @error "Input file not found for cropping: $input_path"
        return nothing
    end
    try
        img = load(input_path)
        cropped = crop_white_margins(img; threshold=threshold)

        if more == 1
            h, w = size(cropped)
            if h < 800 || w < 1500
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
            save(output_path, cropped)
        elseif more == 2
            save(output_path, cropped)
        else
            save(output_path, cropped)
        end
        return output_path
    catch e
        @error "Error cropping image $input_path:" e
        return nothing
    end
end

_clamp01(x) = max(0.0, min(1.0, x))

function brighten_image!(path::AbstractString; factor::Real=1.18)
    if !isfile(path)
        @warn "Cannot brighten missing image $path"
        return nothing
    end
    try
        img = load(path)
        boosted = map(img) do c
            rgba = RGBA(c)
            RGBA(_clamp01(rgba.r * factor), _clamp01(rgba.g * factor), _clamp01(rgba.b * factor), alpha(rgba))
        end
        save(path, boosted)
        return path
    catch e
        @warn "Failed to brighten image $path" e
        return nothing
    end
end

function save_geometry_snapshot(np, pv, mesh, filename, W, L;
    color=nothing,
    window_size=(400, 400),
    zoom=1.0,
    view_angle=nothing,
    num_periods_x=2,
    num_periods_y=2,
    contour=true,
    full=false,
    design_field="p",
    flipy=true,
    opacity=1.0,
    axes_viewport=(0.0, 0.0, 0.35, 0.35),
    kwargs...)

    if isnothing(mesh)
        @warn "Mesh is nothing, skipping snapshot: $filename"
        return
    end

    pv.global_theme.transparent_background = true
    plotter = pv.Plotter(off_screen=true, window_size=window_size, border=false)

    contour_color = isnothing(color) ? "#b097d1" : color

    plot_material(np, pv, plotter, mesh, W, L;
        colorbar=false,
        num_periods_x=num_periods_x,
        num_periods_y=num_periods_y,
        contour=contour,
        design_field=design_field,
        contour_color=contour_color,
        full=full,
        flipy=flipy,
        opacity=opacity,
        kwargs...
    )

    plotter.camera.zoom(zoom)

    plotter.add_axes(
        line_width=5,
        cone_radius=0.4,
        shaft_length=0.8,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.6, 0.2),
        viewport=axes_viewport,
    )

    try
        mkpath(dirname(filename))
        plotter.screenshot(filename)
        println("Saved geometry snapshot to $filename")
    catch e
        @error "Failed to save screenshot $filename: $e"
    finally
        plotter.clear()
        plotter.close()
    end
end

function plot_field_slices(np, pv, field_mesh, design_mesh, W, L, hd;
    slice_y_coord=nothing, slice_x_coord=nothing, slice_z_coord=nothing,
    color="#d62728", main_plot_kwargs=Dict(), plotter_kwargs=Dict())

    pv.global_theme.transparent_background = true
    plotter = pv.Plotter(off_screen=true; plotter_kwargs...)

    plot_field(np, pv, plotter, field_mesh, W, L, 0.0; main_plot_kwargs...)

    slice_plane_opacity = 0.5

    if !isnothing(slice_y_coord)
        num_periods_x = get(main_plot_kwargs, "num_periods_x", 1)
        full = get(main_plot_kwargs, "full", false)
        x_multiplier = full ? 1 : 2
        total_width = num_periods_x * W * x_multiplier
        mesh_bounds = field_mesh.bounds
        z_depth = mesh_bounds[6] - mesh_bounds[5]

        plane_y = pv.Plane(center=(W / 2, slice_y_coord + W * 0.75, -hd / 2), direction=(0, 1, 0),
            i_size=total_width * 2, j_size=z_depth * 1.25,
            i_resolution=max(1, num_periods_x * 5), j_resolution=10)
        plotter.add_mesh(plane_y, color=color, opacity=slice_plane_opacity, pickable=false)
    end

    plotter.background_color = "white"
    return plotter
end


function get_spectral_data(path::String)
    if isfile(joinpath(path, "spectral.jld2"))
        return JLD2.load_object(joinpath(path, "spectral.jld2"))
    end

    if isfile(joinpath(path, "ani-spectral.jld2"))
        return JLD2.load_object(joinpath(path, "ani-spectral.jld2"))
    end

    output_path = joinpath(path, "output.txt")
    if isfile(output_path)
        @warn "spectral.jld2 not found in $path. Parsing output.txt for spectral data."
        output_file = read(output_path, String)
        λ_s = Float64[]
        gys = Float64[]
        gxs = Float64[]
        ggs = Float64[]
        for line in split(output_file, "\n")
            m = match(r"\(λ_, gy, gx, gg, norm_xy\) = \(([\d\.]+), ([\d\.]+), ([\d\.]+), ([\d\.]+),", line)
            if !isnothing(m)
                push!(λ_s, parse(Float64, m[1]))
                push!(gys, parse(Float64, m[2]))
                push!(gxs, parse(Float64, m[3]))
                push!(ggs, parse(Float64, m[4]))
            end
        end
        if !isempty(λ_s)
            return λ_s, gys, gxs, ggs
        end
    end

    error("Could not find or parse spectral data in $path")
end

end # module VisualizationExt

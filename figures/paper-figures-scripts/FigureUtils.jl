
module FigureUtils

using PyCall
using Images
using FileIO
using Colors
using ColorSchemes

# Export common functions and constants
export plot_material, crop_white_margins, brighten_image!, hex_to_rgba, plot_geometry_for_inset
export COLOR_METAL_ISOTROPIC, COLOR_METAL_ANISOTROPIC, COLOR_METAL_ISO2ANI, COLOR_METAL_INELASTIC
export COLOR_DIEL_ISOTROPIC, COLOR_DIEL_ANISOTROPIC, COLOR_DIEL_ISO2ANI, COLOR_DIEL_INELASTIC
export COLOR_PUMP, COLOR_EMISSION

# --- Global Color Definitions ---
# Metal colors
const COLOR_METAL_ISOTROPIC = "#8c78a7"  # Base purple
const COLOR_METAL_ANISOTROPIC = "#B18B7D"  # More blue-purple
const COLOR_METAL_ISO2ANI = "#231e29"  # Dark purple
const COLOR_METAL_INELASTIC = "#6d7b75"  # Green-purple

# Dielectric colors
const COLOR_DIEL_ISOTROPIC = "#D7263D"  # Base red
const COLOR_DIEL_ANISOTROPIC = "#FF6B35"  # Orange-red
const COLOR_DIEL_ISO2ANI = "#400b12"  # Dark purple-red
const COLOR_DIEL_INELASTIC = "#B5179E"  # Purple-red

# Other colors
const COLOR_PUMP = "#1B3B6F"  # Blue for pump wavelength
const COLOR_EMISSION = :green  # Green for emission wavelength

# --- Helper Functions ---

function plot_geometry_for_inset(np, pv, mesh_pyobject, output_filename;
    scalar_field="p", window_size=(400, 400), color="#b097d1", flipy=false)

    pv.global_theme.transparent_background = true
    plotter = pv.Plotter(off_screen=true, window_size=window_size)

    if !hasproperty(mesh_pyobject, :bounds)
        error("Mesh object missing 'bounds'.")
    end
    xl, xr, yl, yr, _, _ = mesh_pyobject.bounds
    W = xr - xl
    L = yr - yl

    # Use existing plot_material
    plot_material(np, pv, plotter, mesh_pyobject, W, L;
        colorbar=false, num_periods_x=2, num_periods_y=2,
        contour=true, design_field=scalar_field, color=color, flipy=flipy)

    # Add small axes
    plotter.add_axes(
        line_width=5,
        cone_radius=0.4,
        shaft_length=0.8,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.6, 0.2),
        viewport=(0.66, 0.0, 1.0, 0.34),
    )

    plotter.screenshot(output_filename)
    plotter.clear()
    plotter.close()
    println("Saved geometry inset to $output_filename")
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
        return path
    catch e
        @warn "Failed to brighten image $path" e
        return nothing
    end
end

# --- Plotting Functions ---

function plot_material(np, pv, plotter, design, W, L;
    colorbar=true, title="\$\\rho(x)\$", font_size=20, title_font_size=24,
    num_periods_x=1, num_periods_y=1, ontop=false, design_field="p",
    contour=true, full=false, clim=(0.0, 1.0), opacity=nothing,
    reflectybool=true, flipx=false, flipy=true,
    material_type="metal", case="isotropic", color=nothing)

    pv.global_theme.allow_empty_mesh = true

    contours = design.contour(np.linspace(0, 1, 2), scalars=design_field)
    signx = (flipx) ? -1 : 1
    signy = (flipy) ? -1 : 1
    reflectx = pv.DataSetFilters.reflect(design, (1, 0, 0), point=(-W / 2 * signx, 0, 0))
    reflecty = pv.DataSetFilters.reflect(design, (0, 1, 0), point=(0, -L / 2 * signy, 0))
    reflectxy = pv.DataSetFilters.reflect(reflecty, (1, 0, 0), point=(-W / 2 * signx, 0, 0))

    # Determine color
    contour_color = if !isnothing(color)
        pv.Color(color)
    else
        if material_type == "metal"
            if case == "isotropic"
                pv.Color(COLOR_METAL_ISOTROPIC)
            elseif case == "anisotropic"
                pv.Color(COLOR_METAL_ANISOTROPIC)
            elseif case == "iso2ani"
                pv.Color(COLOR_METAL_ISO2ANI)
            elseif case == "inelastic"
                pv.Color(COLOR_METAL_INELASTIC)
            else
                pv.Color(COLOR_METAL_ISOTROPIC)
            end
        else  # dielectric
            if case == "isotropic"
                pv.Color(COLOR_DIEL_ISOTROPIC)
            elseif case == "anisotropic"
                pv.Color(COLOR_DIEL_ANISOTROPIC)
            elseif case == "iso2ani"
                pv.Color(COLOR_DIEL_ISO2ANI)
            elseif case == "inelastic"
                pv.Color(COLOR_DIEL_INELASTIC)
            else
                pv.Color(COLOR_DIEL_ISOTROPIC)
            end
        end
    end

    scalargs = if ontop
        ["title" => title, "vertical" => false, "label_font_size" => font_size, "title_font_size" => title_font_size, "position_x" => 0.25, "position_y" => 0.8, "color" => "Black", "use_opacity" => false]
    else
        ["title" => title, "vertical" => true, "label_font_size" => font_size, "title_font_size" => title_font_size, "position_x" => 0.0, "position_y" => 0.35, "color" => "Black", "use_opacity" => false]
    end

    for i in 0:1:num_periods_x-1
        for j in 0:1:num_periods_y-1
            # Plot
            ps = Vector(0:0.01:1.0)
            opacity_val = (isnothing(opacity)) ? (x -> ifelse(x < 1.0, 0.0, x / 20)).(ps) : opacity
            plotter.add_mesh(design.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, opacity=opacity_val, show_scalar_bar=(colorbar & (i == j) & (j == 0)), scalar_bar_args=Dict(scalargs))
            if !full
                plotter.add_mesh(reflectx.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity_val)
                if reflectybool
                    plotter.add_mesh(reflecty.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity_val)
                    plotter.add_mesh(reflectxy.translate((i * W * 2, j * L * 2, 0), inplace=false), color=contour_color, clim=clim, show_scalar_bar=false, opacity=opacity_val)
                end
            end
        end
    end

    return plotter
end

end # module

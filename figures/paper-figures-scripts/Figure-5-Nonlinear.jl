using DistributedEmitterOpt
using PyCall, CairoMakie
const Viz = Base.get_extension(DistributedEmitterOpt, :VisualizationExt); using .Viz
using CSV, DataFrames
using Images, JLD2
using ColorSchemes, Colors

Figure, Axis = CairoMakie.Figure, CairoMakie.Axis
CairoMakie.activate!()

# Set global font and size
CairoMakie.update_theme!(
    font="Times",
    fontsize=22
)

# --- Color Definitions ---
const COLOR_OPTIMIZED_NO_THRESHOLD = "#b097d1"
const COLOR_SPHERES = "#008000"
const COLOR_OPTIMIZED_POINTS = "#CC0000"
const COLOR_GEOMETRY = "#A9A9A9"

function color_for_eth(eth)
    if eth == Inf
        return COLOR_OPTIMIZED_NO_THRESHOLD
    end
    t = clamp((log10(eth) - log10(5.0)) / (log10(25.0) - log10(5.0)), 0.0, 1.0)
    r1, g1, b1 = parse.(Int, [COLOR_OPTIMIZED_POINTS[2:3], COLOR_OPTIMIZED_POINTS[4:5], COLOR_OPTIMIZED_POINTS[6:7]], base=16)
    r2, g2, b2 = parse.(Int, [COLOR_OPTIMIZED_NO_THRESHOLD[2:3], COLOR_OPTIMIZED_NO_THRESHOLD[4:5], COLOR_OPTIMIZED_NO_THRESHOLD[6:7]], base=16)
    r = round(Int, r1 + t * (r2 - r1))
    g = round(Int, g1 + t * (g2 - g1))
    b = round(Int, b1 + t * (b2 - b1))
    return "#$(string(r, base=16, pad=2))$(string(g, base=16, pad=2))$(string(b, base=16, pad=2))"
end



# --- PyVista Setup ---
try
    global pv = pyimport("pyvista")
    global np = pyimport("numpy")
    pv.global_theme.font.family = "times"
    pv.global_theme.transparent_background = true
    global PYVISTA_AVAILABLE = true
catch e
    @warn "PyCall: Failed to import PyVista or NumPy. VTU loading/plotting will be disabled."
    global PYVISTA_AVAILABLE = false
end

# --- Configuration & Paths ---
text_size = 22
title_size = 22
axis_size = 22
linewidth_main = 3.0
marker_size = 30

# Output paths
temp_path = "figures/paper-figures-scripts/temporary/fig5"
mkpath(temp_path)
mkpath("figures/paper-figures-scripts/figures")
final_figure_filename = "figures/paper-figures-scripts/figures/Figure-5-Nonlinear.pdf"

Eth_values = [5, 10, 17.5, 25, Inf]

# --- Helper Functions ---
# Function to plot geometry using Visualization module
# plot_geometry_pyvista and crop_white_margins removed (use Visualization versions)

# --- 1. Generate and Crop Geometry Images ---
if PYVISTA_AVAILABLE
    geom_filenames_cropped = Dict{Float64,String}()
    geom_filenames_raw = Dict{Float64,String}()

    println("--- Generating Geometry Images ---")
    for eth in Eth_values
        geom_path = "figures/paper-figures-scripts/data/Nonlinear/geometries" # Relative to project root

        if eth == Inf
            vtu_filename = joinpath(geom_path, "designinf.vtu")
        else
            vtu_filename = joinpath(geom_path, "design$(replace(string(eth), "." => "-")).vtu")
        end
        color = color_for_eth(eth)

        if !isfile(vtu_filename)
            @warn "VTU file not found, skipping: $vtu_filename"
            continue
        end

        raw_png = joinpath(temp_path, "geom_Eth_$(replace(string(eth), "." => "-"))_raw.png")
        cropped_png = joinpath(temp_path, "geom_Eth_$(replace(string(eth), "." => "-"))_cropped.png")
        geom_filenames_raw[eth] = raw_png
        geom_filenames_cropped[eth] = cropped_png

        try
            mesh = pv.read(vtu_filename)
            if "p" in mesh.array_names
                mesh.active_scalars_name = "p"
            end

            # Calculate bounds
            xl, xr, yl, yr, _, _ = mesh.bounds
            W = xr - xl
            L = yr - yl

            save_geometry_snapshot(np, pv, mesh, raw_png, W, L;
                color=color, # passed as string (hex)
                window_size=(300, 300),
                zoom=1.0,
                axes_viewport=(0.0, 0.0, 0.35, 0.35),
                num_periods_x=2,
                num_periods_y=2,
                contour=true,
                flipy=true
            )

            crop_white_margins(raw_png, cropped_png)
        catch e
            @error "Failed to process geometry for Eth=$eth: $e"
        end
    end
    println("--- Finished Geometry Image Generation ---")
else
    @warn "PyVista not available. Skipping geometry generation."
    geom_filenames_cropped = Dict{Float64,String}()
end

# --- 2. Load CSV Data ---
println("--- Loading CSV Data ---")
csv_path = "figures/paper-figures-scripts/data/Nonlinear/csv-data"
try
    global df_baseline = CSV.read(joinpath(csv_path, "baseline-sweep.csv"), DataFrame, header=["Eth", "Enhancement"])
    global df_sphere = CSV.read(joinpath(csv_path, "sphere-sweep-retry.csv"), DataFrame, header=["Eth", "Enhancement"])
    # global df_single_nonlinear = CSV.read(joinpath(csv_path, "single-nonlinear.csv"), DataFrame, header=["Eth", "Enhancement"])
    global df_multi_nonlinear = CSV.read(joinpath(csv_path, "multiple-nonlinear.csv"), DataFrame, header=["Eth", "Enhancement"])
    @info "Successfully loaded all CSV data."
catch e
    @error "Failed to load CSV data: $e"
    global df_baseline = DataFrame(Eth=[], Enhancement=[])
    global df_sphere = DataFrame(Eth=[], Enhancement=[])
    # global df_single_nonlinear = DataFrame(Eth=[], Enhancement=[])
    global df_multi_nonlinear = DataFrame(Eth=[], Enhancement=[])
end

# --- 3. Create Final Figure (CairoMakie) ---
println("--- Generating Final Figure ---")

fig = Figure(resolution=(1000, 700), fontsize=22, fonts=(; regular="Times", bold="Times Bold"))

# --- Top Panel: Geometries ---
geom_grid = fig[1, 1] = GridLayout()
Label(geom_grid[1, 1:length(Eth_values), Top()], L"(a) Optimized Geometries for Different $E_{\mathrm{th}}$", fontsize=title_size, font=:bold, padding=(0, 0, 5, 0), halign=:center)

loaded_geom_images = Dict{Float64,Any}()
for (i, eth) in enumerate(Eth_values)
    if haskey(geom_filenames_cropped, eth) && isfile(geom_filenames_cropped[eth])
        loaded_geom_images[eth] = load(geom_filenames_cropped[eth])
    else
        loaded_geom_images[eth] = nothing
    end

    ax_geom = Axis(geom_grid[2, i], aspect=DataAspect())
    if !isnothing(loaded_geom_images[eth])
        image!(ax_geom, rotr90(loaded_geom_images[eth]))
    else
        text!(ax_geom, 0.5, 0.5, text=" Eth=$eth \n Image Error", align=(:center, :center), fontsize=10)
    end
    hidedecorations!(ax_geom)
    hidespines!(ax_geom)

    label_grid = GridLayout(geom_grid[3, i])
    color = color_for_eth(eth)
    ax_marker = Axis(label_grid[1, 1], width=30, height=30)
    scatter!(ax_marker, [0.5], [0.5], color=color, marker=:cross, markersize=20, strokewidth=0)
    hidedecorations!(ax_marker)
    hidespines!(ax_marker)
    Label(label_grid[1, 2], "$eth", fontsize=22, halign=:left, padding=(0, 0, 0, 1))
end
Label(geom_grid[4, 1:length(Eth_values), Top()], L"$E_{\mathrm{th}}$ During Optimization", fontsize=22, font=:regular, padding=(0, 0, 10, 0), halign=:center)

# --- Bottom Panel: Performance Plot ---
ax_perf = Axis(fig[2, 1],
    xlabel=L"$E_{\mathrm{th}}$ (Damage Threshold relative to $|\mathbf{E}_{\mathrm{in}}|$)",
    ylabel="Raman Enhancement",
    title="(b) Performance vs. Damage Threshold",
    titlesize=title_size,
    titlefont=:bold,
    yscale=log10,
    xgridwidth=2, ygridwidth=2, xgridcolor=(:black, 0.20), ygridcolor=(:black, 0.20))

if !isempty(df_baseline)
    lines!(ax_perf, df_baseline.Eth, df_baseline.Enhancement, color=COLOR_OPTIMIZED_NO_THRESHOLD, linewidth=linewidth_main, label=L"Optimized w/ $E_{\mathrm{th}}=∞$")
end
if !isempty(df_sphere)
    lines!(ax_perf, df_sphere.Eth, df_sphere.Enhancement, color=COLOR_SPHERES, linewidth=linewidth_main, linestyle=:dash, label="Sphere Baseline")
end
# if !isempty(df_single_nonlinear)
#     lines!(ax_perf, df_single_nonlinear.Eth, df_single_nonlinear.Enhancement, color=COLOR_OPTIMIZED_POINTS, linewidth=linewidth_main, label=L"Optimized w/ $E_{\mathrm{th}}=10$")
# end

if !isempty(df_multi_nonlinear)
    eth_values = collect(df_multi_nonlinear.Eth)
    colors = color_for_eth.(eth_values)
    scatter!(ax_perf, df_multi_nonlinear.Eth, df_multi_nonlinear.Enhancement, color=colors, markersize=marker_size, marker=:cross, strokewidth=0, label=L"Optimized w/ Specific $E_{\mathrm{th}}$")
end

text!(ax_perf, 45, 6e3, text=L"Optimized w/ $E_{\mathrm{th}}=∞$", color=COLOR_OPTIMIZED_NO_THRESHOLD, fontsize=text_size, align=(:left, :center))

save(final_figure_filename, fig)
println("Saved figure to $final_figure_filename")

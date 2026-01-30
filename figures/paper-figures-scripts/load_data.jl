#using Pkg # Comment out if activating externally
using Pkg; Pkg.activate("/Users/ianhammond/GitHub/Emitter3DTopOpt") # Added activation
# Ensure necessary packages are installed
# Pkg.add(["PyCall", "JLD2", "YAML", "Logging"]) # Uncomment if needed

using PyCall
using JLD2
using YAML
using Logging
using Printf # For formatting log messages
using LinearAlgebra # Might be needed if gradient/p_opt are vectors/matrices

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = @__DIR__
YAML_FILE = joinpath(SCRIPT_DIR, "Figures.scp.yaml")

# --- PyCall Setup for PyVista ---
# (Keep the PyCall setup block from the previous version here)
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
# ----------------------------


# --- Load YAML Configuration ---
function load_config(yaml_path=YAML_FILE)
    """Loads the experiment configuration from the YAML file."""
    if !isfile(yaml_path)
        @error "YAML configuration file not found: $yaml_path"
        return nothing
    end
    try
        config = YAML.load_file(yaml_path)
        @info "Successfully loaded configuration from $yaml_path"
        return config
    catch e
        @error "Error parsing YAML file $yaml_path: $e"
        return nothing
    end
end

# --- Global Config Variable (Load once) ---
global CONFIG = load_config()
if isnothing(CONFIG)
    error("Failed to load YAML configuration. Exiting.")
end

# Make LOCAL_SAVE_PATH relative to script directory if it's a relative path
global LOCAL_SAVE_PATH_RAW = get(CONFIG, "local_save_path", "./data/")
global LOCAL_SAVE_PATH = isabspath(LOCAL_SAVE_PATH_RAW) ? LOCAL_SAVE_PATH_RAW : joinpath(SCRIPT_DIR, LOCAL_SAVE_PATH_RAW)
global STANDARD_FILES = get(CONFIG, "standard_files", Dict())

# --- Data Loading Function ---
function get_data(group_key::String, variation_key::String, file_key::String)
    """
    Loads and potentially processes data for a specific experiment, variation, and file type.

    Args:
        group_key (String): Top-level key (e.g., "Constrained_Metal").
        variation_key (String): Variation key or "nominal".
        file_key (String): File key (e.g., "spectral", "results").

    Returns:
        Loaded/processed data (PyObject, Dict, etc.) or nothing on error.
    """
    if isnothing(CONFIG)
        @error "Configuration not loaded. Cannot get data."
        return nothing
    end

    # 1. Find the correct experiment entry (Same logic as before)
    if !haskey(CONFIG["experiments"], group_key)
        @error "Group key '$group_key' not found in YAML experiments."
        return nothing
    end
    group_data = CONFIG["experiments"][group_key]
    is_nominal_case = (variation_key == "nominal")
    local entry
    local local_path_base::String
    if is_nominal_case
        if !haskey(group_data, "nominal")
            @error "Nominal entry not found for group '$group_key'."
            return nothing
        end
        entry = group_data["nominal"]
        local_path_base = joinpath(LOCAL_SAVE_PATH, group_key, "nominal")
    else
        if !haskey(group_data, "variations") || !haskey(group_data["variations"], variation_key)
            @error "Variation key '$variation_key' not found for group '$group_key'."
            return nothing
        end
        entry = group_data["variations"][variation_key]
        local_path_base = joinpath(LOCAL_SAVE_PATH, group_key, variation_key)
    end

    # 2. Determine the filename (Same logic as before)
    current_files_template = deepcopy(STANDARD_FILES)
    if isa(entry, Dict) && haskey(entry, "files") && !isnothing(entry["files"])
        merge!(current_files_template, entry["files"])
    end
    if !haskey(current_files_template, file_key)
        @error "File key '$file_key' not found in standard files or entry override for $group_key/$variation_key."
        return nothing
    end
    file_name = current_files_template[file_key]

    # 3. Construct the full local path (Same logic as before)
    local_file_path = joinpath(local_path_base, file_name)

    # 4. Check if file exists locally (Same logic as before)
    if !isfile(local_file_path)
        @error "File not found locally: $local_file_path"
        @warn "Attempted to load for $group_key/$variation_key/$file_key."
        return nothing
    end

    # 5. Load AND PROCESS based on file key or type
    @info "Loading $file_key from: $local_file_path"
    try
        # --- JLD2 Loading and Processing ---
        if file_key == "spectral" || file_key == "angular"
            # Expecting a Tuple like (Vector, Vector, Vector, Vector)
            loaded_data = JLD2.load_object(local_file_path)
            # Check structure: Tuple of length 4 with vectors inside
            if isa(loaded_data, Tuple) && length(loaded_data) == 4 && all(isa.(loaded_data, AbstractVector))
                 @info "Successfully loaded '$file_key' data as Tuple{Vector, Vector, Vector, Vector}."
                 # Return a dictionary with descriptive keys
                 if file_key == "spectral"
                    return Dict(
                        "wavelengths" => loaded_data[1],
                        "g_y" => loaded_data[2],
                        "g_x" => loaded_data[3],
                        "g_combined" => loaded_data[4] # Name for gy+gx
                        )
                 else # angular
                    return Dict(
                        "angles" => loaded_data[1],
                        "g_y" => loaded_data[2],
                        "g_x" => loaded_data[3],
                        "g_combined" => loaded_data[4] # Name for gy+gx
                        )
                 end
            else
                 @warn "Loaded '$file_key' data is not the expected Tuple{Vector, Vector, Vector, Vector} format. Type: $(typeof(loaded_data)). Returning raw data."
                 return loaded_data # Return whatever was loaded for debugging
            end

        elseif file_key == "results"
             # Expecting Tuple: (g_opt, p_opt, grad, g_ar, g_biggest, p_biggest)
             loaded_tuple = JLD2.load_object(local_file_path)
             if isa(loaded_tuple, Tuple) && length(loaded_tuple) >= 5 # Check length needed
                 @info "Successfully loaded 'results' data Tuple."
                 # Extract and rename into a dictionary
                 results_dict = Dict(
                     "design" => loaded_tuple[2],    # p_opt
                     "gradient" => loaded_tuple[3],  # grad
                     "g_array" => loaded_tuple[4],   # g_ar
                     "g_final" => loaded_tuple[5]    # g_biggest
                 )
                 return results_dict
             else
                 @warn "Loaded 'results' data is not the expected Tuple format or length. Type: $(typeof(loaded_tuple)). Returning raw data."
                 return loaded_tuple # Return raw for debugging
             end

        # --- VTU Loading ---
        elseif endswith(file_name, ".vtu")
            if !PYVISTA_AVAILABLE
                 @error "Cannot load VTU file - PyVista interface not available."
                 return nothing
            end
            mesh = pv.read(local_file_path)
            @info "Successfully loaded VTU file using PyVista via PyCall."
            return mesh # Returns a PyObject

        # --- Fallback ---
        else
             @warn "Unhandled file key '$file_key' with extension $(splitext(file_name)[2]). Attempting generic JLD2 load."
             try
                 loaded_data = JLD2.load_object(local_file_path)
                 @info "Loaded '$file_key' using generic JLD2 load_object."
                 return loaded_data
             catch jld_err
                 @warn "Generic JLD2 load failed for '$file_key': $jld_err. Returning nothing."
                 return nothing
             end
        end
    catch e
        @error "Error loading/processing file $local_file_path for key '$file_key': $e"
        # Base.showerror(stderr, e, catch_backtrace()) # Uncomment for detailed stacktrace
        return nothing
    end
end
module FigureLoader

using JLD2
using FileIO
using PyCall
using CSV
using DataFrames

export get_figure_data

# Hardcoded base path relative to this file
# Assumes structure: src/Visualization/FigureLoader.jl
# Data is at: figures/paper-figures-scripts/data
const DATA_REL_PATH = "../../figures/paper-figures-scripts/data"

function get_data_root()
    # Normalize path relative to the file's location
    joinpath(@__DIR__, DATA_REL_PATH)
end

"""
    get_figure_data(group_key::String, variation_key::String, file_key::String)

Loads data for figure generation. Replaces the legacy `get_data` and YAML config.
"""
function get_figure_data(group_key::String, variation_key::String, file_key::String)
    data_root = get_data_root()

    # Construct base path
    # Handle the slightly inconsistent structure (sometimes nominal is a dir, sometimes direct)
    # Based on the migration script:
    # Freeform_Metal/nominal
    # Constrained_Metal/nominal
    # Constrained_Metal/polarization_Bi
    # Nonlinear/csv-data

    local_path = ""

    if group_key == "Nonlinear"
        if file_key == "csv"
            local_path = joinpath(data_root, "Nonlinear", "csv-data", variation_key) # variation_key here acts as filename
        elseif file_key == "geometry"
            local_path = joinpath(data_root, "Nonlinear", "geometries", variation_key) # variation_key here acts as filename
        end
    elseif group_key == "Starting"
        # variation_key is material (metal/dielectric)
        local_path = joinpath(data_root, "data_starting", variation_key, "y_design.vtu")
    elseif group_key == "Bonus"
        # variation_key is filename
        local_path = joinpath(data_root, "post-anisotropy-bonus", variation_key)
    else
        # Standard Experiment Structure
        # data/<Group>/<Variation>/<File>

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
        # Fallback for missing Constrained_Metal/nominal/spectral.jld2
        # Use Bonus iso-spectral as it represents the same physics (Isotropic Metal)
        if group_key == "Constrained_Metal" && variation_key == "nominal" && file_key == "spectral"
            alt_path = joinpath(data_root, "post-anisotropy-bonus", "iso-spectral.jld2")
            if isfile(alt_path)
                println("Warning: Main file missing. Using fallback: $alt_path")
                data = load(alt_path)
                # Re-use logic below
                local_path = alt_path # update for logging/extension check logic
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
        # Extract single object if it was saved that way, or return dict
        # The legacy scripts expect specific formats (Dict for spectral, Tuple for results)
        # We try to mimic that here.

        if file_key == "spectral" && haskey(data, "single_stored_object")
            # Legacy JLD2 might load as Dict("single_stored_object" => Tuple(...))
            val = data["single_stored_object"]
            if isa(val, Tuple) && length(val) == 4
                return Dict("wavelengths" => val[1], "g_y" => val[2], "g_x" => val[3], "g_combined" => val[4])
            end
            return val
        elseif file_key == "results" && haskey(data, "single_stored_object")
            val = data["single_stored_object"]
            if isa(val, Tuple)
                # Convert standard results tuple to Dict
                res = Dict{String,Any}()
                # Mapping based on legacy Figure-4 logic and observations
                # 1: Final Objective? (Float64)
                # 2: Design (Vector)
                # 3: Gradient (Vector)
                # 4: History (Vector{Any})
                # 5: g_final (Float64)
                # 6: p_biggest (Vector) - Optional
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

end # module

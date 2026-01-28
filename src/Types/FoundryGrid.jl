"""
    FoundryGrid

2D rectilinear grid for lithography-compatible topology optimization.
Design parameters live on a regular (x, y) grid and are interpolated
to the 3D FEM mesh via bilinear interpolation.

## Fields
- `x` — Grid x-coordinates (monotonic)
- `y` — Grid y-coordinates (monotonic)
- `params` — Design parameter matrix (nx × ny)
- `nodes` — Cell vertex coordinates for Jacobian computation
- `jacobian` — Cached ∂(mesh quadrature)/∂(grid param) mapping
"""
mutable struct FoundryGrid
    x::Vector{Float64}
    y::Vector{Float64}
    params::Matrix{Float64}
    nodes::Vector
    jacobian::Matrix{Float64}

    function FoundryGrid(x, y, params, nodes)
        nx, ny = length(x), length(y)
        nnodes = length(nodes)
        jacobian = zeros(Float64, nx * ny, nnodes)
        new(x, y, params, nodes, jacobian)
    end
end

Base.show(io::IO, g::FoundryGrid) = print(io, "FoundryGrid($(length(g.x))×$(length(g.y)))")

# ═══════════════════════════════════════════════════════════════════════════════
# Grid construction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    getgrid(labels, Ω, np, nodes; sizes=nothing) -> FoundryGrid

Build a FoundryGrid from mesh topology. The grid covers the x-y extent of the
design region with approximately √np points per side.
"""
function getgrid(labels, Ω, np, nodes; sizes=nothing)
    if isnothing(sizes)
        # Extract design region bounds from Design volume cells
        dimension = num_cell_dims(Ω.model)
        cellmask_d = Gridap.Geometry.get_face_mask(labels, "Design", dimension)

        # Get coordinates of all nodes in design region cells
        design_cell_ids = findall(cellmask_d)

        if isempty(design_cell_ids)
            error("No Design cells found in mesh. Check physical group 'Design' exists.")
        end

        # Get cell-to-node connectivity and extract bounds
        cell_coords = Gridap.Geometry.get_cell_coordinates(Ω)

        all_x = Float64[]
        all_y = Float64[]
        for cell_id in design_cell_ids
            cell_nodes = cell_coords[cell_id]
            for node in cell_nodes
                push!(all_x, node[1])
                push!(all_y, node[2])
            end
        end

        lowx, highx = minimum(all_x), maximum(all_x)
        lowy, highy = minimum(all_y), maximum(all_y)
        lowz, highz = 0.0, 0.0  # z not needed for 2D grid
    else
        (lowx, highx, lowy, highy, lowz, highz) = sizes
    end

    Δx, Δy, Δz = (highx - lowx), (highy - lowy), (highz - lowz)

    # Compute grid density to match np DOFs
    ρ = if Δz > 0.0
        √(np) / (Δx * Δy * Δz)^(1 / 3)
    else
        √(np) / (Δx * Δy)^(1 / 2)
    end

    nx, ny = Int(ceil(Δx * ρ)), Int(ceil(Δy * ρ))
    x = collect(range(lowx, highx, length=nx))
    y = collect(range(lowy, highy, length=ny))
    params = zeros(Float64, nx, ny)

    FoundryGrid(x, y, params, nodes)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Grid → Mesh interpolation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pf_grid(r, params, gridx, gridy) -> Real

Bilinear interpolation of grid parameters at point r = (x, y, z).
"""
function pf_grid(r, params::Matrix{Float64}, gridx::Vector{Float64}, gridy::Vector{Float64})
    x, y = r[1], r[2]

    # Find bracketing indices
    rx = searchsortedfirst(gridx, x)
    ry = searchsortedfirst(gridy, y)

    # Clamp to grid bounds
    rx = clamp(rx, 2, length(gridx))
    ry = clamp(ry, 2, length(gridy))
    lx, ly = rx - 1, ry - 1

    # Distances to grid points
    Δx_r = gridx[rx] - x
    Δx_l = x - gridx[lx]
    Δy_r = gridy[ry] - y
    Δy_l = y - gridy[ly]

    # Bilinear interpolation
    denom = (Δy_r + Δy_l) * (Δx_r + Δx_l)
    if denom < 1e-12
        return params[rx, ry]
    end

    pf = (params[lx, ly] * Δx_r * Δy_r +
          params[rx, ly] * Δx_l * Δy_r +
          params[lx, ry] * Δx_r * Δy_l +
          params[rx, ry] * Δx_l * Δy_l) / denom

    return pf
end

"""Convenience method using FoundryGrid."""
pf_grid(r, grid::FoundryGrid) = pf_grid(r, grid.params, grid.x, grid.y)

# ═══════════════════════════════════════════════════════════════════════════════
# Jacobian for adjoint
# ═══════════════════════════════════════════════════════════════════════════════

"""
    getjacobian!(grid::FoundryGrid) -> Matrix

Compute and cache the Jacobian mapping from grid parameters to mesh cell centers.
Used for backpropagating gradients from mesh to grid.
"""
function getjacobian!(grid::FoundryGrid)
    fill!(grid.jacobian, 0.0)
    nx, ny = length(grid.x), length(grid.y)
    tempjacobian = reshape(grid.jacobian, (nx, ny, size(grid.jacobian, 2)))

    for (i, element) in enumerate(grid.nodes)
        _update_jacobian!(i, element, tempjacobian, grid.x, grid.y)
    end

    grid.jacobian[:, :] = reshape(tempjacobian, (nx * ny, size(grid.jacobian, 2)))[:, :]
    return grid.jacobian
end

"""Update jacobian for a single mesh element (assumes 4 vertices per face)."""
function _update_jacobian!(i, element, tempjacobian, gridx, gridy)
    for vertex in element
        _update_jacobian_vertex!(i, vertex, 0.25, tempjacobian, gridx, gridy)
    end
end

function _update_jacobian_vertex!(i, vertex, weight, tempjacobian, gridx, gridy)
    x, y = vertex[1], vertex[2]

    # Find bracketing indices
    rx = searchsortedfirst(gridx, x)
    ry = searchsortedfirst(gridy, y)

    rx = clamp(rx, 2, length(gridx))
    ry = clamp(ry, 2, length(gridy))
    lx, ly = rx - 1, ry - 1

    Δx_r = gridx[rx] - x
    Δx_l = x - gridx[lx]
    Δy_r = gridy[ry] - y
    Δy_l = y - gridy[ly]

    denom = (Δy_r + Δy_l) * (Δx_r + Δx_l)
    if denom < 1e-12
        tempjacobian[rx, ry, i] += weight
        return
    end

    # Bilinear weights
    tempjacobian[lx, ly, i] += weight * Δx_r * Δy_r / denom
    tempjacobian[rx, ly, i] += weight * Δx_l * Δy_r / denom
    tempjacobian[lx, ry, i] += weight * Δx_r * Δy_l / denom
    tempjacobian[rx, ry, i] += weight * Δx_l * Δy_l / denom
end

"""
    FoundryGrid

2D rectilinear grid for lithography-compatible topology optimization.
Design parameters live on a regular (x, y) grid and are interpolated
to the 3D FEM mesh via bilinear interpolation.

## Fields
- `x` — Grid x-coordinates (monotonic)
- `y` — Grid y-coordinates (monotonic)
- `params` — Design parameter matrix (nx × ny)
- `nodes` — Mesh node coordinates for interpolation/adjoint mapping
- `adj_idx` — Cached grid indices for adjoint scatter (4 × nnodes)
- `adj_w` — Cached weights for adjoint scatter (4 × nnodes)
- `adj_ready` — Whether adjoint cache is initialized
"""
mutable struct FoundryGrid
    x::Vector{Float64}
    y::Vector{Float64}
    params::Matrix{Float64}
    nodes::Vector
    adj_idx::Matrix{Int}
    adj_w::Matrix{Float64}
    adj_ready::Bool

    function FoundryGrid(x, y, params, nodes)
        nnodes = length(nodes)
        adj_idx = zeros(Int, 4, nnodes)
        adj_w = zeros(Float64, 4, nnodes)
        new(x, y, params, nodes, adj_idx, adj_w, false)
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
# Adjoint scatter cache for bilinear interpolation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    getjacobian!(grid::FoundryGrid)

Dense global Jacobian storage was removed to avoid catastrophic memory usage.
Use `apply_grid_adjoint!` for gradient backpropagation.
"""
function getjacobian!(grid::FoundryGrid)
    error("Dense FoundryGrid Jacobian was removed; use apply_grid_adjoint! instead.")
end

"""
    prepare_grid_adjoint!(grid::FoundryGrid)

Precompute the 4-point bilinear weights and grid indices for each mesh node.
This lets us apply the adjoint mapping without forming a dense Jacobian.
"""
function prepare_grid_adjoint!(grid::FoundryGrid)
    nx, ny = length(grid.x), length(grid.y)
    lin = LinearIndices((nx, ny))

    @inbounds for (i, node) in enumerate(grid.nodes)
        x, y = node[1], node[2]

        rx = searchsortedfirst(grid.x, x)
        ry = searchsortedfirst(grid.y, y)
        rx = clamp(rx, 2, nx)
        ry = clamp(ry, 2, ny)
        lx, ly = rx - 1, ry - 1

        Δx_r = grid.x[rx] - x
        Δx_l = x - grid.x[lx]
        Δy_r = grid.y[ry] - y
        Δy_l = y - grid.y[ly]

        denom = (Δy_r + Δy_l) * (Δx_r + Δx_l)
        if denom < 1e-12
            idx = lin[rx, ry]
            grid.adj_idx[:, i] .= idx
            grid.adj_w[1, i] = 1.0
            grid.adj_w[2, i] = 0.0
            grid.adj_w[3, i] = 0.0
            grid.adj_w[4, i] = 0.0
            continue
        end

        grid.adj_idx[1, i] = lin[lx, ly]
        grid.adj_idx[2, i] = lin[rx, ly]
        grid.adj_idx[3, i] = lin[lx, ry]
        grid.adj_idx[4, i] = lin[rx, ry]

        grid.adj_w[1, i] = Δx_r * Δy_r / denom
        grid.adj_w[2, i] = Δx_l * Δy_r / denom
        grid.adj_w[3, i] = Δx_r * Δy_l / denom
        grid.adj_w[4, i] = Δx_l * Δy_l / denom
    end

    grid.adj_ready = true
    return grid
end

"""
    apply_grid_adjoint!(out, grid, node_grad) -> out

Apply the adjoint of the grid→mesh interpolation without forming a dense matrix.
`node_grad` is ∂g/∂pf at mesh nodes (Pf DOFs). `out` is ∂g/∂p on the 2D grid.
"""
function apply_grid_adjoint!(out::Vector{Float64}, grid::FoundryGrid, node_grad::Vector{Float64})
    if !grid.adj_ready
        prepare_grid_adjoint!(grid)
    end
    fill!(out, 0.0)

    @inbounds for i in eachindex(node_grad)
        gi = node_grad[i]
        out[grid.adj_idx[1, i]] += grid.adj_w[1, i] * gi
        out[grid.adj_idx[2, i]] += grid.adj_w[2, i] * gi
        out[grid.adj_idx[3, i]] += grid.adj_w[3, i] * gi
        out[grid.adj_idx[4, i]] += grid.adj_w[4, i] * gi
    end

    return out
end

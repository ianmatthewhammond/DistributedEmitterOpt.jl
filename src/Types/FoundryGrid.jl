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
    legacy_jacobian::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    legacy_ready::Bool

    function FoundryGrid(x, y, params, nodes)
        nnodes = length(nodes)
        adj_idx = zeros(Int, 4, nnodes)
        adj_w = zeros(Float64, 4, nnodes)
        new(x, y, params, nodes, adj_idx, adj_w, false, nothing, false)
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
        # Legacy-compatible bounds: prefer DesignNodes (0D) if present.
        # This matches old foundry grid construction behavior.
        got_designnodes = false
        lowx = highx = lowy = highy = lowz = highz = 0.0

        try
            cellmask_dn = Gridap.Geometry.get_face_mask(labels, "DesignNodes", 0)
            points = Ω.grid.node_coordinates
            masked = cellmask_dn .* points

            dx = sort([p[1] for p in masked if norm(p) > 0.0])
            dy = sort([p[2] for p in masked if norm(p) > 0.0])
            dz = sort([p[3] for p in masked if norm(p) > 0.0])

            if !isempty(dx) && !isempty(dy) && !isempty(dz)
                lowx, highx = dx[1], dx[end]
                lowy, highy = dy[1], dy[end]
                lowz, highz = dz[1], dz[end]
                got_designnodes = true
            end
        catch
            got_designnodes = false
        end

        if !got_designnodes
            # Fallback: extract x/y from Design cells and z from design bounds helper.
            dimension = num_cell_dims(Ω.model)
            cellmask_d = Gridap.Geometry.get_face_mask(labels, "Design", dimension)
            design_cell_ids = findall(cellmask_d)

            if isempty(design_cell_ids)
                error("No Design cells found in mesh. Check physical group 'Design' exists.")
            end

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
            lowz, highz = getdesignz(labels, Ω)
        end
    else
        (lowx, highx, lowy, highy, lowz, highz) = sizes
    end

    Δx, Δy, Δz = (highx - lowx), (highy - lowy), (highz - lowz)

    # Legacy density rule: include design thickness when available.
    ρ = if Δz > 0.0
        √(np) / (Δx * Δy * Δz)^(1 / 3)
    else
        √(np) / (Δx * Δy)^(1 / 2)
    end

    nx, ny, _ = Int(ceil(Δx * ρ)), Int(ceil(Δy * ρ)), Int(ceil(Δz * ρ))
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

"""
    prepare_legacy_grid_jacobian!(grid::FoundryGrid)

Build and cache the explicit node<-grid interpolation Jacobian used by the
legacy foundry mapping mode.
"""
function prepare_legacy_grid_jacobian!(grid::FoundryGrid)
    if grid.legacy_ready
        return grid
    end
    if !grid.adj_ready
        prepare_grid_adjoint!(grid)
    end

    nnodes = length(grid.nodes)
    ngrid = length(grid.x) * length(grid.y)
    rows = Vector{Int}(undef, 4 * nnodes)
    cols = Vector{Int}(undef, 4 * nnodes)
    vals = Vector{Float64}(undef, 4 * nnodes)

    k = 1
    @inbounds for i in 1:nnodes
        for j in 1:4
            rows[k] = i
            cols[k] = grid.adj_idx[j, i]
            vals[k] = grid.adj_w[j, i]
            k += 1
        end
    end

    grid.legacy_jacobian = sparse(rows, cols, vals, nnodes, ngrid)
    grid.legacy_ready = true
    return grid
end

"""
    apply_grid_adjoint_legacy!(out, grid, node_grad) -> out

Apply adjoint using the explicit Jacobian assembly, matching legacy workflow.
"""
function apply_grid_adjoint_legacy!(out::Vector{Float64}, grid::FoundryGrid, node_grad::Vector{Float64})
    if !grid.legacy_ready
        prepare_legacy_grid_jacobian!(grid)
    end
    fill!(out, 0.0)
    out .= transpose(grid.legacy_jacobian) * node_grad
    return out
end

"""
    build_foundry_pf(pf_vec, sim, control)

Construct the foundry-mode filtered field on `sim.Pf` according to the selected
projection mapping mode.
"""
function build_foundry_pf(pf_vec::Vector{Float64}, sim, control)
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    sim.grid.params[:, :] .= reshape(pf_vec, (nx, ny))

    if control.foundry_projection_mode == :legacy
        return (r -> pf_grid(r, sim.grid))
    elseif control.foundry_projection_mode == :current
        pf_vals = [pf_grid(node, sim.grid) for node in sim.grid.nodes]
        return FEFunction(sim.Pf, pf_vals)
    else
        error("Unsupported foundry_projection_mode=$(control.foundry_projection_mode). Use :current or :legacy.")
    end
end

"""
    map_foundry_adjoint!(out, sim, control, node_grad)

Map sensitivities from `sim.Pf` DOFs back to foundry grid DOFs with the matching
adjoint implementation for the selected projection mapping mode.
"""
function map_foundry_adjoint!(out::Vector{Float64}, sim, control, node_grad::Vector{Float64})
    if control.foundry_projection_mode == :legacy
        return apply_grid_adjoint_legacy!(out, sim.grid, node_grad)
    elseif control.foundry_projection_mode == :current
        return apply_grid_adjoint!(out, sim.grid, node_grad)
    else
        error("Unsupported foundry_projection_mode=$(control.foundry_projection_mode). Use :current or :legacy.")
    end
end

"""
    scatter_point_legacy!(out, grid, x, weight)

Legacy bilinear scatter from a physical point `x` with quadrature weight `weight`
onto 2D grid DOFs.
"""
function scatter_point_legacy!(out::Vector{Float64}, grid::FoundryGrid, x, weight::Float64)
    gx, gy = grid.x, grid.y
    nx, ny = length(gx), length(gy)
    lin = LinearIndices((nx, ny))
    xx, yy = x[1], x[2]

    rx = searchsortedfirst(gx, xx)
    if rx > nx
        rx -= 1
    end
    lx = rx - 1

    ry = searchsortedfirst(gy, yy)
    if ry > ny
        ry -= 1
    end
    ly = ry - 1

    Δx_r = gx[rx] - xx
    Δy_r = gy[ry] - yy

    @inbounds if (ly == 0) && (lx == 0)
        out[lin[rx, ry]] += weight
    elseif ly == 0
        Δx_l = xx - gx[lx]
        den = (Δx_r + Δx_l)
        out[lin[lx, ry]] += weight * Δx_r / den
        out[lin[rx, ry]] += weight * Δx_l / den
    elseif lx == 0
        Δy_l = yy - gy[ly]
        den = (Δy_r + Δy_l)
        out[lin[rx, ly]] += weight * Δy_r / den
        out[lin[rx, ry]] += weight * Δy_l / den
    else
        Δx_l = xx - gx[lx]
        Δy_l = yy - gy[ly]
        den = (Δy_r + Δy_l) * (Δx_r + Δx_l)
        out[lin[lx, ly]] += weight * Δx_r * Δy_r / den
        out[lin[rx, ly]] += weight * Δx_l * Δy_r / den
        out[lin[lx, ry]] += weight * Δx_r * Δy_l / den
        out[lin[rx, ry]] += weight * Δx_l * Δy_l / den
    end

    return out
end

"""
    accumulate_density_to_grid!(out, grid, dΩ, density)

Integrate a scalar cell field `density` over `dΩ` and scatter each quadrature
point contribution to 2D grid DOFs using the legacy bilinear rule.
"""
function accumulate_density_to_grid!(out::Vector{Float64}, grid::FoundryGrid, dΩ, density)
    p = Gridap.CellData.get_cell_points(dΩ)
    cell_vals = density(p)
    cell_w = get_cell_weights_legacy(dΩ)
    cell_x = Gridap.CellData.get_array(p)

    @inbounds for cell in 1:length(cell_vals)
        vals = cell_vals[cell]
        ws = cell_w[cell]
        xs = cell_x[cell]
        nq = length(ws)
        for q in 1:nq
            scatter_point_legacy!(out, grid, xs[q], ws[q] * vals[q])
        end
    end

    return out
end

function get_cell_weights_legacy(dΩ)
    get_cell_weights_legacy(dΩ.quad)
end

function get_cell_weights_legacy(quad::Gridap.CellData.CellQuadrature)
    pd = Gridap.CellData.PhysicalDomain()
    rd = Gridap.CellData.ReferenceDomain()
    if quad.data_domain_style == pd && quad.integration_domain_style == pd
        return quad.cell_weight
    elseif quad.data_domain_style == rd && quad.integration_domain_style == pd
        cell_map = Gridap.CellData.get_cell_map(quad.trian)
        cell_Jt = Gridap.Arrays.lazy_map(∇, cell_map)
        cell_Jtx = Gridap.Arrays.lazy_map(Gridap.CellData.evaluate, cell_Jt, quad.cell_point)
        cell_m = Gridap.Arrays.lazy_map(Gridap.Arrays.Broadcasting(Gridap.TensorValues.meas), cell_Jtx)
        return Gridap.Arrays.lazy_map(Gridap.Arrays.Broadcasting(*), quad.cell_weight, cell_m)
    elseif quad.data_domain_style == rd && quad.integration_domain_style == rd
        return quad.cell_weight
    else
        error("Unsupported quadrature domain style for legacy cell weights")
    end
end

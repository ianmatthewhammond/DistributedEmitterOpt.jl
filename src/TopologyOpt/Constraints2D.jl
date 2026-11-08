"""
    Constraints2D

Grid-based linewidth constraints for 2D DOF mode.
Uses Zygote AD for gradient computation through the filter-project pipeline.
"""

import Zygote

# ═══════════════════════════════════════════════════════════════════════════════
# Legacy constraint helpers (ported from old code)
# ═══════════════════════════════════════════════════════════════════════════════

function centeredfft(arr::AbstractArray, newshape::NTuple{N,Int}) where {N}
    currshape = size(arr)
    startind = floor.((currshape .- newshape) ./ 2)
    endind = startind .+ newshape
    myslice = [Int(st) + 1:Int(en) for (st, en) in zip(startind, endind)]
    return arr[myslice...]
end

function npgradient(ar::AbstractArray)
    if length(size(ar)) == 1
        front = ar[2] - ar[1]
        middle = (ar[3:end] .- ar[1:end - 2]) ./ 2
        back = ar[end] - ar[end - 1]
        rarray = [front; middle; back]
    elseif length(size(ar)) == 2
        front1 = reshape(ar[2, 1:end] .- ar[1, 1:end], 1, :)
        middle1 = (ar[3:end, 1:end] .- ar[1:end - 2, 1:end]) ./ 2
        back1 = reshape(ar[end, 1:end] .- ar[end - 1, 1:end], 1, :)
        rarray1 = [front1; middle1; back1]
        rarray1 = reshape(rarray1, 1, size(rarray1)...)
        front2 = reshape(ar[1:end, 2] .- ar[1:end, 1], :, 1)
        middle2 = (ar[1:end, 3:end] .- ar[1:end, 1:end - 2]) ./ 2
        back2 = reshape(ar[1:end, end] .- ar[1:end, end - 1], :, 1)
        rarray2 = hcat(front2, middle2, back2)
        rarray2 = reshape(rarray2, 1, size(rarray2)...)
        rarray = cat(rarray1, rarray2, dims=1)
    else
        throw(ErrorException("Not Implemented"))
    end
    return rarray
end

"""
    gradient_sq(f, resolution; mirror, halo=(false, false)) -> Matrix

|∇f|² at the cell's nodes from central differences, in physical units. Per axis, the neighbour
across a wall is:
- with `halo` set for that axis, the extra node `f` carries beyond the wall (the filter's own value
  there, e.g. `convolvedcti(...; halo=true)` on a tiled axis); the cell is `f` without it;
- otherwise, with `mirror` (a Bool or an (x, y) pair) set, the reflected inner node, so the normal
  derivative on the wall is zero: DCT-I on an untiled cell, whose walls are mirrors;
- otherwise the node on the opposite edge: a periodic cell of the cell's size, as `convolvefft` tiles it.
"""
function gradient_sq(f::AbstractMatrix, resolution::Float64; mirror, halo::NTuple{2,Bool}=(false, false))
    mx, my = mirror isa Bool ? (mirror, mirror) : mirror
    hx, hy = halo
    rows = hx ? (2:size(f, 1)-1) : (1:size(f, 1))
    cols = hy ? (2:size(f, 2)-1) : (1:size(f, 2))
    fx = hx ? f[:, cols] : wall_pad(f[:, cols], mx, 1)
    fy = hy ? f[rows, :] : wall_pad(f[rows, :], my, 2)
    gx = (fx[3:end, :] .- fx[1:end-2, :]) .* (resolution / 2)
    gy = (fy[:, 3:end] .- fy[:, 1:end-2]) .* (resolution / 2)
    return gx .^ 2 .+ gy .^ 2
end

# One node beyond each wall along `dim`: the reflected inner node (mirror) or the opposite edge.
function wall_pad(f::AbstractMatrix, mirror::Bool, dim::Int)
    if dim == 1
        return mirror ? vcat(f[2:2, :], f, f[end-1:end-1, :]) : vcat(f[end:end, :], f, f[1:1, :])
    else
        return mirror ? hcat(f[:, 2:2], f, f[:, end-1:end-1]) : hcat(f[:, end:end], f, f[:, 1:1])
    end
end

# The cell inside a field that carries a one-node halo on the axes flagged in `halo`.
function strip_halo(f::AbstractMatrix, halo::NTuple{2,Bool})
    rows = halo[1] ? (2:size(f, 1)-1) : (1:size(f, 1))
    cols = halo[2] ? (2:size(f, 2)-1) : (1:size(f, 2))
    return f[rows, cols]
end

"""
    indicator_solid(x, c0, filter_f, threshold_f, resolution; mirror=true, halo=(false, false)) -> Matrix

Solid indicator: the projected field where the filtered field is flat. `mirror` and `halo` match
the filter's boundary (see `gradient_sq`); `filter_f` returns the halo on the axes flagged in `halo`.
"""
function indicator_solid(x::AbstractMatrix, c0::Float64, filter_f, threshold_f, resolution::Float64;
    mirror=true, halo::NTuple{2,Bool}=(false, false))
    filtered = filter_f(x)
    design_field = threshold_f(strip_halo(filtered, halo))
    return design_field .* exp.(-c0 .* gradient_sq(filtered, resolution; mirror, halo))
end

"""
    indicator_void(x, c0, filter_f, threshold_f, resolution; mirror=true, halo=(false, false)) -> Matrix
"""
function indicator_void(x::AbstractMatrix, c0::Float64, filter_f, threshold_f, resolution::Float64;
    mirror=true, halo::NTuple{2,Bool}=(false, false))
    filtered = filter_f(x)
    design_field = threshold_f(strip_halo(filtered, halo))
    return (1 .- design_field) .* exp.(-c0 .* gradient_sq(filtered, resolution; mirror, halo))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Control functions (objective for constraint)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    control_solid(p; c0, ηe, filter_f, threshold_f, resolution, mirror=true, halo=(false, false)) -> Real
"""
function control_solid(p::AbstractMatrix; c0::Float64, ηe::Float64,
    filter_f, threshold_f, resolution::Float64, mirror=true, halo::NTuple{2,Bool}=(false, false))
    filtered_field = strip_halo(filter_f(p), halo)
    I_s = flatten(indicator_solid(p, c0, filter_f, threshold_f, resolution; mirror, halo))
    flt = flatten(filtered_field)
    minfield = min.(flt .- ηe, 0.0) .^ 2
    return sum(I_s .* minfield) / length(flt)
end

"""
    control_void(p; c0, ηd, filter_f, threshold_f, resolution, mirror=true, halo=(false, false)) -> Real
"""
function control_void(p::AbstractMatrix; c0::Float64, ηd::Float64,
    filter_f, threshold_f, resolution::Float64, mirror=true, halo::NTuple{2,Bool}=(false, false))
    filtered_field = strip_halo(filter_f(p), halo)
    I_v = flatten(indicator_void(p, c0, filter_f, threshold_f, resolution; mirror, halo))
    flt = flatten(filtered_field)
    minfield = min.(ηd .- flt, 0.0) .^ 2
    return sum(I_v .* minfield) / length(flt)
end

flatten(x) = vec(x)

# ═══════════════════════════════════════════════════════════════════════════════
# NLopt constraint interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    glc_solid(x, grad; sim, control) -> Float64

Solid linewidth constraint for NLopt. Constraint satisfied when ≤ 0.
"""
function glc_solid(x::Vector{Float64}, grad::Vector{Float64}; sim, control)
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    Lx = sim.grid.x[end] - sim.grid.x[1]
    Ly = sim.grid.y[end] - sim.grid.y[1]
    resolution = (nx - 1) / Lx

    # Build filter/threshold functions
    R = control.R_filter[1]
    h = conic_filter(R, Lx, Ly, resolution)
    # On an axis convolvedcti tiles, the wall neighbour is the next copy's filtered value (the halo);
    # on an untiled axis DCT-I's walls are mirrors; the FFT filter is periodic.
    halo = control.use_dct ? dcti_tiled((nx, ny), size(h)) : (false, false)
    filter_f = p -> control.use_dct ? convolvedcti(p; h, halo=any(halo)) : convolvefft(p; h)
    threshold_f = p -> smoothed_projection(p, control, sim)

    # Hyperparameters
    ηe = control.η_erosion
    b1 = control.b1
    c0 = 64 * R^2

    # Reshape
    p = reshape(x, (nx, ny))

    # Objective
    M1 = p -> control_solid(p; c0, ηe, filter_f, threshold_f, resolution, mirror=control.use_dct, halo)

    # Gradient via Zygote
    if length(grad) > 0
        g1 = Zygote.gradient(M1, p)
        grad[:] = flatten(g1[1])
    end

    return M1(p) - b1
end

"""
    glc_void(x, grad; sim, control) -> Float64

Void linewidth constraint for NLopt.
"""
function glc_void(x::Vector{Float64}, grad::Vector{Float64}; sim, control)
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    Lx = sim.grid.x[end] - sim.grid.x[1]
    Ly = sim.grid.y[end] - sim.grid.y[1]
    resolution = (nx - 1) / Lx

    # Build filter/threshold functions
    R = control.R_filter[1]
    h = conic_filter(R, Lx, Ly, resolution)
    # On an axis convolvedcti tiles, the wall neighbour is the next copy's filtered value (the halo);
    # on an untiled axis DCT-I's walls are mirrors; the FFT filter is periodic.
    halo = control.use_dct ? dcti_tiled((nx, ny), size(h)) : (false, false)
    filter_f = p -> control.use_dct ? convolvedcti(p; h, halo=any(halo)) : convolvefft(p; h)
    threshold_f = p -> smoothed_projection(p, control, sim)

    # Hyperparameters
    ηd = control.η_dilation
    b1 = control.b1
    c0 = 64 * R^2

    # Reshape
    p = reshape(x, (nx, ny))

    # Objective
    M2 = p -> control_void(p; c0, ηd, filter_f, threshold_f, resolution, mirror=control.use_dct, halo)

    # Gradient via Zygote
    if length(grad) > 0
        g2 = Zygote.gradient(M2, p)
        grad[:] = flatten(g2[1])
    end

    return M2(p) - b1
end

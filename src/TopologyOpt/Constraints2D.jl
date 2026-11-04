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
    gradient_sq(f, resolution; mirror) -> Matrix

|∇f|² at the grid nodes from central differences, in physical units. With `mirror=true`
(the DCT filter, whose cell has mirror walls on every edge) the neighbour across a wall is
the reflected inner node, so the normal derivative on a wall is zero. With `mirror=false`
(the FFT filter, a periodic cell of `size(f)` nodes) it is the node on the opposite edge.
"""
function gradient_sq(f::AbstractMatrix, resolution::Float64; mirror::Bool)
    fx = mirror ? vcat(f[2:2, :], f, f[end-1:end-1, :]) : vcat(f[end:end, :], f, f[1:1, :])
    fy = mirror ? hcat(f[:, 2:2], f, f[:, end-1:end-1]) : hcat(f[:, end:end], f, f[:, 1:1])
    gx = (fx[3:end, :] .- fx[1:end-2, :]) .* (resolution / 2)
    gy = (fy[:, 3:end] .- fy[:, 1:end-2]) .* (resolution / 2)
    return gx .^ 2 .+ gy .^ 2
end

"""
    indicator_solid(x, c0, filter_f, threshold_f, resolution; mirror=true) -> Matrix

Solid indicator: the projected field where the filtered field is flat. `mirror` matches the
filter's boundary (see `gradient_sq`).
"""
function indicator_solid(x::AbstractMatrix, c0::Float64, filter_f, threshold_f, resolution::Float64;
    mirror::Bool=true)
    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)
    return design_field .* exp.(-c0 .* gradient_sq(filtered_field, resolution; mirror))
end

"""
    indicator_void(x, c0, filter_f, threshold_f, resolution; mirror=true) -> Matrix
"""
function indicator_void(x::AbstractMatrix, c0::Float64, filter_f, threshold_f, resolution::Float64;
    mirror::Bool=true)
    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)
    return (1 .- design_field) .* exp.(-c0 .* gradient_sq(filtered_field, resolution; mirror))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Control functions (objective for constraint)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    control_solid(p; c0, ηe, filter_f, threshold_f, resolution, mirror=true) -> Real
"""
function control_solid(p::AbstractMatrix; c0::Float64, ηe::Float64,
    filter_f, threshold_f, resolution::Float64, mirror::Bool=true)
    filtered_field = filter_f(p)
    I_s = flatten(indicator_solid(p, c0, filter_f, threshold_f, resolution; mirror))
    flt = flatten(filtered_field)
    minfield = min.(flt .- ηe, 0.0) .^ 2
    return sum(I_s .* minfield) / length(flt)
end

"""
    control_void(p; c0, ηd, filter_f, threshold_f, resolution, mirror=true) -> Real
"""
function control_void(p::AbstractMatrix; c0::Float64, ηd::Float64,
    filter_f, threshold_f, resolution::Float64, mirror::Bool=true)
    filtered_field = filter_f(p)
    I_v = flatten(indicator_void(p, c0, filter_f, threshold_f, resolution; mirror))
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
    filter_f = p -> control.use_dct ? convolvedcti(p; h) : convolvefft(p; h)
    threshold_f = p -> smoothed_projection(p, control, sim)

    # Hyperparameters
    ηe = control.η_erosion
    b1 = control.b1
    c0 = 64 * R^2

    # Reshape
    p = reshape(x, (nx, ny))

    # Objective
    M1 = p -> control_solid(p; c0, ηe, filter_f, threshold_f, resolution, mirror=control.use_dct)

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
    filter_f = p -> control.use_dct ? convolvedcti(p; h) : convolvefft(p; h)
    threshold_f = p -> smoothed_projection(p, control, sim)

    # Hyperparameters
    ηd = control.η_dilation
    b1 = control.b1
    c0 = 64 * R^2

    # Reshape
    p = reshape(x, (nx, ny))

    # Objective
    M2 = p -> control_void(p; c0, ηd, filter_f, threshold_f, resolution, mirror=control.use_dct)

    # Gradient via Zygote
    if length(grad) > 0
        g2 = Zygote.gradient(M2, p)
        grad[:] = flatten(g2[1])
    end

    return M2(p) - b1
end

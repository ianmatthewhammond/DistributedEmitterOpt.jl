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
    indicator_solid(x, c0, filter_f, threshold_f, resolution) -> Matrix
"""
function indicator_solid(x::AbstractMatrix, c0::Float64, filter_f, threshold_f, resolution::Float64)
    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)

    filtered_field = repeat(filtered_field, outer=(1, 3))'
    filtered_field = repeat(filtered_field, outer=(3, 1))'

    gradient_filtered_field = centeredfft(npgradient(filtered_field), (2, size(x)...))
    grad_mag = (gradient_filtered_field[1, :, :] .* resolution) .^ 2 .+
               (gradient_filtered_field[2, :, :] .* resolution) .^ 2
    if length(size(grad_mag)) != 2
        throw(ErrorException("The gradient fields must be 2 dimensional. Check input array and filter functions."))
    end
    return design_field .* exp.(-c0 .* grad_mag)
end

"""
    indicator_void(x, c0, filter_f, threshold_f, resolution) -> Matrix
"""
function indicator_void(x::AbstractMatrix, c0::Float64, filter_f, threshold_f, resolution::Float64)
    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)

    filtered_field = repeat(filtered_field, outer=(1, 3))'
    filtered_field = repeat(filtered_field, outer=(3, 1))'

    gradient_filtered_field = centeredfft(npgradient(filtered_field), (2, size(x)...))
    grad_mag = (gradient_filtered_field[1, :, :] .* resolution) .^ 2 .+
               (gradient_filtered_field[2, :, :] .* resolution) .^ 2
    if length(size(grad_mag)) != 2
        throw(ErrorException("The gradient fields must be 2 dimensional. Check input array and filter functions."))
    end
    return (1 .- design_field) .* exp.(-c0 .* grad_mag)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Control functions (objective for constraint)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    control_solid(p; c0, ηe, filter_f, threshold_f, resolution) -> Real
"""
function control_solid(p::AbstractMatrix; c0::Float64, ηe::Float64,
    filter_f, threshold_f, resolution::Float64)
    filtered_field = filter_f(p)
    I_s = flatten(indicator_solid(p, c0, filter_f, threshold_f, resolution))
    flt = flatten(filtered_field)
    minfield = min.(flt .- ηe, 0.0) .^ 2
    return sum(I_s .* minfield) / length(flt)
end

"""
    control_void(p; c0, ηd, filter_f, threshold_f, resolution) -> Real
"""
function control_void(p::AbstractMatrix; c0::Float64, ηd::Float64,
    filter_f, threshold_f, resolution::Float64)
    filtered_field = filter_f(p)
    I_v = flatten(indicator_void(p, c0, filter_f, threshold_f, resolution))
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
    M1 = p -> control_solid(p; c0, ηe, filter_f, threshold_f, resolution)

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
    M2 = p -> control_void(p; c0, ηd, filter_f, threshold_f, resolution)

    # Gradient via Zygote
    if length(grad) > 0
        g2 = Zygote.gradient(M2, p)
        grad[:] = flatten(g2[1])
    end

    return M2(p) - b1
end

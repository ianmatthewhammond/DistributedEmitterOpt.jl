"""
    Filtering2D

DCT/FFT-based convolution filtering for 2D DOF mode.
Uses conic kernel for smooth, shift-invariant filtering.
"""

import FFTW
import Zygote

# ═══════════════════════════════════════════════════════════════════════════════
# Filter kernels
# ═══════════════════════════════════════════════════════════════════════════════

"""
    conic_filter(R, Lx, Ly, nx, ny) -> Matrix

Build conic filter kernel with radius R, normalized to sum to 1.
Grid has nx×ny points covering domain Lx×Ly.
"""
function conic_filter(R::Float64, Lx::Float64, Ly::Float64, nx::Int, ny::Int)
    # Coordinate arrays (centered at origin)
    x = range(-Lx / 2, Lx / 2, length=nx)
    y = range(-Ly / 2, Ly / 2, length=ny)

    # Build kernel
    h = zeros(Float64, nx, ny)
    for i in 1:nx, j in 1:ny
        r = sqrt(x[i]^2 + y[j]^2)
        h[i, j] = r <= R ? (1.0 - r / R) : 0.0
    end

    # Normalize
    h ./= sum(h)

    return h
end

"""
    conic_filter(radius, Lx, Ly, resolution) -> Matrix

Legacy 2D kernel signature (used by Constraints2D).
"""
function conic_filter(radius::Float64, Lx::Float64, Ly::Float64, resolution::Float64)
    return conic_filter_dct(radius, Lx, Ly, resolution)
end

"""
    conic_filter_dct(radius, Lx, Ly, resolution) -> Matrix

Legacy DCT-compatible conic kernel (matches old code behavior).
This version is intentionally unnormalized.
"""
function conic_filter_dct(radius::Float64, Lx::Float64, Ly::Float64, resolution::Float64)
    xv = 0:1 / resolution:ceil(2 * radius / Lx) * Lx / 2
    yv = 0:1 / resolution:ceil(2 * radius / Ly) * Ly / 2

    X = repeat(xv, 1, length(yv))
    Y = repeat(yv', length(xv), 1)
    mask = X .^ 2 + Y .^ 2 .< radius ^ 2
    iftruemask = (1 .- sqrt.(abs.(X .^ 2 .+ Y .^ 2)) ./ radius)
    iffalsemask = zeros(size(X))
    h = ifelse.(mask, iftruemask, iffalsemask)

    return h
end

# ═══════════════════════════════════════════════════════════════════════════════
# DCT-I convolution (efficient for symmetric grids)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DCT_I(x) -> Matrix

Type-I Discrete Cosine Transform.
"""
DCT_I(x) = FFTW.r2r(x, FFTW.REDFT00)

# Custom adjoint to avoid Zygote tracing FFTW plan construction.
Zygote.@adjoint function DCT_I(x::Matrix{Float64})
    y = FFTW.r2r(x, FFTW.REDFT00)
    function back(ȳ)
        return (DCT_I_adj(ȳ),)
    end
    return y, back
end

"""
    DCT_I_adj(x) -> Matrix

Adjoint of DCT-I under the standard Euclidean inner product.
This matches FFTW's unnormalized REDFT00 conventions.
"""
function DCT_I_adj(x::Matrix{Float64})
    y = copy(x)
    # Apply S^{-1}: scale endpoints by 2 (corners get 4)
    y[1, :] .*= 2.0
    y[end, :] .*= 2.0
    y[:, 1] .*= 2.0
    y[:, end] .*= 2.0
    y = DCT_I(y)
    # Apply S: scale endpoints by 0.5 (corners get 0.25)
    y[1, :] .*= 0.5
    y[end, :] .*= 0.5
    y[:, 1] .*= 0.5
    y[:, end] .*= 0.5
    return y
end

"""
    iDCT_I(x) -> Matrix

Inverse Type-I DCT (self-adjoint up to normalization).
"""
function iDCT_I(x)
    N = (2 * (size(x, 1) - 1)) * (2 * (size(x, 2) - 1))
    DCT_I(x) / N
end

"""
    centeredfft(arr, newshape) -> Matrix

Extract the centered subarray with shape `newshape` from `arr`.
"""
function centeredfft(arr::Matrix{Float64}, newshape::Tuple{Int,Int})
    currshape = size(arr)
    startind = floor.((currshape .- newshape) ./ 2)
    endind = startind .+ newshape
    myslice = (Int(startind[1]) + 1:Int(endind[1]),
               Int(startind[2]) + 1:Int(endind[2]))
    return arr[myslice...]
end

"""
    centeredfft_adjoint(arr, fullshape) -> Matrix

Adjoint of centeredfft: embed `arr` into a zero array of `fullshape`.
"""
function centeredfft_adjoint(arr::Matrix{Float64}, fullshape::Tuple{Int,Int})
    newshape = size(arr)
    startind = floor.((fullshape .- newshape) ./ 2)
    endind = startind .+ newshape
    myslice = (Int(startind[1]) + 1:Int(endind[1]),
               Int(startind[2]) + 1:Int(endind[2]))
    out = zeros(Float64, fullshape...)
    out[myslice...] .= arr
    return out
end

"""
    properpaddcti(arr, pad_to) -> Matrix

Pad array for DCT-I convolution (legacy behavior).
"""
function properpaddcti(arr::Matrix{Float64}, pad_to::Tuple{Int,Int})
    pad_size = pad_to .- (2 .* size(arr)) .+ 1

    top = zeros(pad_size[1], size(arr, 2))
    bottom = zeros(pad_size[1], size(arr, 2) - 1)
    middle = zeros(pad_to[1], pad_size[2])

    top_left = arr[1:end, 1:end]
    top_right = reverse(arr[2:end, 1:end], dims=1) .* 0.0
    bottom_left = reverse(arr[1:end, 2:end], dims=2) .* 0.0
    bottom_right = reverse(reverse(arr[2:end, 2:end], dims=2), dims=1) .* 0.0

    return hcat(
        vcat(top_left, top, top_right),
        middle,
        vcat(bottom_left, bottom, bottom_right),
    )
end

"""
    edgepaddcti(arr, pad) -> Matrix

Edge padding for DCT-I convolution (legacy behavior).
"""
function edgepaddcti(arr::Matrix{Float64}, pad)
    left = repeat(arr[1, 1:end], outer=(1, pad[1][1]))'
    right = repeat(arr[end, 1:end], outer=(1, pad[1][2]))'
    top = repeat(arr[1:end, 1], outer=(1, pad[2][1]))
    bottom = repeat(arr[1:end, end], outer=(1, pad[2][2]))

    top_left = repeat([arr[1, 1]], outer=(pad[2][1], pad[1][1]))'
    top_right = repeat([arr[end, 1]], outer=(pad[2][1], pad[1][2]))'
    bottom_left = repeat([arr[1, end]], outer=(pad[2][2], pad[1][1]))'
    bottom_right = repeat([arr[end, end]], outer=(pad[2][2], pad[1][2]))'

    return hcat(
        vcat(top_left, top, top_right),
        vcat(left, arr, right),
        vcat(bottom_left, bottom, bottom_right),
    )
end

"""
    convolvedcti(x; h) -> Matrix

Convolve x with kernel h using DCT-I (efficient for symmetric boundaries).
"""
function convolvedcti(x::Matrix{Float64}; h::Matrix{Float64})
    sx, sy = size(x)
    kx, ky = size(h)

    npx = Int(ceil((2 * kx - 1) / sx))
    npy = Int(ceil((2 * ky - 1) / sy))
    if npx % 2 == 0
        npx += 1
    end
    if npy % 2 == 0
        npy += 1
    end

    x = repeat(x, outer=(npx, npy))
    x = edgepaddcti(x, ((0, 0), (0, 0)))
    h = properpaddcti(h, (npx * sx, npy * sy))

    freq_filter = DCT_I(h)
    freq_filter = freq_filter ./ maximum(freq_filter)
    xout = centeredfft(iDCT_I(DCT_I(x) .* freq_filter), (sx, sy))
    return xout
end

"""
    convolvedcti_adjoint(x; h) -> Matrix

Adjoint of DCT-I convolution with kernel h.
"""
function convolvedcti_adjoint(x::Matrix{Float64}; h::Matrix{Float64})
    sx, sy = size(x)
    kx, ky = size(h)

    npx = Int(ceil((2 * kx - 1) / sx))
    npy = Int(ceil((2 * ky - 1) / sy))
    if npx % 2 == 0
        npx += 1
    end
    if npy % 2 == 0
        npy += 1
    end

    fullshape = (npx * sx, npy * sy)
    h = properpaddcti(h, fullshape)
    freq_filter = DCT_I(h)
    freq_filter = freq_filter ./ maximum(freq_filter)

    y = centeredfft_adjoint(x, fullshape)
    y = DCT_I_adj(y)
    y .= y .* freq_filter
    y = DCT_I_adj(y)
    N = (2 * (fullshape[1] - 1)) * (2 * (fullshape[2] - 1))
    y ./= N

    # Adjoint of repeat(x, outer=(npx,npy)) is block sum
    out = zeros(Float64, sx, sy)
    for i in 0:npx-1, j in 0:npy-1
        out .+= y[i * sx + 1:(i + 1) * sx, j * sy + 1:(j + 1) * sy]
    end

    return out
end

# ═══════════════════════════════════════════════════════════════════════════════
# FFT convolution (general case)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    convolvefft(x; h) -> Matrix

Convolve x with kernel h using FFT (general boundaries).
"""
function convolvefft(x::Matrix{Float64}; h::Matrix{Float64})
    # Legacy repeat + symmetric pad path (matches old codebase behavior)
    sx, sy = size(x)
    kx, ky = size(h)

    npx = Int(ceil((2 * kx - 1) / sx))
    npy = Int(ceil((2 * ky - 1) / sy))
    if npx % 2 == 0
        npx += 1
    end
    if npy % 2 == 0
        npy += 1
    end

    x_rep = repeat(x, outer=(npx, npy))
    x_rep = edgepad(x_rep, ((0, 0), (0, 0)))
    h_pad = properpad(h, (npx * sx, npy * sy))
    h_pad ./= sum(h_pad)

    xout = centeredfft(real.(FFTW.ifft(FFTW.fft(x_rep) .* FFTW.fft(h_pad))), (sx, sy))
    return xout
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    filter_grid(p_vec, sim, control) -> Vector

Filter design vector using 2D convolution.

## Arguments
- `p_vec` — Flat design vector
- `sim` — Simulation (for grid dimensions)
- `control` — Control (for filter params)
"""
function filter_grid(p_vec::Vector{Float64}, sim, control)
    if !control.use_filter
        return copy(p_vec)
    end

    # Get grid dimensions
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    Lx = sim.grid.x[end] - sim.grid.x[1]
    Ly = sim.grid.y[end] - sim.grid.y[1]

    # Build kernel
    R = control.R_filter[1]  # Use x-radius for 2D
    h = if control.use_dct
        resolution = (nx - 1) / Lx
        conic_filter_dct(R, Lx, Ly, resolution)
    else
        resolution = (nx - 1) / Lx
        conic_filter(R, Lx, Ly, resolution)
    end

    # Reshape to 2D
    x = reshape(p_vec, (nx, ny))

    # Convolve
    if control.use_dct
        y = convolvedcti(x; h)
    else
        y = convolvefft(x; h)
    end

    # Flatten back
    return vec(y)
end

"""
    filter_grid_adjoint(∂g_∂pf, sim, control) -> Vector

Adjoint of filter operation (same as forward for symmetric kernel).
"""
function filter_grid_adjoint(∂g_∂pf::Vector{Float64}, sim, control)
    if !control.use_filter
        return copy(∂g_∂pf)
    end

    nx, ny = length(sim.grid.x), length(sim.grid.y)
    Lx = sim.grid.x[end] - sim.grid.x[1]
    Ly = sim.grid.y[end] - sim.grid.y[1]
    R = control.R_filter[1]
    h = if control.use_dct
        resolution = (nx - 1) / Lx
        conic_filter_dct(R, Lx, Ly, resolution)
    else
        resolution = (nx - 1) / Lx
        conic_filter(R, Lx, Ly, resolution)
    end

    x = reshape(∂g_∂pf, (nx, ny))
    y = control.use_dct ? convolvedcti_adjoint(x; h) : convolvefft(x; h)
    return vec(y)
end

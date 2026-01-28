"""
    Filtering2D

DCT/FFT-based convolution filtering for 2D DOF mode.
Uses conic kernel for smooth, shift-invariant filtering.
"""

import FFTW

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

# ═══════════════════════════════════════════════════════════════════════════════
# DCT-I convolution (efficient for symmetric grids)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DCT_I(x) -> Matrix

Type-I Discrete Cosine Transform.
"""
DCT_I(x) = FFTW.r2r(x, FFTW.REDFT00)

"""
    iDCT_I(x) -> Matrix

Inverse Type-I DCT (self-adjoint up to normalization).
"""
function iDCT_I(x)
    N = (2 * (size(x, 1) - 1)) * (2 * (size(x, 2) - 1))
    DCT_I(x) / N
end

"""
    convolvedcti(x; h) -> Matrix

Convolve x with kernel h using DCT-I (efficient for symmetric boundaries).
"""
function convolvedcti(x::Matrix{Float64}; h::Matrix{Float64})
    # Transform kernel and signal
    H = DCT_I(h)
    X = DCT_I(x)

    # Pointwise multiply in frequency domain
    Y = H .* X

    # Inverse transform
    y = iDCT_I(Y)

    return y
end

# ═══════════════════════════════════════════════════════════════════════════════
# FFT convolution (general case)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    convolvefft(x; h) -> Matrix

Convolve x with kernel h using FFT (general boundaries).
"""
function convolvefft(x::Matrix{Float64}; h::Matrix{Float64})
    # Pad to avoid wraparound
    nx, ny = size(x)
    nxh, nyh = size(h)

    # FFT size
    nxpad = nx + nxh - 1
    nypad = ny + nyh - 1

    # Pad arrays
    x_pad = zeros(Float64, nxpad, nypad)
    h_pad = zeros(Float64, nxpad, nypad)
    x_pad[1:nx, 1:ny] = x
    h_pad[1:nxh, 1:nyh] = h

    # Center kernel
    h_pad = circshift(h_pad, (-nxh ÷ 2, -nyh ÷ 2))

    # FFT convolve
    X = FFTW.fft(x_pad)
    H = FFTW.fft(h_pad)
    Y = X .* H
    y_pad = real.(FFTW.ifft(Y))

    # Extract valid region
    y = y_pad[1:nx, 1:ny]

    return y
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

    # Build kernel with matching grid size
    R = control.R_filter[1]  # Use x-radius for 2D
    h = conic_filter(R, Lx, Ly, nx, ny)

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
    # For symmetric kernels, adjoint = forward
    return filter_grid(∂g_∂pf, sim, control)
end

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

Inverse Type-I DCT: the forward transform divided by N = 2(nx − 1) · 2(ny − 1). DCT-I is its own inverse up to that
factor. It is not its own transpose: the endpoint weights make its matrix non-symmetric in the Euclidean inner
product, so adjoints use `DCT_I_adj`, never `DCT_I` or `iDCT_I`.
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
    fold_axis_dcti(h, n, dim) -> Matrix

A one-sided DCT-I kernel (`h` at offsets 0, 1, 2, … along `dim`, the other half implied by symmetry) folded onto a
cell of `n` nodes along `dim`. A DCT-I of the cell is the cell mirrored for ever, with period N = 2(n − 1), so the
kernel that acts on the cell is the sum of the kernel's images in all mirrored copies: offset d collects `h` at
d + kN (k ≥ 0) and at kN − d (k ≥ 1). On a wall node (d = 0 or d = n − 1) both lists name the same samples, so a
sample that lands there from an offset beyond the cell counts twice. A kernel shorter than the cell is returned as is.

The cell needs at least two nodes along `dim` (a DCT-I of one node is undefined); fewer is an `ArgumentError`.
"""
function fold_axis_dcti(h::Matrix{Float64}, n::Int, dim::Int)
    n >= 2 || throw(ArgumentError("fold_axis_dcti: a DCT-I cell needs at least 2 nodes along every axis, got $n along axis $dim"))
    k = size(h, dim)
    k <= n - 1 && return h
    N = 2 * (n - 1)
    out = zeros(Float64, dim == 1 ? (n, size(h, 2)) : (size(h, 1), n))
    for i in 0:k-1
        r = mod(i, N)
        j = r <= n - 1 ? r : N - r                       # the image of offset i inside the cell
        w = (i > 0 && (r == 0 || r == n - 1)) ? 2.0 : 1.0
        if dim == 1
            @views out[j+1, :] .+= w .* h[i+1, :]
        else
            @views out[:, j+1] .+= w .* h[:, i+1]
        end
    end
    return out
end

"""
    dcti_frequency_filter(h, cell) -> Matrix

The DCT-I frequency response of the one-sided kernel `h` on a cell of `cell` nodes, normalized to 1 at its maximum
(the mean). A kernel longer than the cell along an axis is folded onto it (`fold_axis_dcti`), so the filter always
sees the cell continued by its mirror images, whatever the kernel's size. For a kernel that fits (2k − 1 ≤ n on both
axes) the array transformed is the one `properpaddcti` builds, value for value.

The cell needs at least two nodes along each axis. The response is rebuilt on every call, as it was before the
kernel was folded; it depends only on the kernel and the cell size.
"""
function dcti_frequency_filter(h::Matrix{Float64}, cell::Tuple{Int,Int})
    all(>=(2), cell) || throw(ArgumentError("dcti_frequency_filter: a DCT-I cell needs at least 2 nodes along every axis, got $cell"))
    hf = fold_axis_dcti(fold_axis_dcti(h, cell[1], 1), cell[2], 2)
    hp = zeros(Float64, cell)
    hp[1:size(hf, 1), 1:size(hf, 2)] .= hf
    freq_filter = DCT_I(hp)
    return freq_filter ./ maximum(freq_filter)
end

"""
    dcti_tiles(cell, kernel) -> (npx, npy)

Always (1, 1): `convolvedcti` filters the cell alone, with DCT-I's mirror walls on every axis. A kernel longer
than the cell is folded onto it (`fold_axis_dcti`) instead of tiling the cell. Before that fix a long kernel made
`convolvedcti` tile the cell by plain repetition, which put a shifted copy beside the cell where its mirror image
belongs. Kept so that callers written for the tiled filter keep working.
"""
dcti_tiles(cell::Tuple{Int,Int}, kernel::Tuple{Int,Int}) = (1, 1)

"""
    dcti_tiled(cell, kernel) -> (Bool, Bool)

Always (false, false): see `dcti_tiles`.
"""
dcti_tiled(cell::Tuple{Int,Int}, kernel::Tuple{Int,Int}) = dcti_tiles(cell, kernel) .> 1

"""
    legacy_dcti_tiles(cell, kernel) -> (npx, npy)

How many copies of the cell `convolvedcti` tiled by repetition along x and y before a long kernel was folded onto
the cell (filter revision 1). (1, 1) means the kernel fits the cell (2k − 1 ≤ n on both axes) and revisions 1 and 2
filter identically; anything larger means revision 1 filtered a different, wrongly continued field. Used only to
recognize optimizer state written under revision 1 (`dcti_fold_changes_filter`).
"""
function legacy_dcti_tiles(cell::Tuple{Int,Int}, kernel::Tuple{Int,Int})
    np = Int.(ceil.((2 .* kernel .- 1) ./ cell))
    return map(n -> iseven(n) ? n + 1 : n, np)
end

"""
    dcti_fold_changes_filter(sim, control) -> Bool

Whether this foundry problem's 2D DCT-I filter (the design filter of `filter_grid` and the filter inside the
length-scale constraints, both built from `control.R_filter[1]` on `sim.grid`) is one that filter revision 1 tiled
by repetition, i.e. whether objective and constraint values computed before the kernel was folded belong to a
different problem. False for the FFT filter and for every kernel that fits the cell.

The test is conservative: it reproduces the old allocation rule (tiling whenever 2k − 1 > n along an axis, k counting
the kernel array's zero padding), not an operator comparison. A kernel array that the old filter tiled although its
nonzero part acts as the identity (a 2 × 2 cell with a radius below one pitch) is reported as changed. It does not ask
whether the filter is called either; `_filter_revision_matters` adds that.
"""
function dcti_fold_changes_filter(sim, control)
    control.use_dct || return false
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    Lx = sim.grid.x[end] - sim.grid.x[1]
    Ly = sim.grid.y[end] - sim.grid.y[1]
    resolution = (nx - 1) / Lx
    h = conic_filter_dct(control.R_filter[1], Lx, Ly, resolution)
    return any(legacy_dcti_tiles((nx, ny), size(h)) .> 1)
end

"""
    convolvedcti(x; h, halo=false) -> Matrix

Convolve x with kernel h using DCT-I: the cell is continued beyond every wall by its mirror image (the reflected
inner nodes; the wall node is not repeated), for a kernel of any size. `x` needs at least two nodes along each axis.

`halo` is accepted for callers written for the tiled filter. No axis is tiled any more, so the neighbour of a wall
node is the mirror image of the inner node and no halo is added: the result has the size of `x`.
"""
function convolvedcti(x::Matrix{Float64}; h::Matrix{Float64}, halo::Bool=false)
    # The kernel is a constant of the filter; its folding mutates an array, which Zygote must not trace.
    freq_filter = Zygote.ignore() do
        dcti_frequency_filter(h, size(x))
    end
    return iDCT_I(DCT_I(x) .* freq_filter)
end

"""
    convolvedcti_adjoint(x; h) -> Matrix

Adjoint (transpose in the Euclidean inner product) of `convolvedcti` with kernel h. It is not the forward map:
DCT-I's endpoint weights make the filter matrix non-symmetric, so the transpose is built from `DCT_I_adj` with the
same folded frequency response.
"""
function convolvedcti_adjoint(x::Matrix{Float64}; h::Matrix{Float64})
    sx, sy = size(x)
    freq_filter = dcti_frequency_filter(h, (sx, sy))

    y = DCT_I_adj(copy(x))
    y .= y .* freq_filter
    y = DCT_I_adj(y)
    N = (2 * (sx - 1)) * (2 * (sy - 1))
    y ./= N

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

Adjoint (transpose) of `filter_grid`. For the DCT-I filter this is `convolvedcti_adjoint`, which differs from the
forward map at the walls (endpoint weights); only the periodic FFT filter with a symmetric kernel is its own transpose.
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

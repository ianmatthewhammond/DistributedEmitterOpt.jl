# # Metal 2D Foundry: Evaluate Objective from an Image
#md ```@meta
#md EditURL = "https://github.com/ianmatthewhammond/DistributedEmitterOpt.jl/tree/main/docs/src/examples/metal_2d_image_eval.jl"
#md ```
#
# This example evaluates the SERS objective for a fixed metal design specified
# by a grayscale image (metal = 1, void = 0). No optimization is performed.

using DistributedEmitterOpt
using LinearAlgebra

# Optional (if you want to load an image):
# using FileIO, ImageIO, ImageTransformations, ColorTypes

# ## 1. Mesh + simulation (foundry mode)
λ = 532.0
geo = SymmetricGeometry(λ; L=200.0, W=200.0, hd=120.0, hsub=60.0)
geo.l1 = 40.0
geo.l2 = 20.0
geo.l3 = 40.0

outdir = mktempdir()
meshfile = joinpath(outdir, "mesh.msh")
genmesh(geo, meshfile; per_x=true, per_y=true)

sim = build_simulation(meshfile; foundry_mode=true, dir_x=false, dir_y=false)

# ## 2. Physics + objective (elastic, isotropic)
env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)
inputs = [FieldConfig(λ; θ=0.0, pol=:y)]

# Empty outputs => elastic scattering
pde = MaxwellProblem(env=env, inputs=inputs, outputs=FieldConfig[])

objective = SERSObjective(
    αₚ=Matrix{ComplexF64}(I, 3, 3),
    volume=true,
    surface=false,
    use_damage_model=false
)

control = Control(
    use_filter=true,
    R_filter=(20.0, 20.0, 20.0),
    use_dct=true,  # DCT filtering for 2D foundry mode
    use_projection=true,
    β=8.0,
    η=0.5,
    use_ssp=true
)

prob = OptimizationProblem(pde, objective, sim, UmfpackSolver();
    foundry_mode=true,
    control=control,
    root=outdir
)

# ## 3. Load design from image (grayscale)
# The image should be normalized so 1.0 = metal, 0.0 = void.
#
# img = load("design.png")
# img_gray = channelview(colorview(Gray, img))
# img_resized = imresize(img_gray, (length(sim.grid.x), length(sim.grid.y)))
# p_img = clamp.(Float64.(img_resized), 0.0, 1.0)
#
# If your image uses white=void, invert it:
# p_img = 1 .- p_img
#
# If the pattern looks transposed or flipped, try:
# p_img = reverse(permutedims(p_img), dims=2)

# Placeholder: simple radial pattern (replace with image data)
nx, ny = length(sim.grid.x), length(sim.grid.y)
x = range(-1.0, 1.0, length=nx)
y = range(-1.0, 1.0, length=ny)
p_img = [sqrt(xi^2 + yj^2) < 0.4 ? 1.0 : 0.0 for xi in x, yj in y]

# ## 4. Evaluate objective (no optimization)
prob.p .= vec(p_img)

# Evaluate objective + gradient
# (If you only need g, pass an empty gradient vector to objective_and_gradient!)
g, ∇g = evaluate(prob, prob.p)

println("Objective value = ", g)

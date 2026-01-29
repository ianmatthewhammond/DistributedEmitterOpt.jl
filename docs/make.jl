using Documenter
using DistributedEmitterOpt
using Literate

# Generate examples
example_dst = joinpath(@__DIR__, "src", "examples")
example_sources = [
    "metal_2d_image_eval.jl",
    "anisotropic_3d_inelastic_optimization.jl",
    "dielectric_2d_elastic_optimization.jl",
]

for example_src in example_sources
    Literate.markdown(
        joinpath(example_dst, example_src),
        example_dst;
        execute=false,
        documenter=true,
        codefence="```julia" => "```"
    )
end

makedocs(
    sitename="DistributedEmitterOpt.jl",
    authors="Ian Hammond",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://ianmatthewhammond.github.io/DistributedEmitterOpt.jl",
        assets=String[],
    ),
    modules=[DistributedEmitterOpt],
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Architecture" => "architecture.md",
        "Examples" => [
            "Metal 2D Image Evaluation" => "examples/metal_2d_image_eval.md",
            "Anisotropic 3D Inelastic Optimization" => "examples/anisotropic_3d_inelastic_optimization.md",
            "Dielectric 2D Elastic Optimization" => "examples/dielectric_2d_elastic_optimization.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Physics" => "api/physics.md",
            "Objectives" => "api/objectives.md",
            "TopologyOpt" => "api/topologyopt.md",
            "Optimization" => "api/optimization.md",
        ],
    ],
    doctest=false,
    warnonly=true,
)

deploydocs(
    repo="github.com/ianmatthewhammond/DistributedEmitterOpt.jl.git",
    devbranch="main",
    push_preview=true,
)

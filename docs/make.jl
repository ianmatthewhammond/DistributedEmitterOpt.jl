using Documenter
using DistributedEmitterOpt
using Literate

# Generate examples
example_src = joinpath(@__DIR__, "src", "examples", "isotropic_3d.jl")
example_dst = joinpath(@__DIR__, "src", "examples")
Literate.markdown(example_src, example_dst; execute=false, documenter=true, codefence="```julia" => "```")

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
            "Isotropic 3D Optimization" => "examples/isotropic_3d.md",
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

using Documenter
using DistributedEmitterOpt

makedocs(
    sitename="DistributedEmitterOpt.jl",
    authors="Ian Hammond",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://hammy4815.github.io/DistributedEmitterOpt.jl",
        assets=String[],
    ),
    modules=[DistributedEmitterOpt],
    repo="https://github.com/hammy4815/DistributedEmitterOpt.jl",
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Architecture" => "architecture.md",
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
    repo="github.com/hammy4815/DistributedEmitterOpt.jl.git",
    devbranch="main",
    push_preview=true,
)

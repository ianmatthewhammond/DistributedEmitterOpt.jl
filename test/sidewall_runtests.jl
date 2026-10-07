using Test

@testset "Sidewall optimization" begin
    for file in ("sidewall_tests.jl", "sidewall_tvvoid_tests.jl")
        # Separate namespaces keep test fixtures and constants independent.
        @testset "$file" begin
            sandbox = Module(gensym(:SidewallTests))
            Core.eval(sandbox, :(include(path) = Base.include($sandbox, path)))
            Base.include(sandbox, joinpath(@__DIR__, file))
        end
    end
end

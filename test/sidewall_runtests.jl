using Test
include("standalone_library.jl")

# These need the optional NLopt-MMA-CCSA library (see standalone_library.jl).
const NEEDS_STANDALONE_LIBRARY = ("mmaccsa_state_header_tests.jl", "epigraph_tests.jl",
                                  "epigraph_resume_tests.jl")

@testset "Sidewall optimization" begin
    for file in ("sidewall_tests.jl", "robust_sidewall_tests.jl", "sidewall_tvvoid_tests.jl",
                 "robust_checkpoint_tests.jl", "epigraph_checkpoint_tests.jl",
                 "mmaccsa_state_header_tests.jl", "epigraph_tests.jl",
                 "epigraph_resume_tests.jl", "sidewall_legacy_refusal_tests.jl")
        if file in NEEDS_STANDALONE_LIBRARY && !STANDALONE_LIBRARY_READY
            @testset "$file" begin
                @test_skip "needs NLopt-MMA-CCSA"
            end
            continue
        end
        # Separate namespaces keep test fixtures and constants independent.
        @testset "$file" begin
            sandbox = Module(gensym(:SidewallTests))
            Core.eval(sandbox, :(include(path) = Base.include($sandbox, path)))
            Base.include(sandbox, joinpath(@__DIR__, file))
        end
    end
end

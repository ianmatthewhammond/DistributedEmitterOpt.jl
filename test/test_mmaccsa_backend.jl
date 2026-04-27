using Test

# Standalone backend smoke test — does NOT require any DEO machinery, just
# the standalone NLopt-MMA-CCSA shared library. Mirrors the structure of
# tests in the NLopt-MMA-CCSA repo to verify the wrapper inside DEO is
# wired correctly: cfunction trampolines, struct layouts, sign convention,
# checkpoint save/load.

include(joinpath(@__DIR__, "..", "src", "Optimization", "MmaccsaBackend.jl"))
using .MmaccsaBackend

@testset "MmaccsaBackend" begin
    @test library_available()

    # Quadratic centered at (0.25, -0.50). NLopt-MMA-CCSA minimizes; we hand
    # it the raw quadratic (no sign flip needed at this layer). The DEO
    # integration in Optimizer.jl is what flips for max.
    function quad(x, grad)
        if !isempty(grad)
            grad[1] = 2 * (x[1] - 0.25)
            grad[2] = 2 * (x[2] + 0.5)
        end
        return (x[1] - 0.25)^2 + (x[2] + 0.5)^2
    end

    @testset "unconstrained MMA" begin
        p = [-1.2, 1.0]
        ret, fmin, p_opt = minimize!(:mma, quad, p;
            lb=[-2.0, -2.0], ub=[2.0, 2.0], maxeval=80)
        @test ret > 0
        @test fmin < 1e-8
        @test p_opt ≈ [0.25, -0.5] atol=1e-3
    end

    @testset "unconstrained CCSAQ" begin
        p = [-1.2, 1.0]
        ret, fmin, p_opt = minimize!(:ccsaq, quad, p;
            lb=[-2.0, -2.0], ub=[2.0, 2.0], maxeval=80)
        @test ret > 0
        @test fmin < 1e-8
    end

    @testset "constraint x[1] + x[2] - 0.25 ≤ 0" begin
        function lin(x, grad)
            if !isempty(grad)
                grad[1] = 1.0
                grad[2] = 1.0
            end
            return x[1] + x[2] - 0.25
        end
        p = [0.8, 0.8]
        ret, fmin, p_opt = minimize!(:mma, quad, p;
            lb=[-2.0, -2.0], ub=[2.0, 2.0], maxeval=60,
            constraints=[(lin, 0.0)])
        @test ret > 0
        @test isfinite(fmin)
        # Constraint must be (approximately) satisfied at the optimum.
        @test (p_opt[1] + p_opt[2] - 0.25) ≤ 1e-6
    end

    @testset "checkpoint save/load roundtrip" begin
        # Run uninterrupted as the reference.
        full_state = MmaState()
        p_ref = [-1.2, 1.0]
        _, fref, p_ref_opt = minimize!(:mma, quad, p_ref;
            lb=[-2.0, -2.0], ub=[2.0, 2.0], maxeval=40, state=full_state)
        free_state!(full_state)

        # Run-pause-resume across a serialized checkpoint.
        split_state = MmaState()
        p1 = [-1.2, 1.0]
        _, _, p_after_first = minimize!(:mma, quad, p1;
            lb=[-2.0, -2.0], ub=[2.0, 2.0], maxeval=12, state=split_state)

        path = tempname() * ".mma_state"
        save_state(path, split_state)
        free_state!(split_state)

        loaded = load_state(path)
        try
            @test loaded.ref[] != C_NULL
            _, fdisk, p_after_disk = minimize!(:mma, quad, p_after_first;
                lb=[-2.0, -2.0], ub=[2.0, 2.0], maxeval=28, state=loaded)
            @test fref ≈ fdisk atol=1e-10 rtol=1e-10
            @test p_ref_opt ≈ p_after_disk atol=1e-10 rtol=1e-10
        finally
            free_state!(loaded)
            rm(path; force=true)
        end
    end
end

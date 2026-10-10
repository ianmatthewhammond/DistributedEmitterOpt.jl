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
        path = tempname() * ".mma_state"
        committed_p = Ref(copy(p1))
        committed_evals = Ref(0)
        checkpoint = (x, fmin, nevals, k) -> begin
            save_state(path, split_state)
            committed_p[] = copy(x)
            committed_evals[] = nevals
        end
        minimize!(:mma, quad, p1;
            lb=[-2.0, -2.0], ub=[2.0, 2.0], maxeval=12, state=split_state,
            checkpoint_callback=checkpoint)
        free_state!(split_state)

        loaded = load_state(path)
        try
            @test loaded.ref[] != C_NULL
            _, fdisk, p_after_disk = minimize!(:mma, quad, committed_p[];
                lb=[-2.0, -2.0], ub=[2.0, 2.0], maxeval=40-committed_evals[], state=loaded)
            @test fref ≈ fdisk atol=1e-10 rtol=1e-10
            @test p_ref_opt ≈ p_after_disk atol=1e-10 rtol=1e-10
        finally
            free_state!(loaded)
        end
    end

    # Epigraph with an unbounded epigraph variable: maximize t subject to t <= g_s(p) for
    # three nearly parallel scenarios. t rises every outer iteration and its sigma grows
    # without a cap (ub = Inf), which is the regime where the L-BFGS dual's line search
    # fails on an already-converged dual.
    epi_np = 24
    W = [0.5 + 0.5 * sin(1.3 * j + 0.7 * s) for s in 1:3, j in 1:epi_np]
    Q = [0.3 + 0.2 * cos(0.9 * j - 0.4 * s) for s in 1:3, j in 1:epi_np]
    S = [100.0, 100.0 * (1 + 3e-3), 100.0 * (1 - 2e-3)]
    g_s(s, p) = S[s] * sum(W[s, j] * p[j] - Q[s, j] * p[j]^2 + 0.05 * sin(3.0 * p[j] + s) for j in 1:epi_np) / epi_np
    function epi_con(s)
        return (x, grad) -> begin
            p = view(x, 1:epi_np)
            if !isempty(grad)
                for j in 1:epi_np
                    grad[j] = -S[s] * (W[s, j] - 2 * Q[s, j] * p[j] + 0.15 * cos(3.0 * p[j] + s)) / epi_np
                end
                grad[end] = 1.0
            end
            x[end] - g_s(s, p)
        end
    end
    epi_obj(x, grad) = (isempty(grad) || (fill!(grad, 0.0); grad[end] = -1.0); -x[end])
    p0 = fill(0.2, epi_np)
    x0 = [p0; 0.9 * minimum(g_s(s, p0) for s in 1:3)]
    run_epi(alg; kw...) = minimize!(alg, epi_obj, copy(x0);
        lb=[zeros(epi_np); 0.0], ub=[ones(epi_np); Inf], maxeval=150,
        constraints=[(epi_con(s), 0.0) for s in 1:3], kw...)

    @testset "dual solver: MMA by default, L-BFGS on request" begin
        @test dual_algorithm_supported(:mma)
        @test dual_algorithm_supported(:lbfgs)
        for alg in (:mma, :ccsaq)
            ret_default, f_default, _ = run_epi(alg)
            ret_mma, f_mma, _ = run_epi(alg; dual_algorithm=:mma)
            ret_lbfgs, _, _ = run_epi(alg; dual_algorithm=:lbfgs)
            @test ret_default == 5 && f_default < -25.36   # ran the whole budget
            @test ret_mma == ret_default && f_mma === f_default   # the default IS the MMA dual
            @test ret_lbfgs == -1                          # control: the dual choice matters here
        end
        @test_throws ArgumentError run_epi(:ccsaq; dual_algorithm=:bogus)
    end

    @testset "a library without the MMA dual is refused up front" begin
        # Needs a build of the library that predates the MMA dual; point
        # DEO_TEST_LEGACY_MMACCSA_LIB at one to run this.
        legacy = get(ENV, "DEO_TEST_LEGACY_MMACCSA_LIB", "")
        if isempty(legacy)
            @test_skip "DEO_TEST_LEGACY_MMACCSA_LIB not set"
        else
            src = joinpath(@__DIR__, "..", "src", "Optimization", "MmaccsaBackend.jl")
            script = """
                include($(repr(src))); using .MmaccsaBackend
                q(x, g) = (isempty(g) || (g[1] = 2x[1]); x[1]^2)
                c(x, g) = (isempty(g) || (g[1] = -1.0); 0.1 - x[1])
                println("SUPPORTED_MMA=", dual_algorithm_supported(:mma))
                println("SUPPORTED_LBFGS=", dual_algorithm_supported(:lbfgs))
                msg = try
                    minimize!(:ccsaq, q, [0.5]; lb=[-1.0], ub=[1.0], maxeval=5, constraints=[(c, 0.0)]); "none"
                catch e; sprint(showerror, e) end
                println("DEFAULT_ERROR=", msg)
                r, _, _ = minimize!(:ccsaq, q, [0.5]; lb=[-1.0], ub=[1.0], maxeval=5,
                                    constraints=[(c, 0.0)], dual_algorithm=:lbfgs)
                println("LBFGS_RET=", r)
                """
            out = read(addenv(`$(Base.julia_cmd()) --startup-file=no -e $script`,
                              "JULIA_NLOPT_MMACCSA_LIB" => legacy), String)
            @test occursin("SUPPORTED_MMA=false", out)
            @test occursin("SUPPORTED_LBFGS=true", out)
            @test occursin("does not support dual_algorithm=:mma", out)
            @test occursin(r"LBFGS_RET=[1-6]\b", out)
        end
    end
end

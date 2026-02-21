"""
Logger tests for DistributedEmitterOpt.jl

Validates:
- Core logger filtering/sinks behavior
- Environment-variable defaults for logger config
- Lightweight optimization-path logging integration
"""

using DistributedEmitterOpt
using Logging
using Test

function build_logger_problem(root::String)
    g = SymmetricGeometry(532.0; L=40.0, W=40.0, hd=20.0, hsub=20.0)
    g.l1 = 20.0
    g.l2 = 10.0
    g.l3 = 20.0
    g.hair = 40.0
    g.hs = 30.0
    g.ht = 20.0

    meshfile = joinpath(root, "logger_test_mesh.msh")
    genmesh(g, meshfile; per_x=false, per_y=false)

    cfg = RunLogConfig(
        enabled=true,
        min_level=Logging.Debug,
        sink=:file,
        path="logger_test.log",
        force_flush=true,
        every_iter=1,
        epoch=true,
        objective=true,
        memory=true,
        solver=true,
        constraints=true,
    )
    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33, logger_cfg=cfg)
    pde = MaxwellProblem(
        env=env,
        inputs=[FieldConfig(532.0; θ=0.0, pol=:y)],
        outputs=FieldConfig[],
    )
    objective = SERSObjective(volume=true, surface=false, use_damage_model=false)
    control = Control(
        use_filter=true,
        R_filter=(20.0, 20.0, 20.0),
        use_dct=false,
        use_projection=true,
        β=8.0,
        η=0.5,
        use_ssp=true,
        flag_volume=true,
    )

    prob = OptimizationProblem(pde, objective, meshfile, UmfpackSolver();
        per_x=false,
        per_y=false,
        foundry_mode=false,
        control=control,
        root=root,
        degree=4,
    )
    init_uniform!(prob, 0.5)
    rm(meshfile; force=true)
    return prob
end

@testset "Run Logger Core" begin
    mktempdir() do tmp
        logfile = joinpath(tmp, "core.log")
        cfg = RunLogConfig(
            enabled=true,
            min_level=Logging.Debug,
            sink=:file,
            path=logfile,
            every_iter=2,
            epoch=true,
            objective=true,
            memory=true,
            solver=true,
            constraints=true,
        )

        with_run_logger(cfg; root=tmp) do
            log_info(:objective, "iter1 should be filtered"; iter=1, g=1.0)
            log_info(:objective, "iter2 should be logged"; iter=2, g=2.0)
            log_debug(:memory, "memory debug line"; iter=2, free_mem_mb=123.0)
        end

        txt = read(logfile, String)
        @test !occursin("iter1 should be filtered", txt)
        @test occursin("iter2 should be logged", txt)
        @test occursin("memory debug line", txt)
    end
end

@testset "Run Logger Env Defaults" begin
    withenv(
        "DEO_LOG_ENABLE" => "1",
        "DEO_LOG_LEVEL" => "DEBUG",
        "DEO_LOG_FORCE_STDOUT" => "1",
    ) do
        cfg = RunLogConfig()
        @test cfg.enabled
        @test cfg.min_level == Logging.Debug
        @test cfg.force_flush
    end
end

@testset "Run Logger Optimization Integration" begin
    mktempdir() do tmp
        prob = build_logger_problem(tmp)
        g_opt, p_opt = optimize!(
            prob;
            max_iter=1,
            β_schedule=[8.0],
            use_constraints=false,
            tol=1e-6,
            empty_history=true,
            backup=false,
        )
        @test isfinite(g_opt)
        @test length(p_opt) == length(prob.p)

        logfile = joinpath(tmp, "logger_test.log")
        @test isfile(logfile)
        txt = read(logfile, String)
        @test occursin("Flat-substrate normalization baseline", txt)
        @test occursin("normalized objective", txt)
        @test occursin("Epoch start", txt)
    end
end

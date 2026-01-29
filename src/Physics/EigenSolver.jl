"""
    EigenSolver

Shift-invert eigensolver for the generalized eigenproblem.
Uses the solver cache for LU reuse (factorization of A - σE).
"""

"""Container for eigen solve outputs."""
struct EigenSolveResult
    vals::Vector{ComplexF64}         # Eigenvalues λ
    vecs::Vector{Vector{ComplexF64}} # Eigenvectors (block form)
    info::Any
    A::Any
    E::Any
    ε_design::Any
    ε_background::Any
    shift::Float64
end

"""
    solve_eigen!(pde, pt, sim, pool) -> EigenSolveResult

Solve the shifted eigenproblem via KrylovKit:
    (A - σE)^{-1} E x = μ x
Then λ = 1/μ + σ.
"""
function solve_eigen!(pde::EigenProblem, pt, sim, pool::SolverCachePool)
    phys = build_eigen_phys_params(pde, sim)
    A, E, εm, ε0 = assemble_eigen_matrices(pt, sim, phys)

    σ = pde.shift
    cache = pool.eigen_cache

    if (!has_eigen_factor(cache)) || (pool.eigen_shift != σ)
        eigen_lu!(cache, A - σ * E)
        pool.eigen_shift = σ
    end

    eigsolvef = x -> begin
        y = E * x
        eigen_solve!(cache, y)
    end

    vals, vecs, info = KrylovKit.eigsolve(
        eigsolvef,
        rand(ComplexF64, size(A, 1)),
        pde.num_modes;
        krylovdim=pde.krylovdim,
        which=pde.which
    )

    λs = (1 ./ vals) .+ σ

    return EigenSolveResult(λs, vecs, info, A, E, εm, ε0, σ)
end

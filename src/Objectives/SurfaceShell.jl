"""
SurfaceShell — the `:tv_void` surface measure for `SERSObjective`.

Replaces the band weight `pt(1−pt)` (a 1/β-thin volume band on the design sidewalls) with
the void-side total variation of the Helmholtz-smoothed *physical* material indicator

    χ   = pt      on Ω_design,   1 on Ω_substrate,   0 elsewhere      (what assemble_maxwell sees)
    χ_s : (1 − t²∇²) χ_s = χ   on the whole mesh, zero-flux boundaries (mirror planes)
    S   = 2 · (1 − χ) · |∇χ_s|                                          [1/nm]

so that  ∫ I·S dΩ  is an *area-weighted* sampling of I in the void within ≈ t of every
material/void interface — sidewalls, roofs at the design-region top, and the exposed
substrate floor alike. For an isolated flat face in an unbounded domain with ε_grad = 0 the
normal profile is exp(−d/t)/t on the void side, so ∫ S dn = 1 (half the band mass lies on
the void side; the factor 2 restores unit area). In practice that is approximate: the
domain is finite (the zero-flux mirror/box boundaries bend χ_s within ≈ t of them), the
shell has finite thickness t so faces closer than a few t to each other overlap, the mesh
resolves the exp(−d/t) profile only to quadrature accuracy, and ε_grad > 0 adds a floor of
2(1−χ)·ε_grad per unit volume wherever ∇χ_s ≈ 0 (bulk void, far from any face). The
surface measure is defined for finite or infinite projection sharpness. Its quantitative
area normalization must be checked for the selected mesh, boundaries and shell thickness.

Integration domain: Ω_raman ∪ Ω_design (the same two measures the volume term uses); on
Ω_raman the factor (1−χ) is 1.

Adjoint. Writing n = |∇χ_s|_ε, û = ∇χ_s / n, and g = Σ_pairs w ∫ dmg·I·S:
  ∂g/∂pt (pointwise density on Ω_design) = −2·dmg·I·n  +  λ_h,
  where λ solves the same Helmholtz system with right-hand side
  r_i = Σ_pairs w ∫_{raman∪design} 2·(1−χ)·dmg·I · (û ⋅ ∇φ_i) dΩ,
because χ_s = A⁻¹ b(pt) with b_i = ∫_design φ_i·pt (+ substrate constant) and A = Aᵀ.
The field adjoint sources are the volume-term sources with (1−pt) → S.
"""

using SparseArrays
import LinearAlgebra: cholesky, Symmetric

# ──────────────────────────────────────────────────────────────────────────────
# Helmholtz system (assembled + factorised once per (sim, t); cached on the objective)
# ──────────────────────────────────────────────────────────────────────────────
#
# Cache keys are VALUES, never object identity. The smoothed field χ_s = F⁻¹ b is a pure
# function of the right-hand side b (the material indicator in weak form), so `b_ref` — the
# vector b of the last solve — is an exact key: equal b ⇒ equal χ_s, and any change to the
# design changes b. Keying on the identity of `pt` (the previous design, `pt_ref !== pt`) was
# wrong: in the legacy foundry path every evaluation's pt is a closure that compares
# `===`-equal to the previous one, so χ_s stayed frozen at the flat-normalization p ≡ 0 shell
# while the adjoint would differentiate a moving shell. The factorisation likewise keeps
# a reference to its Simulation rather than an objectid that could be reused after GC.
mutable struct ShellCache
    sim_ref::Any           # the Simulation A was assembled on (compared with ===, kept alive here)
    t::Float64
    F::Any                 # cholesky factorisation of A = ∫ t²∇v⋅∇u + v u dΩ  (Pf × Pf)
    b_ref::Any             # right-hand side vector the cached χ_s was solved from (value key)
    χs::Any                # FEFunction on Pf
end
ShellCache() = ShellCache(nothing, NaN, nothing, nothing, nothing)

_shell_cache(obj) = (obj.shell_cache[] === nothing && (obj.shell_cache[] = ShellCache()); obj.shell_cache[])

function shell_factor!(obj, sim0)
    c = _shell_cache(obj)
    if c.F === nothing || c.sim_ref !== sim0 || c.t != obj.t_shell
        t2 = obj.t_shell^2
        A = assemble_matrix(sim0.Pf, sim0.Pf) do u, v
            ∫(t2 * (∇(v) ⋅ ∇(u)) + v * u)sim0.dΩ
        end
        c.F = cholesky(Symmetric(A))
        c.sim_ref = sim0; c.t = obj.t_shell
        c.b_ref = nothing; c.χs = nothing
    end
    return c
end

"""
    validate_objective(obj::SERSObjective, sim)

Construction-time guard (called by `OptimizationProblem`): `:tv_void` needs the mesh's
"Substrate" region so the material indicator is 1 there. Without it the indicator silently
loses the substrate and the flat-substrate baseline collapses to a small positive number that
`flat_substrate_norm` would still accept. Fail closed unless `require_substrate=false`.
"""
function validate_objective(obj::SERSObjective, sim)
    (obj.surface && obj.surface_mode == :tv_void) || return nothing
    _shell_require_substrate(obj, default_sim(sim))
    return nothing
end

function _shell_require_substrate(obj, sim0)
    if sim0.dΩ_substrate === nothing && obj.require_substrate
        error("SERSObjective surface_mode=:tv_void: the mesh has no \"Substrate\" region, so the material " *
              "indicator χ = pt·1_design + 1_substrate cannot be built. Add the Substrate physical group " *
              "(SymmetricGeometry/genmesh does) or pass require_substrate=false for a deliberately substrate-free geometry.")
    end
end

"Right-hand side  b_i = ∫_design φ_i·pt + ∫_substrate φ_i  (the material indicator in weak form)."
function shell_rhs(obj, pt, sim0)
    _shell_require_substrate(obj, sim0)
    pt_cf = _pt_cf(pt, sim0)
    b = assemble_vector(sim0.Pf) do v
        ∫(v * pt_cf)sim0.dΩ_design
    end
    if sim0.dΩ_substrate !== nothing
        b .+= assemble_vector(sim0.Pf) do v
            ∫(v * 1.0)sim0.dΩ_substrate
        end
    end
    return b
end

"""
    shell_smooth(obj, pt, sim0) → χ_s::FEFunction on sim0.Pf

The right-hand side b(pt) is assembled on EVERY call and compared by value with the one the
cached χ_s was solved from; the (cheap, back-substitution) solve is skipped only when b is
identical. This is exact — χ_s depends on the design only through b — and cannot be fooled by
aliasing or by `===`-equal closures. Within one evaluation the objective, the adjoint
sources and the explicit sensitivity all see the same pt and reuse the solve.
"""
function shell_smooth(obj, pt, sim0)
    c = shell_factor!(obj, sim0)
    b = shell_rhs(obj, pt, sim0)
    if c.χs === nothing || c.b_ref === nothing || c.b_ref != b
        χs_vec = c.F \ b
        c.χs = FEFunction(sim0.Pf, χs_vec)
        c.b_ref = b
    end
    return c.χs
end

"""
    shell_chi_s_values(obj) → Vector or nothing
Free DOF values of the currently cached χ_s (a copy); for tests and post-processing.
"""
function shell_chi_s_values(obj)
    c = obj.shell_cache[]
    (c === nothing || c.χs === nothing) && return nothing
    return copy(get_free_dof_values(c.χs))
end

_gnorm(ε) = (g -> sqrt(g ⋅ g + ε^2))

# In the legacy foundry path `pt` is a plain function of position; Gridap needs a CellField
# whenever pt is combined with a Number, so wrap it on the design triangulation.
_pt_cf(pt, sim0) = pt isa Function ? CellField(pt, sim0.Ω_design) : pt

"""
    shell_weights(obj, pt, sim0) → (n, û, S_raman, S_design)
n = |∇χ_s|_ε, û = ∇χ_s/n, S_raman = 2n, S_design = 2(1−pt)n  (all lazy CellFields).
"""
function shell_weights(obj, pt, sim0)
    χs = shell_smooth(obj, pt, sim0)
    ∇χs = ∇(χs)
    n = _gnorm(obj.ε_grad) ∘ ∇χs
    û = ∇χs / n
    pt_cf = _pt_cf(pt, sim0)
    S_raman = 2.0 * n
    S_design = 2.0 * (1.0 - pt_cf) * n
    return n, û, S_raman, S_design
end

function _shell_check_single_sim(sim, sim_out)
    sim0 = default_sim(sim)
    (sim_out === sim0 || (sim_out.model === sim0.model && sim_out.Pf === sim0.Pf)) ||
        error("SERSObjective surface_mode=:tv_void requires a shared model and scalar space; multi-mesh bundles are not supported yet")
end

# ──────────────────────────────────────────────────────────────────────────────
# Objective value contribution
# ──────────────────────────────────────────────────────────────────────────────
"g contribution of the :tv_void term for one (in,out) pair (already multiplied by dmg·I)."
function shell_objective_pair(obj, dmg, integrand, pt_out, sim, sim_out)
    _shell_check_single_sim(sim, sim_out)
    _, _, S_r, S_d = shell_weights(obj, pt_out, sim_out)
    return real(sum(∫(dmg * integrand * S_r)sim_out.dΩ_raman)) +
           real(sum(∫(dmg * integrand * S_d)sim_out.dΩ_design))
end

# ──────────────────────────────────────────────────────────────────────────────
# Explicit sensitivity: pointwise density ∂g/∂pt on Ω_design, summed over pairs
# ──────────────────────────────────────────────────────────────────────────────
"""
    shell_explicit_density(obj, pde, fields, pt, sim) → CellField on default_sim(sim)
Pointwise ∂g/∂pt for the :tv_void term, with the pair weights and the adjoint Helmholtz
solve already folded in. Multiply by a test function / SSP chain rule / grid scatter as
the caller's DOF mode requires.
"""
function shell_explicit_density(obj, pde::MaxwellProblem, fields::Dict, pt, sim)
    sim0 = default_sim(sim)
    n, û, _, _ = shell_weights(obj, pt, sim0)
    pt_cf = _pt_cf(pt, sim0)
    inputs = pde.inputs
    outputs = effective_outputs(pde)

    direct = nothing                       # Σ w · (−2·dmg·I·n)
    r = zeros(Float64, num_free_dofs(sim0.Pf))
    for fc_in in inputs
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        sim_in = sim_for(sim, fc_in)
        Ep = fields[key_in]
        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            sim_out = sim_for(sim, fc_out)
            _shell_check_single_sim(sim, sim_out)
            Ee = fields[key_out]
            weight = fc_in.weight * fc_out.weight
            Ep_out = sim_out === sim_in ? Ep : map_field(Ep, sim_out, :U)
            dmg = if obj.use_damage_model
                (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep_out)
            else
                1.0
            end
            αc1, αc2 = α_cellfields(obj.αₚ, sim_out.Ω)
            I_re = real(α̂ₚ² ∘ (Ee, Ee', Ep_out, Ep_out, αc1, αc2))

            d = (-2.0 * weight) * dmg * I_re * n
            direct = direct === nothing ? d : direct + d

            r .+= weight .* assemble_vector(sim0.Pf) do v
                ∫(2.0 * dmg * I_re * (û ⋅ ∇(v)))sim0.dΩ_raman +
                ∫(2.0 * (1.0 - pt_cf) * dmg * I_re * (û ⋅ ∇(v)))sim0.dΩ_design
            end
        end
    end
    direct === nothing && return nothing
    c = shell_factor!(obj, sim0)
    λ_h = FEFunction(sim0.Pf, c.F \ r)
    return direct + λ_h
end

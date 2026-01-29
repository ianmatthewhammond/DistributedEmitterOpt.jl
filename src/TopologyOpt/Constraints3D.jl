"""
    Constraints3D

FE-based linewidth constraints for 3D DOF mode.
Uses manual adjoint through the Helmholtz filter.
"""

import LinearAlgebra: dot

# ═══════════════════════════════════════════════════════════════════════════════
# Constraint functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    control_solid_fe(pf, pt, sim, control) -> Float64

Solid-phase constraint for 3D DOF mode.
"""
function control_solid_fe(pf, pt, sim, control)
    ηe = control.η_erosion
    c0 = 64 * control.R_filter[1]^2

    ∇pf = ∇(pf)
    exp_term = (x -> exp(-c0 * dot(x, x))) ∘ ∇pf
    I_s = pt * exp_term
    minfield = (x -> min(x, 0.0)^2) ∘ (pf - ηe)

    norm_fac = sum(∫(1.0)sim.dΩ_design)
    M = sum(∫(I_s * minfield)sim.dΩ_design) / norm_fac

    return real(M)
end

"""
    control_void_fe(pf, pt, sim, control) -> Float64

Void-phase constraint for 3D DOF mode.
"""
function control_void_fe(pf, pt, sim, control)
    ηd = control.η_dilation
    c0 = 64 * control.R_filter[1]^2

    ∇pf = ∇(pf)
    exp_term = (x -> exp(-c0 * dot(x, x))) ∘ ∇pf
    I_v = (1 - pt) * exp_term
    minfield = (x -> min(x, 0.0)^2) ∘ (ηd - pf)

    norm_fac = sum(∫(1.0)sim.dΩ_design)
    M = sum(∫(I_v * minfield)sim.dΩ_design) / norm_fac

    return real(M)
end

# ═══════════════════════════════════════════════════════════════════════════════
# NLopt constraint interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    glc_solid_fe(p_vec, grad, obj) -> Float64

Solid linewidth constraint for NLopt (3D DOF mode).
"""
function glc_solid_fe(p_vec::Vector{Float64}, grad::Vector{Float64}, obj)
    sim = obj.sim
    control = obj.control
    cache = obj.cache_pump  # Reuse cache for filter

    # Filter
    pf_vec = filter_helmholtz!(p_vec, cache, sim, control)
    pf = FEFunction(sim.Pf, pf_vec)

    # Project
    pt = project_fe(pf_vec, sim, control)

    # Compute constraint
    M = control_solid_fe(pf, pt, sim, control)

    # Gradient via manual adjoint
    if length(grad) > 0
        # ∂M/∂pf via FE assembly
        ηe = control.η_erosion
        c0 = 64 * control.R_filter[1]^2
        ∇pf = ∇(pf)
        exp_term = (x -> exp(-c0 * dot(x, x))) ∘ ∇pf
        I_s = pt * exp_term
        minfield = (x -> min(x, 0.0)^2) ∘ (pf - ηe)
        dmin = (x -> 2 * min(x, 0.0)) ∘ (pf - ηe)

        norm_fac = sum(∫(1.0)sim.dΩ_design)
        ∂M_∂pf = assemble_vector(sim.Pf) do v
            dpt = ((p, ∇p, pf, ∇pf) -> dpt_dpf(p, ∇p, pf, ∇pf; control)) ∘ (v, ∇(v), pf, ∇pf)
            ∫(
                (
                    pt * (-2 * c0 * exp_term) * (∇pf ⋅ ∇(v)) +
                    exp_term * dpt
                ) * minfield +
                v * dmin * I_s
            )sim.dΩ_design
        end ./ norm_fac

        # Chain through filter adjoint
        ∂M_∂p = filter_helmholtz_adjoint!(∂M_∂pf, cache, sim, control)

        grad[:] = ∂M_∂p
    end

    return M - control.b1
end

"""
    glc_void_fe(p_vec, grad, obj) -> Float64

Void linewidth constraint for NLopt (3D DOF mode).
"""
function glc_void_fe(p_vec::Vector{Float64}, grad::Vector{Float64}, obj)
    sim = obj.sim
    control = obj.control
    cache = obj.cache_pump

    # Filter
    pf_vec = filter_helmholtz!(p_vec, cache, sim, control)
    pf = FEFunction(sim.Pf, pf_vec)

    # Project
    pt = project_fe(pf_vec, sim, control)

    # Compute constraint
    M = control_void_fe(pf, pt, sim, control)

    # Gradient
    if length(grad) > 0
        ηd = control.η_dilation
        c0 = 64 * control.R_filter[1]^2
        ∇pf = ∇(pf)
        exp_term = (x -> exp(-c0 * dot(x, x))) ∘ ∇pf
        I_v = (1 - pt) * exp_term
        minfield = (x -> min(x, 0.0)^2) ∘ (ηd - pf)
        dmin = (x -> -2 * min(x, 0.0)) ∘ (ηd - pf)

        norm_fac = sum(∫(1.0)sim.dΩ_design)
        ∂M_∂pf = assemble_vector(sim.Pf) do v
            dpt = ((p, ∇p, pf, ∇pf) -> dpt_dpf(p, ∇p, pf, ∇pf; control)) ∘ (v, ∇(v), pf, ∇pf)
            ∫(
                (
                    (1 - pt) * (-2 * c0 * exp_term) * (∇pf ⋅ ∇(v)) -
                    exp_term * dpt
                ) * minfield +
                v * dmin * I_v
            )sim.dΩ_design
        end ./ norm_fac

        ∂M_∂p = filter_helmholtz_adjoint!(∂M_∂pf, cache, sim, control)

        grad[:] = ∂M_∂p
    end

    return M - control.b1
end

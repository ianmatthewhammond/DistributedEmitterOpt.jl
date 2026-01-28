"""
    Constraints3D

FE-based linewidth constraints for 3D DOF mode.
Uses manual adjoint through the Helmholtz filter.
"""

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

    # Gradient magnitude
    ∇pf = ∇(pf)
    ∇mag = sqrt ∘ (x -> sum(x .^ 2)) ∘ ∇pf

    # Indicator function
    indicator = x -> 1.0 / (1.0 + exp(c0 * (pt(x) - ηe)))

    # Integrate
    M = sum(∫(indicator * ∇mag)sim.dΩ_design)

    return real(M)
end

"""
    control_void_fe(pf, pt, sim, control) -> Float64

Void-phase constraint for 3D DOF mode.
"""
function control_void_fe(pf, pt, sim, control)
    ηd = control.η_dilation
    c0 = 64 * control.R_filter[1]^2

    # Gradient magnitude
    ∇pf = ∇(pf)
    ∇mag = sqrt ∘ (x -> sum(x .^ 2)) ∘ ∇pf

    # Indicator function (reversed)
    indicator = x -> 1.0 / (1.0 + exp(c0 * (ηd - pt(x))))

    # Integrate
    M = sum(∫(indicator * ∇mag)sim.dΩ_design)

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

        ∂M_∂pf = assemble_vector(sim.Pf) do v
            # Gradient magnitude sensitivity
            ∇mag = sqrt ∘ (x -> sum(x .^ 2)) ∘ ∇pf
            indicator = x -> 1.0 / (1.0 + exp(c0 * (pt(x) - ηe)))

            # Chain rule contributions
            ∫(∇(v) ⋅ (∇pf / (∇mag + 1e-10)) * indicator)sim.dΩ_design
        end

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

        ∂M_∂pf = assemble_vector(sim.Pf) do v
            ∇mag = sqrt ∘ (x -> sum(x .^ 2)) ∘ ∇pf
            indicator = x -> 1.0 / (1.0 + exp(c0 * (ηd - pt(x))))

            ∫(∇(v) ⋅ (∇pf / (∇mag + 1e-10)) * indicator)sim.dΩ_design
        end

        ∂M_∂p = filter_helmholtz_adjoint!(∂M_∂pf, cache, sim, control)

        grad[:] = ∂M_∂p
    end

    return M - control.b1
end

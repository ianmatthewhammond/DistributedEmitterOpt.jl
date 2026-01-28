"""
    MaxwellProblem

PDE definition for Maxwell curl-curl equation. Specifies the physical
environment, input/output field configurations, and solver parameters.

This type describes WHAT to solve (environment + configs), not HOW to solve
(that's in simulation + caches).
"""
Base.@kwdef struct MaxwellProblem
    # Physical environment
    env::Environment

    # Field configurations
    inputs::Vector{FieldConfig}
    outputs::Vector{FieldConfig} = FieldConfig[]  # Empty → elastic (reuse inputs)

    # Solver parameters
    α_loss::Float64 = 0.0  # Artificial absorption
end

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience constructors
# ═══════════════════════════════════════════════════════════════════════════════

"""Single-wavelength elastic scattering."""
function MaxwellProblem(λ::Float64; θ::Float64=0.0, pol::Symbol=:y,
    mat_design::MaterialSpec="Ag", kwargs...)
    env = Environment(mat_design=mat_design)
    MaxwellProblem(env=env, inputs=[FieldConfig(λ, θ=θ, pol=pol)]; kwargs...)
end

"""Pump/emission pair (inelastic)."""
function MaxwellProblem(λ_pump::Float64, λ_emission::Float64;
    θ::Float64=0.0, pol::Symbol=:y,
    mat_design::MaterialSpec="Ag", kwargs...)
    env = Environment(mat_design=mat_design)
    MaxwellProblem(
        env=env,
        inputs=[FieldConfig(λ_pump, θ=θ, pol=pol)],
        outputs=[FieldConfig(λ_emission, θ=θ, pol=pol)];
        kwargs...
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Queries
# ═══════════════════════════════════════════════════════════════════════════════

"""Effective outputs (empty → use inputs for elastic case)."""
effective_outputs(pde::MaxwellProblem) =
    isempty(pde.outputs) ? pde.inputs : pde.outputs

"""All unique field configs (for cache building)."""
function all_configs(pde::MaxwellProblem)
    configs = vcat(pde.inputs, effective_outputs(pde))
    dedup = Dict{CacheKey,FieldConfig}()
    for fc in configs
        key = cache_key(fc)
        if !haskey(dedup, key)
            dedup[key] = fc
        end
    end
    collect(values(dedup))
end

"""Is this elastic scattering? (outputs empty or same as inputs)"""
function is_elastic(pde::MaxwellProblem)
    isempty(pde.outputs) || all(
        any(configs_equal(out, inp) for inp in pde.inputs)
        for out in pde.outputs
    )
end

"""Number of unique solves needed."""
num_solves(pde::MaxwellProblem) = length(all_configs(pde))

Base.show(io::IO, pde::MaxwellProblem) =
    print(io, "MaxwellProblem($(length(pde.inputs)) inputs, $(length(pde.outputs)) outputs)")

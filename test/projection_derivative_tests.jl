"""
Projection derivative tests — the SSP flat-field branch.

Regression test for the ×4 flat-field gradient bug: the degenerate branch of
`DSP_dpf` (‖∇ρ̃‖ < 1e-8) returned `∂projection_∂pf(ρ̃, β, η)` WITHOUT the `* p`
test-function factor that the non-degenerate branch carries. On P1 tets each
nodal entry then received the full cell integral instead of its 1/4 share —
a gradient exactly 4× too large wherever the filtered field is flat (in
particular at uniform initialization, where it fires on every design cell and
locks CCSA into permanent step rejection: actual/predicted decrease = 1/4 < 1/2).

Root-caused 2026-07-20; frozen-run forensics and the
escape-race mechanism 2026-08-07.

Feasible endpoints (#34): on a flat field (‖∇ρ̃‖ = 0) the forward SSP is the
formula tanh_projection(ρ̃, η, β); at ρ̃ = 0 `smooth_project` returns 0, the value
of that formula at 0 in exact arithmetic. `DSP_dpf` used to return 0 at ρ̃ = 0
with ∇ρ̃ = 0, although the formula's one-sided derivative into ρ̃ > 0,
P′(0) = β·sech²(βη)/(tanh(βη) + tanh(β(1 − η))), is positive at every finite β.
An all-void FE projection then contributed zero to the design gradient. The endpoint tests below compare the
helpers with one-sided finite differences of the forward model where a Float64
difference resolves the slope (pointwise and through a small FE assembly), and
otherwise with high-precision reference values of the analytic slope: at the
projection strengths of the default β schedule, at strengths where e^{−2βη} alone
is below the smallest positive Float64, at the smallest positive β and at β = ∞.
A difference quotient of the implemented forward is not a reference everywhere:
at the smallest positive β with η = 0.5 the nearby Float64 forward divides 0 by 0, and with a
density of higher precision than β and η it is offset (see `flat_ssp_slope`).

Run with: julia --project test/projection_derivative_tests.jl
"""

using DistributedEmitterOpt
using Gridap
using LinearAlgebra
using Test

const DEO = DistributedEmitterOpt

@testset "DSP_dpf flat-field branch" begin
    R, β, η = 1.5, 8.0, 0.5
    ρ̃ = 0.5

    @testset "carries the test-function factor p" begin
        # In the flat-field limit the SSP derivative must reduce to the plain
        # projection derivative TIMES the test function p — the same factor the
        # non-degenerate branch applies to both of its projected terms.
        for p in (0.25, 0.5, 1.0)
            got = DEO.DSP_dpf(p, ρ̃, [0.0, 0.0, 0.0]; R, β, η)
            want = DEO.∂projection_∂pf(ρ̃, β, η) * p
            @test got ≈ want rtol = 1e-14
        end
        # Linearity in p (the bug made this constant in p):
        d1 = DEO.DSP_dpf(1.0, ρ̃, [0.0, 0.0, 0.0]; R, β, η)
        d4 = DEO.DSP_dpf(0.25, ρ̃, [0.0, 0.0, 0.0]; R, β, η)
        @test d1 ≈ 4 * d4 rtol = 1e-14
    end

    @testset "continuity across the degeneracy threshold" begin
        # The two branches must agree (to the smoothing scale) as ‖∇ρ̃‖ crosses
        # 1e-8. With the missing *p the jump is a factor 1/p — for p=0.25 a 4×
        # discontinuity this test catches immediately.
        #
        # Checked away from ρ̃ = η: there d = 0 exactly, and the non-flat branch
        # keeps a fill-factor term (15/8)·p·∂P that does not vanish as ‖∇ρ̃‖ → 0,
        # so SSP itself is discontinuous at that one point, with or without *p.
        p = 0.25
        n̂ = [1.0, 0.0, 0.0]
        for ρ̃c in (0.3, 0.45, 0.55, 0.7)
            below = DEO.DSP_dpf(p, ρ̃c, 0.99e-8 * n̂; R, β, η)
            above = DEO.DSP_dpf(p, ρ̃c, 1.01e-8 * n̂; R, β, η)
            @test below ≈ above rtol = 1e-6
        end
    end
end

@testset "SSP derivative at the feasible endpoints of a flat field" begin
    R = 2.0
    h = 1e-7
    flat = DEO.VectorValue(0.0, 0.0, 0.0)
    S(ρ̃, β, η) = DEO.smooth_project(ρ̃, flat; R, β, η)

    # (β, η) pairs with the endpoint inside the projection's transition, so that a
    # Float64 one-sided difference resolves the derivative. At β = 32, η = 0.5 the
    # derivative at 0 is about 8e-13 and the difference with h = 1e-7 rounds to 0.
    lower_cases = ((8.0, 0.5), (32.0, 0.15))
    upper_cases = ((8.0, 0.5), (32.0, 0.85))
    # controls that pass the package's validator for each η
    ctl(β, η; kw...) = Control(; R_ssp=R, β, η, η_dilation=η / 2, η_erosion=(1 + η) / 2, kw...)

    @testset "lower bound ρ̃ = 0, one-sided into ρ̃ > 0 (β = $β, η = $η)" for (β, η) in lower_cases
        @test S(0.0, β, η) == 0.0
        fd = (S(h, β, η) - S(0.0, β, η)) / h
        @test fd > 0
        for p in (1.0, 0.25)
            got = DEO.DSP_dpf(p, 0.0, flat; R, β, η)
            @test got ≈ fd * p rtol = 2e-5
            # regression: the endpoint uses the stable form, which agrees with the plain
            # formula wherever that one still has its digits
            @test got == DEO.∂projection_∂pf_stable(0.0, β, η) * p
            @test got ≈ DEO.∂projection_∂pf(0.0, β, η) * p rtol = 1e-9
        end
        # no jump between the endpoint and the nearby flat field
        @test DEO.DSP_dpf(1.0, 0.0, flat; R, β, η) ≈
              DEO.DSP_dpf(1.0, 1e-12, flat; R, β, η) rtol = 1e-9
    end

    @testset "upper bound ρ̃ = 1, one-sided into ρ̃ < 1 (β = $β, η = $η)" for (β, η) in upper_cases
        fd = (S(1.0, β, η) - S(1.0 - h, β, η)) / h
        @test fd > 0
        got = DEO.DSP_dpf(1.0, 1.0, flat; R, β, η)
        @test got ≈ fd rtol = 2e-5
    end

    @testset "interior control (β = $β)" for β in (8.0, 32.0)
        η = 0.5
        for ρ̃ in (0.3, 0.45, 0.7)
            fd = (S(ρ̃ + h, β, η) - S(ρ̃ - h, β, η)) / (2h)
            @test DEO.DSP_dpf(1.0, ρ̃, flat; R, β, η) ≈ fd rtol = 1e-5
        end
    end

    @testset "sidewall partials at ρ̃ = 0" begin
        β, η, slope, z0, z = 8.0, 0.5, 0.1, 0.0, 3.0
        Ssw(ρ̃) = DEO.smooth_project_sidewall(ρ̃, flat, z; R, β, η, slope, z0)
        fd = (Ssw(h) - Ssw(0.0)) / h
        a, c = DEO.sidewall_partials(0.0, flat, z; R, β, η, slope, z0)
        @test a ≈ fd rtol = 2e-5
        @test c == 0.0
        control = ctl(β, η; use_sidewall=true, sidewall_slope=slope, sidewall_z0=z0)
        @test DEO.validate(control)
        @test DEO.dpt_dpf_sidewall(1.0, flat, 0.0, flat, z; control) ≈ fd rtol = 2e-5
    end

    # Reference values of the one-sided slope at ρ̃ = 0,
    #     P′(0) = β·sech²(βη) / (tanh(βη) + tanh(β(1 − η))),
    # computed offline with Python's `decimal` module at 1200 significant digits, with β and η
    # taken as the exact values of their Float64 representations, and cross-checked there
    # against a one-sided difference of the forward formula with h = 1e-60 (relative
    # agreement better than 1e-56; for the last six cases h = 1e-700 and better than
    # 1e-178). Below they are checked again with BigFloat.
    # The β values are those of `optimize!`'s default schedule (and 40, where the plain
    # formula already returns 0 at η = 0.5). At β = 1024 the slope, 3.9e-442, is below the
    # smallest positive Float64, so its Float64 value is 0. β = 600 and 602 at η = 0.5 lie on
    # either side of the switch between the direct and the log form (a = 2βη = 600 and 602,
    # see `∂projection_∂pf_stable`). In the last four cases Float64
    # rounds the factor e^{−2βη} to 0, or to its smallest positive number at β = 379.75,
    # although the slope is representable: three subnormal slopes (β = 746 and 379.75, and
    # β = 1024 at η = 373/1024 on the default schedule) and a normal one (β = 1e300,
    # η = 3.73e-298 = 373/1e300).
    endpoint_reference = [
        (8.0, 0.5, "5.367402650461784479473600370588e-3"),
        (16.0, 0.5, "3.601125591016337269691453440425e-6"),
        (32.0, 0.5, "8.105065951420272462797378661118e-13"),
        (40.0, 0.5, "3.398683404233271196263387826287e-16"),
        (64.0, 0.5, "2.052877939902256451809391403702e-26"),
        (128.0, 0.5, "6.584855993964581956709217494038e-54"),
        (256.0, 0.5, "3.387525661035256453647119773369e-109"),
        (512.0, 0.5, "4.482550821942324689020388109212e-220"),
        (1024.0, 0.5, "3.924465209237463019281600904084e-442"),
        (32.0, 0.15, "4.334345575665294878595035118488e-3"),
        (64.0, 0.15, "5.871592608774768611433900589460e-7"),
        (600.0, 0.5, "3.180475863605172979606415336723e-258"),
        (602.0, 0.5, "4.318653705009771625206685382856e-259"),
        (746.0, 0.5, "1.549120935797615733222113608949e-321"),
        (379.75, 0.9810838961184215, "1.876215371189505035433757980921e-321"),
        (1024.0, 373 / 1024, "2.126407289888416234342418680381e-321"),
        (1e300, 3.73e-298, "2.076569619031655544538686851749e-24"),
    ]
    # Float64 accuracy of the slope (see `∂projection_∂pf_stable`), with a = 2βη at ρ̃ = 0:
    # - a ≤ 600 (direct form): within 48 eps relative plus twice the rounding error of the
    #   product β·η (zero for every such row here: β·η is exact). The direct form is within
    #   about one eps; 48 eps leaves room for a different libm while the log form, about 85 eps
    #   off at β = 512 and 176 eps at β = 600, still fails;
    # - a > 600 (log form): within 4·(a + |log D′| + 4)·eps relative, with
    #   D′ = (tanh(βη) + tanh(β(1 − η)))/β, plus one unit in the last place of the smallest
    #   subnormal for the final rounding of a subnormal (or zero) result.
    function logD(β, η)
        setprecision(BigFloat, 4096) do
            βb, ηb = big(β), big(η)
            Float64(abs(log((tanh(βb * ηb) + tanh(βb * (1 - ηb))) / βb)))
        end
    end
    function matches(got, ref, β, η)
        a = 2 * abs(β * η)
        if a <= 600
            δ = setprecision(BigFloat, 4096) do
                abs(big(β * η) - big(β) * big(η))
            end
            return abs(big(got) - ref) <= (2 * δ + 48 * eps()) * ref
        end
        return abs(big(got) - ref) <= 4 * eps() * (a + logD(β, η) + 4) * ref + big(nextfloat(0.0))
    end

    @testset "endpoint slope against a high-precision reference (β = $β, η = $η)" for (β, η, lit) in endpoint_reference
        ref = setprecision(() -> parse(BigFloat, lit), BigFloat, 256)
        # regression: the literal is the slope (BigFloat evaluation of the closed form)
        bigref = setprecision(BigFloat, 4096) do
            βb, ηb = big(β), big(η)
            βb / cosh(βb * ηb)^2 / (tanh(βb * ηb) + tanh(βb * (1 - ηb)))
        end
        @test abs(bigref - ref) <= big"1e-28" * ref
        # regression: the stable form in BigFloat reproduces the literal
        st = setprecision(BigFloat, 256) do
            DEO.∂projection_∂pf_stable(big(0.0), big(β), big(η))
        end
        @test abs(st - ref) <= big"1e-28" * ref
        # regression: every helper that takes the flat branch gives the Float64 slope, also
        # where Float64 cannot represent e^{−2βη} on its own
        for p in (1.0, 0.25)
            @test matches(DEO.DSP_dpf(p, 0.0, flat; R, β, η), ref * p, β, η)
            @test DEO.DSP_dpf(p, 0.0, flat; R, β, η) > 0 || Float64(ref * p) == 0
        end
        a, c = DEO.sidewall_partials(0.0, flat, 3.0; R, β, η, slope=0.1, z0=0.0)
        @test matches(a, ref, β, η)
        @test c == 0.0
        control = ctl(β, η; use_sidewall=true, sidewall_slope=0.1, sidewall_z0=0.0)
        @test DEO.validate(control)
        @test matches(DEO.dpt_dpf_sidewall(1.0, flat, 0.0, flat, 3.0; control), ref, β, η)
    end

    @testset "endpoint slope from β = 700 to 760 at η = 0.5, across the bottom of the normal range" begin
        # regression: the log form was described as within about one unit in the last place for a
        # subnormal result. Just below the normal range it is not: at β = 716 the result is 51
        # units in the last place from the correctly rounded slope (28 at 717, 9 at 718), because
        # the exponent near −716 is rounded before it is exponentiated. The stated bound is the
        # relative one of the normal range plus the final rounding. Every integer β here (a = β)
        # is checked against BigFloat: normal slopes up to β = 715, subnormal ones from 716, and a
        # correctly rounded slope of 0 from β = 753 (the result may then be 0 or the smallest
        # positive Float64, within the bound).
        n_normal = n_subnormal = n_zero = 0
        for β in 700.0:1.0:760.0
            ref = setprecision(BigFloat, 4096) do
                βb, ηb = big(β), big(0.5)
                βb / cosh(βb * ηb)^2 / (tanh(βb * ηb) + tanh(βb * (1 - ηb)))
            end
            got = DEO.∂projection_∂pf_stable(0.0, β, 0.5)
            @test matches(got, ref, β, 0.5)
            r = Float64(ref)
            r == 0 ? (n_zero += 1) : issubnormal(r) ? (n_subnormal += 1) : (n_normal += 1)
        end
        @test n_normal > 0 && n_subnormal > 0 && n_zero > 0          # the range crosses both edges
        # The miss at β = 716 is bounded (no lower limit: a correctly rounded result also passes).
        ulps = setprecision(BigFloat, 4096) do
            ref = big(716.0) / cosh(big(358.0))^2 / (2 * tanh(big(358.0)))
            Float64(abs(big(DEO.∂projection_∂pf_stable(0.0, 716.0, 0.5)) - ref) / big(nextfloat(0.0)))
        end
        @test ulps < 200
        # A fractional β and an asymmetric η, where the subnormal miss is larger (hundreds of units), still
        # within the stated relative bound.
        for (β, η) in ((715.7, 0.5), (715.0, 0.5005))
            ref = setprecision(BigFloat, 4096) do
                βb, ηb = big(β), big(η)
                βb / cosh(βb * ηb)^2 / (tanh(βb * ηb) + tanh(βb * (1 - ηb)))
            end
            @test matches(DEO.∂projection_∂pf_stable(0.0, β, η), ref, β, η)
        end
    end

    @testset "products with the unit slope are ordinary Float64 products (documented limit)" begin
        # The helpers return the correct unit slope; a product with a direction p is formed in Float64. Where
        # the unit slope is subnormal the product can underflow although the exact product is representable.
        # These cases pin that documented behaviour, so a change to it is noticed.
        exact(β, η, p) = setprecision(BigFloat, 4096) do
            βb, ηb = big(β), big(η)
            big(p) * βb / cosh(βb * ηb)^2 / (tanh(βb * ηb) + tanh(βb * (1 - ηb)))
        end
        # β = 751.5: the unit slope is about 1.29 smallest-subnormal units, the exact product with p = 0.5
        # rounds to one unit, and the helper returns 0.
        @test Float64(exact(751.5, 0.5, 0.5)) > 0
        @test DEO.DSP_dpf(0.5, 0.0, flat; R, β=751.5, η=0.5) == 0.0
        # β = 1024: the unit slope is below the smallest positive Float64 and rounds to 0, so even a normal
        # exact product (direction 1e200, about 3.9e-242) comes out 0.
        @test Float64(exact(1024.0, 0.5, 1e200)) > 1e-300
        @test DEO.DSP_dpf(1e200, 0.0, flat; R, β=1024.0, η=0.5) == 0.0
    end

    @testset "the direct and the log form agree where they meet" begin
        # At a = 600 both internal forms apply to the same (a, D′). They agree within the log
        # form's own error bound (about 4e-14 relative in a Float64 port of these formulas), so
        # the switch adds no jump beyond that.
        a = 600.0
        den = 0.5 * DEO._tanhc(300.0) + 0.5 * DEO._tanhc(300.0)
        direct = DEO._endpoint_slope_direct(a, den)
        logform = DEO._endpoint_slope_log(a, den)
        # regression: the switch sits where both forms are accurate
        @test abs(direct - logform) <= 4 * eps() * (a + abs(log(den)) + 4) * direct
        @test DEO.∂projection_∂pf_stable(0.0, 600.0, 0.5) === direct
        @test DEO.∂projection_∂pf_stable(0.0, 602.0, 0.5) ===
              DEO._endpoint_slope_log(602.0, 0.5 * DEO._tanhc(301.0) + 0.5 * DEO._tanhc(301.0))
    end

    @testset "endpoint slope at the smallest positive β (β = $β, η = $η)" for β in (nextfloat(0.0), 1e-300), η in (0.5, 0.15)
        # The slope tends to 1 as β → 0⁺; at these β it differs from 1 by less than 1e-600
        # (same offline computation as above). Both Control values pass the validator.
        control = ctl(β, η; use_sidewall=true, sidewall_slope=0.1, sidewall_z0=0.0)
        @test DEO.validate(control)
        # regression: no Inf or NaN where βη underflows
        @test DEO.DSP_dpf(1.0, 0.0, flat; R, β, η) ≈ 1.0 rtol = 1e-14
        @test DEO.DSP_dpf(0.25, 0.0, flat; R, β, η) ≈ 0.25 rtol = 1e-14
        @test DEO.DSP_dpf(0.0, 0.0, flat; R, β, η) == 0.0
        a, c = DEO.sidewall_partials(0.0, flat, 3.0; R, β, η, slope=0.1, z0=0.0)
        @test a ≈ 1.0 rtol = 1e-14
        @test c == 0.0
        @test DEO.dpt_dpf_sidewall(1.0, flat, 0.0, flat, 3.0; control) ≈ 1.0 rtol = 1e-14
        @test DEO.dpt_dpf_sidewall(0.0, flat, 0.0, flat, 3.0; control) == 0.0
        bigref = setprecision(BigFloat, 4096) do
            βb, ηb = big(β), big(η)
            βb / cosh(βb * ηb)^2 / (tanh(βb * ηb) + tanh(βb * (1 - ηb)))
        end
        @test abs(bigref - 1) < big"1e-290"
    end

    @testset "other flat points keep the plain formula bit for bit" begin
        # At β = 64, η = 0.5 the plain formula already returns 0 at ρ̃ = 1 and at ρ̃ = 0, while
        # the stable form gives the representable slope 2.05e-26 there. ρ̃ = 1 with ∇ρ̃ = 0 and
        # ρ̃ = 0 with a small nonzero ∇ρ̃ (still flat) each fail one of the two conditions.
        β, η = 64.0, 0.5
        small = DEO.VectorValue(1e-12, 0.0, 0.0)
        control = ctl(β, η; use_sidewall=true, sidewall_slope=0.1, sidewall_z0=0.0)
        @test DEO.validate(control)
        for (ρ̃, ∇ρ̃) in ((1.0, flat), (0.0, small))
            plain = DEO.∂projection_∂pf(ρ̃, β, η)
            @test plain != DEO.∂projection_∂pf_stable(ρ̃, β, η)
            # regression: the stable form is used only where ρ̃ = 0 and ∇ρ̃ = 0 both hold
            for p in (1.0, 0.25)
                @test DEO.DSP_dpf(p, ρ̃, ∇ρ̃; R, β, η) === plain * p
            end
            @test DEO.sidewall_partials(ρ̃, ∇ρ̃, 3.0; R, β, η, slope=0.1, z0=0.0) === (plain, 0.0)
            @test DEO.dpt_dpf_sidewall(1.0, flat, ρ̃, ∇ρ̃, 3.0; control) === plain
        end
    end

    @testset "endpoint slope at β = ∞ is the physical zero" begin
        # The forward step is constant near ρ̃ = 0 for 0 < η < 1, so the one-sided slope is 0.
        β, η = Inf, 0.5
        for p in (1.0, 0.25, 0.0)
            # regression: exactly 0, not NaN
            @test DEO.DSP_dpf(p, 0.0, flat; R, β, η) === 0.0
        end
        @test DEO.sidewall_partials(0.0, flat, 3.0; R, β, η, slope=0.1, z0=0.0) === (0.0, 0.0)
        control = ctl(β, η; use_sidewall=true, sidewall_slope=0.1, sidewall_z0=0.0)
        @test DEO.validate(control)
        @test DEO.dpt_dpf_sidewall(1.0, flat, 0.0, flat, 3.0; control) === 0.0
    end
end

@testset "SSP projection and FE assembly derivative at a flat design" begin
    # g(ρ̃) = ∫ SSP(ρ̃) dΩ on P1 hexahedra, with ρ̃ given directly as the filtered FE
    # field (the filter is not part of this check); its gradient is assembled with
    # `dpt_dpf_field`, as in every ∂g/∂pf assembly. A uniform direction keeps the
    # filtered field flat, so the one-sided difference of g is a true directional
    # derivative. A zero is accepted only where the forward difference is also zero.
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1))
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    n = num_free_dofs(V)
    g(vals, control) = sum(∫(DEO.project_ssp(FEFunction(V, vals), control, Ω))dΩ)
    function grad(vals, control)
        pf = FEFunction(V, vals)
        assemble_vector(v -> ∫(DEO.dpt_dpf_field(v, pf, control, Ω))dΩ, V)
    end
    h = 1e-7

    # (name, β, η, base value, direction, rtol, Control keywords). The sidewall cases go through the
    # `dpt_dpf_sidewall` branch of `dpt_dpf_field`; a flat field is flat for any lean.
    sidewall = (use_sidewall=true, sidewall_slope=0.1, sidewall_z0=0.0)
    cases = [
        ("all void, into the interval, β = 8", 8.0, 0.5, 0.0, 1.0, 2e-5, (;)),
        ("all void, into the interval, β = 32", 32.0, 0.15, 0.0, 1.0, 2e-5, (;)),
        ("all solid, into the interval, β = 8", 8.0, 0.5, 1.0, -1.0, 2e-5, (;)),
        ("all solid, into the interval, β = 32", 32.0, 0.85, 1.0, -1.0, 2e-5, (;)),
        ("interior control", 8.0, 0.5, 0.3, 1.0, 2e-5, (;)),
        ("all void with a sidewall, β = 8", 8.0, 0.5, 0.0, 1.0, 2e-5, sidewall),
        ("interior control with a sidewall", 8.0, 0.5, 0.3, 1.0, 2e-5, sidewall),
    ]
    @testset "$name" for (name, β, η, base, dir, rtol, kw) in cases
        control = Control(; R_ssp=2.0, β, η, η_dilation=η / 2, η_erosion=(1 + η) / 2, kw...)
        @test DEO.validate(control)
        ρ̃ = fill(base, n)
        w = fill(dir, n)
        fd = (g(ρ̃ .+ h .* w, control) - g(ρ̃, control)) / h
        ad = dot(grad(ρ̃, control), w)
        @test fd != 0
        @test ad ≈ fd rtol = rtol
    end

    @testset "all void at β = $β: assembled slope equals the endpoint slope" for (β, lit) in
            ((64.0, "2.052877939902256451809391403702e-26"), (128.0, "6.584855993964581956709217494038e-54"))
        # A Float64 difference of g cannot resolve these slopes (tanh rounds to ±1 in the
        # forward), so the assembly is checked against the reference slope instead
        # (see the endpoint reference testset above). ∫ 1 dΩ = 1 and the P1 basis sums to 1.
        control = Control(R_ssp=2.0, β=β, η=0.5)
        @test DEO.validate(control)
        ad = dot(grad(zeros(n), control), ones(n))
        # regression: the all-void gradient at this β is the endpoint slope, not 0
        @test ad ≈ Float64(parse(BigFloat, lit)) rtol = 1e-12
    end

    @testset "physical zero: β = ∞ at the void endpoint" begin
        control = Control(R_ssp=2.0, β=Inf, η=0.5)
        ρ̃ = zeros(n)
        w = ones(n)
        fd = (g(ρ̃ .+ h .* w, control) - g(ρ̃, control)) / h
        ad = dot(grad(ρ̃, control), w)
        @test fd == 0
        @test ad == 0
    end
end

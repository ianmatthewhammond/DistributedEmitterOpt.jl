using Test
using DistributedEmitterOpt

# Damage (quenching) settings: the factor 1 / (1 + exp(γ (E² - E_th²))) must attenuate the
# signal at high intensity. Invalid settings used to disable it (NaN, ±Inf γ) or reverse it
# (γ < 0), and γ = 0 gives a constant factor 1/2 that is not a threshold at all.
#
# Allowed with use_damage_model = true:
#   γ_damage    finite and > 0   (0, negative, NaN and ±Inf are rejected)
#   E_threshold >= 0 (0 is an active setting centred at zero intensity), negative and NaN rejected, Inf allowed (= no damage)
# With use_damage_model = false the two fields are not read, so they are not validated.

const _VV = DistributedEmitterOpt.VectorValue
const _dmg = DistributedEmitterOpt.damage_factor
const _∂dmg = DistributedEmitterOpt.∂damage_∂E

@testset "damage settings" begin
    @testset "Damage settings validation" begin
        @testset "Invalid steepness rejected" begin
            for γ in (NaN, 0.0, -0.0, -1.0, -1e-9, Inf, -Inf)
                @test_throws ArgumentError SERSObjective(use_damage_model=true, γ_damage=γ, E_threshold=3.0)
                @test_throws ArgumentError SERSObjective(use_damage_model=true, γ_damage=γ, E_threshold=Inf)
            end
        end

        @testset "Invalid threshold rejected" begin
            for E_th in (NaN, -1e-9, -3.0, -Inf)
                @test_throws ArgumentError SERSObjective(use_damage_model=true, γ_damage=1.0, E_threshold=E_th)
            end
        end

        @testset "Every construction route is checked (positional inner constructor)" begin
            α = ComplexF64[1 0 0; 0 1 0; 0 0 1]
            make(use_damage, γ, E_th) = SERSObjective(α, use_damage, γ, E_th, true, false, :band, 10.0, 1e-6, true,
                Ref{Any}(nothing))
            @test_throws ArgumentError make(true, NaN, 3.0)
            @test_throws ArgumentError make(true, 1.0, NaN)
            @test make(true, 1.0, 3.0) isa SERSObjective
            @test make(false, NaN, NaN) isa SERSObjective
        end

        @testset "Valid settings accepted" begin
            obj = SERSObjective(use_damage_model=true, γ_damage=0.5, E_threshold=3.0)
            @test obj.use_damage_model
            @test obj.γ_damage == 0.5
            @test obj.E_threshold == 3.0
            # Infinite threshold is allowed: damage enabled but never active.
            # E_threshold = 0 (and -0.0) is a valid active setting: the factor is 1/2 at zero
            # intensity and falls with it. Existing gradient tests use it to make damage active.
            for E_th0 in (0.0, -0.0)
                obj0 = SERSObjective(use_damage_model=true, γ_damage=4.0, E_threshold=E_th0)
                @test obj0.E_threshold == 0.0
            end
            @test _dmg(0.0; γ=4.0, E_th=0.0) == 0.5
            @test _dmg(1.0; γ=4.0, E_th=0.0) < _dmg(0.25; γ=4.0, E_th=0.0) < 0.5
            obj_inf = SERSObjective(use_damage_model=true, γ_damage=1.0, E_threshold=Inf)
            @test obj_inf.E_threshold == Inf
            @test _dmg(0.0; γ=1.0, E_th=Inf) == 1.0
            @test _dmg(36.0; γ=1.0, E_th=Inf) == 1.0
            # Defaults (damage off, γ = 1, no threshold) are unchanged and valid.
            obj_def = SERSObjective()
            @test !obj_def.use_damage_model
            @test obj_def.γ_damage == 1.0
            @test obj_def.E_threshold == Inf
            @test SERSObjective(use_damage_model=true).E_threshold == Inf
        end

        @testset "Not validated when the damage model is off" begin
            for (γ, E_th) in ((NaN, NaN), (-1.0, -3.0), (0.0, 0.0), (Inf, NaN))
                obj = SERSObjective(use_damage_model=false, γ_damage=γ, E_threshold=E_th)
                @test !obj.use_damage_model
                @test isequal(obj.γ_damage, γ)
                @test isequal(obj.E_threshold, E_th)
            end
        end
    end

    @testset "Settings are converted before they are validated, and the converted values are stored" begin
        α = ComplexF64[1 0 0; 0 1 0; 0 0 1]
        make(use_damage, γ, E_th) = SERSObjective(α, use_damage, γ, E_th, true, false, :band, 10.0, 1e-6, true,
            Ref{Any}(nothing))
        kw(use_damage, γ, E_th) = SERSObjective(use_damage_model=use_damage, γ_damage=γ, E_threshold=E_th)
        for route in (make, kw)
            # regression: a sharpness that overflows Float64 on conversion is Inf, not a valid value
            @test_throws ArgumentError route(true, big"1e400", 3.0)
            @test_throws ArgumentError route(true, big"1e400", Inf)
            @test_throws ArgumentError route(true, big"-1e400", 3.0)
            # regression: a sharpness that underflows on conversion is 0.0, which is not a threshold
            @test_throws ArgumentError route(true, big"1e-400", 3.0)
            @test_throws ArgumentError route(true, big"1e-400", Inf)
            # regression: thresholds are validated after conversion as well
            @test_throws ArgumentError route(true, 1.0, big"-1e400")
            # an overflowing threshold is +Inf (no damage) and an underflowing one is +0.0: both valid
            @test route(true, 1.0, big"1e400").E_threshold === Inf
            @test route(true, 1.0, big"1e-400").E_threshold === 0.0
            @test route(true, 1.0, -big"1e-400").E_threshold === -0.0
            # representable high-precision values are stored as the Float64 the solver uses
            obj = route(true, big"2.5", big"3.5")
            @test obj.γ_damage === 2.5 && obj.E_threshold === 3.5
            # model off: nothing is validated, the converted values are stored
            off = route(false, big"1e400", big"1e-400")
            @test !off.use_damage_model
            @test off.γ_damage === Inf && off.E_threshold === 0.0
        end
    end

    @testset "Calls that converted to valid settings are still accepted" begin
        α = ComplexF64[1 0 0; 0 1 0; 0 0 1]
        make(use_damage, γ, E_th) = SERSObjective(α, use_damage, γ, E_th, true, false, :band, 10.0, 1e-6, true,
            Ref{Any}(nothing))
        kw(use_damage, γ, E_th) = SERSObjective(use_damage_model=use_damage, γ_damage=γ, E_threshold=E_th)
        for route in (make, kw)
            # regression: integer flags are converted to Bool before they are tested
            off = route(0, NaN, NaN)
            @test off.use_damage_model === false
            @test isnan(off.γ_damage) && isnan(off.E_threshold)
            on = route(1, 2.0, 3.0)
            @test on.use_damage_model === true
            @test on.γ_damage === 2.0 && on.E_threshold === 3.0
            # regression: real-valued complex settings convert to the same Float64 as before
            c = route(true, 1 + 0im, 2 + 0im)
            @test c.γ_damage === 1.0 && c.E_threshold === 2.0
            @test route(false, 1 + 0im, 2 + 0im).γ_damage === 1.0
            # a value that does not convert raises what the conversion raises, with the model on or off
            @test_throws InexactError route(true, 1 + 1im, 3.0)
            @test_throws InexactError route(true, 1.0, 2 + 1im)
            @test_throws InexactError route(false, 1 + 1im, 3.0)
            @test_throws InexactError route(2, 1.0, 3.0)
            # a converted value is still validated
            @test_throws ArgumentError route(1, 0.0, 3.0)
            @test_throws ArgumentError route(true, 1 + 0im, -2 + 0im)
        end
    end

    @testset "Damage factor for valid active settings" begin
        for (γ, E_th) in ((2.0, 3.0), (0.5, 3.0), (1.0, 1.5), (5.0, 6.0))
            E²_grid = range(0.0, 2 * E_th^2; length=401)
            vals = [_dmg(u; γ=γ, E_th=E_th) for u in E²_grid]
            @test all(v -> 0.0 < v <= 1.0, vals)
            @test issorted(vals; rev=true)                    # non-increasing in intensity
            @test _dmg(E_th^2; γ=γ, E_th=E_th) ≈ 0.5          # half-quenched at the threshold
            @test vals[end] < vals[1]
        end

        # Well below threshold the signal is untouched.
        @test _dmg(0.0; γ=2.0, E_th=3.0) ≈ 1.0 atol = 1e-6
        # Well above threshold it is strongly attenuated (the invalid γ = -1 case rose to ~1 here).
        @test _dmg(36.0; γ=2.0, E_th=3.0) < 1e-6

        @testset "Field derivative matches central finite difference near threshold" begin
            for (γ, E_th) in ((0.8, 3.0), (2.0, 3.0))
                # |E|² = 9.5 (just above E_th² = 9), 8.5 (just below) and 9 (at threshold).
                for E in (_VV(2.5, 1.0, 1.5), _VV(2.0, 1.5, 1.5), _VV(3.0, 0.0, 0.0))
                    d = _∂dmg(E; γ=γ, E_th=E_th)
                    f(c) = _dmg(sum(abs2, c); γ=γ, E_th=E_th)
                    c0 = collect(Tuple(E))
                    h = 1e-6
                    fd = map(1:3) do i
                        cp = copy(c0); cp[i] += h
                        cm = copy(c0); cm[i] -= h
                        (f(cp) - f(cm)) / (2h)
                    end
                    @test collect(Tuple(d)) ≈ fd rtol = 1e-5 atol = 1e-8
                end
            end
        end
    end
end

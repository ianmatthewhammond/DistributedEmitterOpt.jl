using Test
using DistributedEmitterOpt

# Surface-shell settings of SERSObjective. Both are stored as Float64 and are checked on the stored
# value, so a wider-typed input outside the Float64 range is refused instead of being stored as Inf
# or 0.0. Allowed:
#   t_shell   finite and > 0, with t_shell^2 finite and > 0 (the shell filter assembles t_shell^2)
#   ε_grad    :tv_void: finite and > 0, with ε_grad^2 finite and > 0 (|∇χ_s|_ε = sqrt(|∇χ_s|^2 + ε_grad^2))
#             :band:    finite and >= 0 (not used)

const _α_shell = ComplexF64[1 0 0; 0 1 0; 0 0 1]

# Both construction routes: the keyword constructor and the positional inner constructor.
_shell_kw(mode, t, ε) = SERSObjective(surface=true, surface_mode=mode, t_shell=t, ε_grad=ε)
_shell_pos(mode, t, ε; cache=Ref{Any}(nothing)) =
    SERSObjective(_α_shell, false, 1.0, Inf, true, true, mode, t, ε, true, cache)

# The message of the ArgumentError a call throws ("" if it throws nothing or something else).
function _shell_err(f)
    try
        f()
    catch e
        return e isa ArgumentError ? e.msg : ""
    end
    return ""
end

@testset "shell settings" begin
    @testset "t_shell is converted before it is validated" begin
        for route in (_shell_kw, _shell_pos), mode in (:band, :tv_void)
            # regression: overflows to Inf on conversion
            @test_throws ArgumentError route(mode, big"1e400", 1e-6)
            # regression: underflows to 0.0 on conversion
            @test_throws ArgumentError route(mode, big"1e-400", 1e-6)
            @test_throws ArgumentError route(mode, -big"1e-400", 1e-6)
            # representable wider-typed values are stored as the Float64 the solver uses
            obj = route(mode, big"7.5", 1e-6)
            @test obj.t_shell === 7.5
        end
    end

    @testset "ε_grad is converted before it is validated" begin
        for route in (_shell_kw, _shell_pos)
            # regression: underflows to 0.0, the value :tv_void exists to refuse (0/0 in flat regions)
            @test_throws ArgumentError route(:tv_void, 10.0, big"1e-400")
            # regression: overflows to Inf on conversion, in either mode
            @test_throws ArgumentError route(:tv_void, 10.0, big"1e400")
            @test_throws ArgumentError route(:band, 10.0, big"1e400")
            # :band does not use ε_grad and allows 0, so an underflow to 0.0 is stored as 0.0
            @test route(:band, 10.0, big"1e-400").ε_grad === 0.0
            @test route(:tv_void, 10.0, big"0.25").ε_grad === 0.25
        end
    end

    @testset "The error names the setting and the value it would be stored as" begin
        for route in (_shell_kw, _shell_pos)
            m = _shell_err(() -> route(:tv_void, big"1e400", 1e-6))
            @test occursin("t_shell", m) && occursin("Inf", m)
            m = _shell_err(() -> route(:band, big"1e-400", 1e-6))
            @test occursin("t_shell", m) && occursin("0.0", m)
            m = _shell_err(() -> route(:tv_void, 10.0, big"1e-400"))
            @test occursin("ε_grad", m) && occursin("0.0", m)
            m = _shell_err(() -> route(:band, 10.0, big"1e400"))
            @test occursin("ε_grad", m) && occursin("Inf", m)
        end
    end

    @testset "Squares that leave the Float64 range are refused" begin
        for route in (_shell_kw, _shell_pos)
            for mode in (:band, :tv_void)
                # t_shell^2 = Inf and t_shell^2 = 0.0 for finite positive t_shell
                @test_throws ArgumentError route(mode, 1e200, 1e-6)
                @test_throws ArgumentError route(mode, 1e-200, 1e-6)
                @test occursin("t_shell", _shell_err(() -> route(mode, 1e200, 1e-6)))
                # squares inside the range are accepted
                @test route(mode, 1e150, 1e-6).t_shell === 1e150
                @test route(mode, 1e-150, 1e-6).t_shell === 1e-150
            end
            # ε_grad^2 = 0.0 is the 0/0 again, ε_grad^2 = Inf an infinite measure
            @test_throws ArgumentError route(:tv_void, 10.0, 1e-200)
            @test_throws ArgumentError route(:tv_void, 10.0, 1e200)
            @test occursin("ε_grad", _shell_err(() -> route(:tv_void, 10.0, 1e-200)))
            @test route(:tv_void, 10.0, 1e-150).ε_grad === 1e-150
            # :band does not use ε_grad: its square is not checked
            @test route(:band, 10.0, 1e-200).ε_grad === 1e-200
        end
    end

    @testset "Calls that converted to valid settings are still accepted" begin
        for route in (_shell_kw, _shell_pos)
            obj = route(:tv_void, 7, 1//4)
            @test obj.t_shell === 7.0 && obj.ε_grad === 0.25
            @test route(:tv_void, 10 + 0im, 1e-6).t_shell === 10.0
            @test route(:band, 10.0, 0).ε_grad === 0.0
            # a value that does not convert raises what the conversion raises
            @test_throws InexactError route(:tv_void, 1 + 1im, 1e-6)
            @test_throws InexactError route(:tv_void, 10.0, 1e-6 + 1im)
        end
        # defaults are unchanged
        obj = SERSObjective()
        @test obj.t_shell === 10.0 && obj.ε_grad === 1e-6
        obj = SERSObjective(surface=true, surface_mode=:tv_void)
        @test obj.t_shell === 10.0 && obj.ε_grad === 1e-6
    end

    @testset "A refused setting leaves the passed shell cache untouched" begin
        cache = Ref{Any}(nothing)
        @test_throws ArgumentError _shell_pos(:tv_void, big"1e400", 1e-6; cache=cache)
        @test_throws ArgumentError _shell_pos(:tv_void, 10.0, big"1e-400"; cache=cache)
        @test cache[] === nothing
    end
end

using DistributedEmitterOpt, Test
@testset "Cited aluminium and titanium datasets" begin
    n,k=DistributedEmitterOpt.refindex("Al")
    @test n(325.0) ≈ 0.31201764705882357 atol=1e-6
    @test k(325.0) ≈ 3.9001549019607844 atol=1e-5
    @test n(326.28) ≈ 0.31474 atol=1e-12
    @test k(326.28) ≈ 3.9165 atol=1e-12
    n,k=DistributedEmitterOpt.refindex("Ti")
    @test n(325.0) ≈ 1.575 atol=1e-4
    @test k(325.0) ≈ 2.6933333333333334 atol=1e-4
    for material in ("Al","Al_McPeak","Ti","Ti_Palm")
        n,k=DistributedEmitterOpt.refindex(material)
        @test isfinite(n(488.0)) && n(488.0)>0
        @test isfinite(k(488.0)) && k(488.0)>=0
    end
end

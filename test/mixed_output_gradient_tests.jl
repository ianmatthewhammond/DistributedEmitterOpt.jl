include("gradient_common.jl")
BLAS.set_num_threads(2)
@testset "Shared-model mixed-polarization gradient regression" begin
    for foundry in (false,true)
        @testset "foundry=$foundry" begin
            Random.seed!(941)
            prob=build_test_problem(foundry_mode=foundry,complex_config=true,isotropic=false)
            p=0.4 .+ 0.2 .* rand(length(prob.p))
            grad=zeros(length(p));objective_and_gradient!(grad,p,prob)
            d=randn(length(p));d ./= norm(d)
            for h in (1e-3,1e-4,1e-5)
                adj,fd,rel=only(central_fd_check(prob,p,grad,[d];δ=h))
                println("MIXED foundry=$foundry h=$h adj=$adj fd=$fd relative=$rel");flush(stdout)
                @test rel<TOL_RELATIVE
            end
        end
    end
end

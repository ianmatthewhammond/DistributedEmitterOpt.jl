include("oblique_common.jl")
struct UnconjugatedAdjointSolver <: DEO.AbstractSolver end
DEO.lu!(::UnconjugatedAdjointSolver,A)=lu(A)
DEO.filter_lu!(::UnconjugatedAdjointSolver,A)=lu(A)
DEO.solve!(::UnconjugatedAdjointSolver,F,b::Vector)=F\b
DEO.solve_adjoint!(::UnconjugatedAdjointSolver,F,b::Vector)=transpose(F)\b

@testset "Adjoint conjugation negative control" begin
    fc=FieldConfig(532;tilt_x=20,tilt_y=10,pol=:x)
    good=angular_test_problem([fc],[fc])
    bad=OptimizationProblem(good.pde,good.objective,good.sim,UnconjugatedAdjointSolver();
        foundry_mode=true,control=good.control)
    _,valid=angular_ladder(good)
    _,broken=angular_ladder(bad)
    println("NEGATIVE valid=$(minimum(valid)) unconjugated=$(minimum(broken))")
    @test valid[end]<1e-4
    @test minimum(broken)>1e-2
end

include("oblique_common.jl")
using SparseArrays

@testset "Maxwell symmetric-storage guard" begin
    complex_symmetric = sparse(ComplexF64[2+im 1+im; 1+im 3-im])
    nonsymmetric = sparse(ComplexF64[2+im 1+im; -1+im 3-im])
    hermitian = sparse(ComplexF64[2 1+im; 1-im 3])
    @test DEO.checked_maxwell_mtype(PardisoSolver(assume_symmetric_maxwell=true), complex_symmetric) == 6
    @test DEO.checked_maxwell_mtype(PardisoSolver(), nonsymmetric) == 13
    @test_throws ArgumentError DEO.checked_maxwell_mtype(PardisoSolver(assume_symmetric_maxwell=true), nonsymmetric)
    @test_throws ArgumentError DEO.checked_maxwell_mtype(PardisoSolver(mtype=6), nonsymmetric)
    @test DEO.checked_maxwell_mtype(PardisoSolver(mtype=4), hermitian) == 4
    @test_throws ArgumentError DEO.checked_maxwell_mtype(PardisoSolver(mtype=4), complex_symmetric)
    for fc in (FieldConfig(532; tilt_x=0, pol=:y),
               FieldConfig(532; tilt_x=20, pol=:x),
               FieldConfig(532; tilt_x=20, tilt_y=10, pol=:x))
        prob = angular_test_problem([fc], [fc])
        sim = DEO.sim_for(prob.sim, fc)
        pt = DEO.interpolate(x -> 0.5, sim.Pf)
        phys = DEO.build_phys_params(fc, prob.pde.env, sim)
        A = assemble_maxwell(pt, sim, phys)
        defect = norm(A - transpose(A)) / norm(A)
        println("MATRIX SYMMETRY tilt_x=$(fc.tilt_x) tilt_y=$(fc.θ) relative=$defect")
        solver = PardisoSolver(assume_symmetric_maxwell=true)
        if fc.tilt_x == 0 && fc.θ == 0
            @test defect < 1e-12
            @test DEO.checked_maxwell_mtype(solver, A) == 6
        else
            @test defect > 1e-10
            @test_throws ArgumentError DEO.checked_maxwell_mtype(solver, A)
        end
    end
end

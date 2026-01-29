
using DistributedEmitterOpt
using LinearAlgebra
using Random
using Test

include("test/gradient_tests.jl")

println("\n=== 2D Foundry Mode Debug ===")
prob = build_test_problem(foundry_mode=true)
p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
rel_err, _ = test_gradient(prob, p0)

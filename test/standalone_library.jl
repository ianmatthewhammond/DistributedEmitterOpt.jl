# Whether the optional NLopt-MMA-CCSA library (v0.1.0 or later) is usable: found, and
# solving the CCSA dual with MMA. Probed in a subprocess so this test process never
# loads it beside NLopt.jl (both export `nlopt_*` symbols). Point
# JULIA_NLOPT_MMACCSA_LIB at a build of
# https://github.com/ianmatthewhammond/NLopt-MMA-CCSA to run the standalone tests.
if !@isdefined(STANDALONE_LIBRARY_READY)
    const STANDALONE_LIBRARY_READY = let
        probe = """
            using DistributedEmitterOpt
            const MB = DistributedEmitterOpt.MmaccsaBackend
            exit(MB.library_available() && MB.dual_algorithm_supported(:mma) ? 0 : 1)
            """
        cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) -e $probe`
        ready = success(pipeline(cmd; stdout=devnull, stderr=devnull))
        ready || @info "Skipping the standalone-backend tests: no NLopt-MMA-CCSA build with the MMA dual found. Set JULIA_NLOPT_MMACCSA_LIB to run them."
        ready
    end
end

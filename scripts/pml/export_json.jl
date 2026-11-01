# Dump the validation JLD2 files to one JSON (no JSON package needed). Run: julia --project export_json.jl out.json in1.jld2 ...
using JLD2
js(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
js(x::Symbol) = js(string(x))
js(x::Bool) = x ? "true" : "false"
js(x::Integer) = string(x)
js(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
js(x::Complex) = "{\"re\":" * js(real(x)) * ",\"im\":" * js(imag(x)) * "}"
js(x::Nothing) = "null"
js(x::AbstractDict) = "{" * join([js(string(k)) * ":" * js(v) for (k, v) in x], ",") * "}"
js(x::NamedTuple) = js(Dict(pairs(x)))
js(x::AbstractVector) = "[" * join(js.(x), ",") * "]"
js(x::AbstractMatrix) = js([x[:, j] for j in 1:size(x, 2)])
js(x) = js(string(x))
out = Dict{String,Any}()
for f in ARGS[2:end]
    out[basename(f)] = load(f)["results"]
end
write(ARGS[1], js(out))
println("wrote ", ARGS[1], " (", filesize(ARGS[1]), " bytes)")

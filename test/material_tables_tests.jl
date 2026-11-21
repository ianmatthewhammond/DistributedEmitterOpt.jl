# Every bundled refractive-index table must load, be strictly ordered, and return the
# tabulated n and k at its own wavelengths and the linear interpolation between them.
# Package convention: resolve_index / complex_index return n - i k.

using DistributedEmitterOpt
using Test

const _MAT_DIR = joinpath(pkgdir(DistributedEmitterOpt), "data", "materials")
const _MAT_NAMES = sort!([splitext(f)[1] for f in readdir(_MAT_DIR) if endswith(f, ".txt")])

# Independent reader for the test: rows of (λ[μm], n, k), comma and/or whitespace separated.
function _raw_rows(path)
    rows = NTuple{3,Float64}[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s, r"[,\s]+")
        vals = tryparse.(Float64, parts)
        any(isnothing, vals) && continue  # header
        push!(rows, (vals[1], vals[2], vals[3]))
    end
    sort!(rows; by = first)
    return rows
end

@testset "material tables" begin
    @testset "bundled material tables" begin
        @test length(_MAT_NAMES) >= 12

        for name in _MAT_NAMES
            @testset "$name" begin
                rows = _raw_rows(joinpath(_MAT_DIR, "$name.txt"))
                @test length(rows) >= 2
                @test all(i -> rows[i][1] < rows[i+1][1], 1:length(rows)-1)

                # The parser's own output, compared directly (not the sorted test rows).
                λp, np_, kp = DistributedEmitterOpt.read_index_table(joinpath(_MAT_DIR, "$name.txt"))
                @test all(>(0), diff(λp))
                @test λp ≈ first.(rows) .* 1e3 rtol = 1e-14
                @test np_ ≈ getindex.(rows, 2) rtol = 1e-14
                @test kp ≈ getindex.(rows, 3) rtol = 1e-14

                n_i, k_i = refindex(name)

                # Tabulated wavelength: exact n and k. Interpolation between rows i and i+1
                # (a valid two-row table has only i = 1).
                i = clamp(length(rows) ÷ 2, 1, length(rows) - 1)
                λ, n, k = rows[i]
                @test n_i(λ * 1e3) ≈ n rtol = 1e-14
                @test k_i(λ * 1e3) ≈ k rtol = 1e-14

                λ1, n1, k1 = rows[i]
                λ2, n2, k2 = rows[i+1]
                t = 0.3
                λm = (λ1 + t * (λ2 - λ1)) * 1e3
                @test n_i(λm) ≈ n1 + t * (n2 - n1) rtol = 1e-9
                @test k_i(λm) ≈ k1 + t * (k2 - k1) rtol = 1e-9

                # n - ik convention through the public lookup.
                @test complex_index(name, λ * 1e3) ≈ complex(n, -k) rtol = 1e-14
            end
        end

        @testset "Si through resolve_index" begin
            rows = _raw_rows(joinpath(_MAT_DIR, "Si.txt"))
            λ, n, k = rows[length(rows) ÷ 2]
            saved = copy(DistributedEmitterOpt._MATERIAL_CACHE)
            try
                delete!(DistributedEmitterOpt._MATERIAL_CACHE, "Si")   # a cold load through the public lookup
                @test resolve_index("Si", λ * 1e3) ≈ complex(n, -k) rtol = 1e-14
                @test real(resolve_index("Si", λ * 1e3)) > 0
            finally
                empty!(DistributedEmitterOpt._MATERIAL_CACHE); merge!(DistributedEmitterOpt._MATERIAL_CACHE, saved)
            end
        end
    end

    @testset "material table parser" begin
        read_table = DistributedEmitterOpt.read_index_table

        mktempdir() do dir
            write_tbl(name, text) = (p = joinpath(dir, name); write(p, text); p)

            # Comma, whitespace, and mixed separators, with comment, blank and CRLF lines.
            p = write_tbl("ok.txt", "# comment\r\nlambda n k\r\n\r\n0.5, 1.0, 2.0\r\n0.6 1.5 2.5\r\n0.7,\t1.75 ,3.0\r\n")
            λ, n, k = read_table(p)
            @test λ ≈ [500.0, 600.0, 700.0]
            @test n == [1.0, 1.5, 1.75]
            @test k == [2.0, 2.5, 3.0]

            # Descending tables are returned ascending.
            λ, n, _ = read_table(write_tbl("desc.txt", "lambda n k\n0.7 3 0\n0.6 2 0\n0.5 1 0\n"))
            @test λ ≈ [500.0, 600.0, 700.0]
            @test n == [1.0, 2.0, 3.0]

            # regression: every malformed table raises an ErrorException naming the file and the
            # source line of the offending row (blank and comment lines count towards line numbers).
            cases = (
                ("badtoken.txt",  "lambda n k\n0.5 1.0 2.0\n0.6 1.5 x\n0.7 1.75 3.0\n", 3),
                ("short.txt",     "0.5 1.0 2.0\n0.6 1.5\n0.7 1.75 3.0\n", 2),
                ("extra.txt",     "0.5 1 2\n0.6 1 2 3\n0.7 1 2\n", 2),
                ("latehdr.txt",   "0.5 1 2\nlambda n k\n0.6 1 2\n", 2),
                ("twohdr.txt",    "lambda n k\nlambda n k\n0.5 1 2\n0.6 1 2\n", 2),
                ("emptyfirst.txt", ",8,9\n0.5 1 2\n0.7 3 4\n", 1),
                ("wordfirst.txt", "oops,8,9\n0.5 1 2\n0.7 3 4\n", 1),
                ("emptyhdr.txt",  ",,\n0.5 1 2\n0.7 3 4\n", 1),
                ("numhdr.txt",    "0.4 n k\n0.5 1 2\n0.7 3 4\n", 1),
                ("twofieldhdr.txt", "lambda n\n0.5 1 2\n0.7 3 4\n", 1),
                ("nan.txt",       "0.5 1 2\n0.6 NaN 2\n0.7 1 2\n", 2),
                ("inf.txt",       "0.5 1 2\n0.6 1 Inf\n0.7 1 2\n", 2),
                ("neg.txt",       "# c\n0.5 1 2\n\n-0.6 1 2\n0.7 1 2\n", 4),
                ("zero.txt",      "0.5 1 2\n0.0 1 2\n0.7 1 2\n", 2),
                ("overflow.txt",  "0.5 1 2\n1e306 3 4\n", 2),
                ("dup.txt",       "0.5 1 2\n0.6 1 2\n0.6 1 2\n0.7 1 2\n", 3),
                ("unsorted.txt",  "0.5 1 2\n\n0.7 1 2\n0.6 1 2\n", 4),
                ("revdup.txt",    "0.7 1 2\n0.6 1 2\n0.6 1 2\n", 3),
                ("descasc.txt",   "0.7 1 2\n0.6 1 2\n0.65 1 2\n", 3),
                # the order is checked after every row is read: the unreadable row is reported
                ("dupthenbad.txt", "0.5 1 2\n0.5 1 2\n0.6 x 2\n", 3),
                # a field is a number by its syntax, whatever its size
                ("bigfirst.txt",  "1e309,1e309,1e309\n0.5 1 2\n0.7 3 4\n", 1),
                ("bignamehdr.txt", "1e309,n,k\n0.5 1 2\n0.7 3 4\n", 1),
                ("bigk.txt",      "0.5 1 2\n0.6 1 1e309\n0.7 1 2\n", 2),
                # hexadecimal floating literals count as numbers too, whatever their size
                ("hexfirst.txt",  "0x1p1024,0x1p1024,0x1p1024\n0.5 1 2\n0.7 3 4\n", 1),
                ("hexnamehdr.txt", "0x1p1024,n,k\n0.5 1 2\n0.7 3 4\n", 1),
                # all wavelengths equal: no direction
                ("allequal.txt",  "0.5 1 2\n0.5 3 4\n", 2),
            )
            for (nm, text, line) in cases
                err = try read_table(write_tbl(nm, text)); nothing catch e; e end
                @test err isa ErrorException
                @test err isa ErrorException && occursin(nm, err.msg) && occursin("line $line", err.msg)
            end
            for (nm, text) in (("overflow2.txt", "0.5 1 2\n1e306 3 4\n"), ("bigfirst2.txt", "1e309,1e309,1e309\n0.5 1 2\n0.7 3 4\n"),
                               ("hexfirst2.txt", "0x1p1024,0x1p1024,0x1p1024\n0.5 1 2\n0.7 3 4\n"))
                err = try read_table(write_tbl(nm, text)); nothing catch e; e end
                @test err isa ErrorException && occursin("finite", err.msg)
            end

            # Row-count errors name the file (no line) and the count.
            for (nm, text) in (("one.txt", "lambda n k\n0.5 1 2\n"), ("none.txt", "lambda n k\n# c\n"))
                err = try read_table(write_tbl(nm, text)); nothing catch e; e end
                @test err isa ErrorException && occursin(nm, err.msg) && occursin("at least 2", err.msg)
            end

            # regression: a first row containing an empty field or any numeric field is not taken for a
            # header, including one of numbers too large for Float64, decimal or hexadecimal: no lookup
            # finds a truncated table.
            D = DistributedEmitterOpt
            saved = copy(D._MATERIAL_CACHE)
            try
                for (file, text) in (("damaged.txt", ",8,9\n0.5 1 2\n0.7 3 4\n"),
                                     ("damagedbig.txt", "1e309,1e309,1e309\n0.5 1 2\n0.7 3 4\n"),
                                     ("damagedbighdr.txt", "1e309,n,k\n0.5 1 2\n0.7 3 4\n"),
                                     ("damagedhex.txt", "0x1p1024,0x1p1024,0x1p1024\n0.5 1 2\n0.7 3 4\n"),
                                     ("damagedhexhdr.txt", "0x1p1024,n,k\n0.5 1 2\n0.7 3 4\n"))
                    write_tbl(file, text)
                    name = relpath(joinpath(dir, splitext(file)[1]), _MAT_DIR)
                    D.clear_material_cache!()
                    for call in (() -> refindex(name), () -> complex_index(name, 600.0), () -> resolve_index(name, 600.0))
                        err = try call(); nothing catch e; e end
                        @test err isa ErrorException && occursin(file, err.msg) && occursin("line 1", err.msg)
                    end
                    @test isempty(D._MATERIAL_CACHE)
                end
            finally
                empty!(D._MATERIAL_CACHE); merge!(D._MATERIAL_CACHE, saved)
            end

            # Accepted: a header of three names, comments, blanks, an inline-free numeric row.
            λ, _, _ = read_table(write_tbl("hdr3.txt", "Wavelength, n, k\n0.5 1 2\n0.6 1 2\n"))
            @test λ ≈ [500.0, 600.0]
            # regression: a field Float64 parses but the literal patterns do not match (NaN) is a number,
            # so a first row "NaN,n,k" is a malformed row, not a header.
            err = try read_table(write_tbl("nanhdr.txt", "NaN,n,k\n0.5 1 2\n0.7 3 4\n")); nothing catch e; e end
            @test err isa ErrorException && occursin("nanhdr.txt", err.msg) && occursin("line 1", err.msg)
            # regression: a literal too small for Float64 is stored as its converted value (here k = 0), not refused.
            _, _, k = read_table(write_tbl("tinyk.txt", "0.5 1 1e-400\n0.7 3 4\n"))
            @test k == [0.0, 4.0]
            # Representable hexadecimal floating literals are data (0x1p-1 = 0.5, 0x1.4p-1 = 0.625).
            λ, n, k = read_table(write_tbl("hexok.txt", "lambda n k\n0x1p-1 1 2\n0x1.4p-1 0x1.8p1 4\n"))
            @test λ ≈ [500.0, 625.0]
            @test n == [1.0, 3.0] && k == [2.0, 4.0]
        end
    end

    # regression: an uncached malformed table is refused by all four lookups and preserves existing cache entries.
    @testset "an uncached malformed table is refused by all four lookups and preserves existing cache entries" begin
        D = DistributedEmitterOpt
        saved = copy(D._MATERIAL_CACHE)
        try
            mktempdir() do dir
                write(joinpath(dir, "broken.txt"), "lambda n k\n0.5 1.0 2.0\n0.6 1.5 x\n0.7 1.75 3.0\n")
                name = relpath(joinpath(dir, "broken"), _MAT_DIR)   # the lookup builds data/materials/<name>.txt
                D.clear_material_cache!()
                primed = D.get_interp("Ag")
                @test collect(keys(D._MATERIAL_CACHE)) == ["Ag"]
                calls = (() -> refindex(name), () -> complex_index(name, 600.0), () -> resolve_index(name, 600.0),
                         () -> D.get_interp(name))
                for call in calls
                    err = try
                        call()
                        nothing
                    catch e
                        e
                    end
                    @test err isa ErrorException
                    @test err isa ErrorException && occursin("broken.txt", err.msg) && occursin("line 3", err.msg)
                end
                @test collect(keys(D._MATERIAL_CACHE)) == ["Ag"]
                @test D._MATERIAL_CACHE["Ag"] === primed
            end
        finally
            empty!(D._MATERIAL_CACHE); merge!(D._MATERIAL_CACHE, saved)
        end
    end
end

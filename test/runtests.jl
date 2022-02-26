using TypedMissings, Test, OffsetArrays

@testset "nonmissingtype" begin
    @test nonmissingtype(Union{Int, TypedMissing}) == Int
    @test nonmissingtype(Union{Rational, TypedMissing}) == Rational
    @test nonmissingtype(Union{Int, TypedMissing, Missing}) == Int
    @test nonmissingtype(Union{Rational, TypedMissing, Missing}) == Rational
    @test nonmissingtype(Any) == Any
    @test nonmissingtype(TypedMissing) == Union{}
    @test nonmissingtype(Union{TypedMissing, Missing}) == Union{}
end

@testset "broadcast" begin
    @test Base.broadcastable(TypedMissing(:a)) isa Ref{TypedMissing}
    @test isequal(TypedMissing(:a) .+ [1, 2], fill(TypedMissing(:a), 2))
end

@testset "convert" begin
    @test convert(TypedMissing, TypedMissing(:a)) === TypedMissing(:a)
    @test convert(TypedMissing, missing) === TypedMissing()
    # TODO: should this (and similar) be allowed?
    @test convert(Missing, TypedMissing(:a)) === missing
    @test convert(Union{Int, TypedMissing}, TypedMissing(:a)) === TypedMissing(:a)
    @test convert(Union{Int, Missing}, TypedMissing(:a)) === missing
    @test convert(Union{Int, Missing}, TypedMissing(:a)) === missing
    @test convert(Union{Int, TypedMissing}, 1) === 1
    @test convert(Union{Int, TypedMissing}, 1.0) === 1
    @test convert(Union{Nothing, TypedMissing}, TypedMissing(:a)) === TypedMissing(:a)
    @test convert(Union{Nothing, Missing}, TypedMissing(:a)) === missing
    @test convert(Union{Nothing, TypedMissing}, missing) === TypedMissing()
    @test convert(Union{Nothing, TypedMissing}, nothing) === nothing
    @test convert(Union{Nothing, TypedMissing, Missing}, missing) === missing
    @test convert(Union{Nothing, TypedMissing}, TypedMissing(:a)) === TypedMissing(:a)
    @test convert(Union{Nothing, TypedMissing, Missing}, TypedMissing(:a)) === TypedMissing(:a)
    @test convert(Union{TypedMissing, Nothing, Float64}, 1) === 1.0
    @test convert(Union{TypedMissing, Missing, Float64}, 1) === 1.0
    @test convert(Union{TypedMissing, Nothing, Missing, Float64}, 1) === 1.0

    @test_throws MethodError convert(TypedMissing, 1)
    @test_throws MethodError convert(Union{Nothing, TypedMissing}, 1)
    @test_throws MethodError convert(Union{Int, TypedMissing}, "a")
end

@testset "promote rules" begin
    @test promote_type(TypedMissing, TypedMissing) == TypedMissing
    @test promote_type(TypedMissing, Int) == Union{TypedMissing, Int}
    @test promote_type(Int, TypedMissing) == Union{TypedMissing, Int}
    @test promote_type(Int, Any) == Any
    @test promote_type(Any, Any) == Any
    @test promote_type(TypedMissing, Any) == Any
    @test promote_type(Any, TypedMissing) == Any
    @test promote_type(Union{Int, TypedMissing}, TypedMissing) == Union{Int, TypedMissing}
    @test promote_type(TypedMissing, Union{Int, TypedMissing}) == Union{Int, TypedMissing}
    @test promote_type(Union{Int, TypedMissing}, Int) == Union{Int, TypedMissing}
    @test promote_type(Int, Union{Int, TypedMissing}) == Union{Int, TypedMissing}
    @test promote_type(Any, Union{Int, TypedMissing}) == Any
    @test promote_type(Union{Nothing, TypedMissing}, Any) == Any
    @test promote_type(Union{Int, TypedMissing}, Union{Int, TypedMissing}) == Union{Int, TypedMissing}
    @test promote_type(Union{Float64, TypedMissing}, Union{String, TypedMissing}) == Any
    @test promote_type(Union{Float64, TypedMissing}, Union{Int, TypedMissing}) == Union{Float64, TypedMissing}
    @test promote_type(Union{Nothing, TypedMissing, Int}, Float64) == Union{Nothing, Float64, TypedMissing}

    @test promote_type(TypedMissing, Missing) == TypedMissing
    @test promote_type(Union{Float64, TypedMissing}, Missing) == Union{Float64, TypedMissing}
    @test promote_type(Union{Float64, Missing}, TypedMissing) == Union{Float64, TypedMissing}
    @test promote_type(Union{Float64, TypedMissing, Nothing}, Missing) == Union{Float64, Nothing, TypedMissing}
    @test promote_type(Union{Float64, Missing, Nothing}, TypedMissing) == Union{Float64, Nothing, TypedMissing}
end

@testset "promotion in various contexts" begin
    @test collect(v for v in (1, TypedMissing(:a))) isa Vector{Union{Int,TypedMissing}}
    @test map(identity, Any[1, TypedMissing(:a)]) isa Vector{Union{Int,TypedMissing}}
    @test broadcast(identity, Any[1, TypedMissing(:a)]) isa Vector{Union{Int,TypedMissing}}
    @test unique((1, TypedMissing(:a))) isa Vector{Union{Int,TypedMissing}}

    @test map(ismissing, Any[1, TypedMissing(:a)]) isa Vector{Bool}
    @test broadcast(ismissing, Any[1, TypedMissing(:a)]) isa BitVector
end

@testset "Array convenience constructors" begin
    @test isequal(Vector{Union{Int,TypedMissing}}(TypedMissing(:a), 2),
                  [TypedMissing(:a), TypedMissing(:a)])
    @test isequal(Matrix{Union{Int,TypedMissing}}(TypedMissing(:a), 2, 2),
                  [TypedMissing(:a) TypedMissing(:a);
                  TypedMissing(:a) TypedMissing(:a)])
    @test isequal(Array{Union{Int,TypedMissing}, 1}(TypedMissing(:a), 2),
                  [TypedMissing(:a), TypedMissing(:a)])
    @test isequal(Array{Union{Int,TypedMissing}, 2}(TypedMissing(:a), 2, 2),
                  [TypedMissing(:a) TypedMissing(:a);
                  TypedMissing(:a) TypedMissing(:a)])
    @test isequal(Array{Union{Int,TypedMissing}}(TypedMissing(:a), 2),
                  [TypedMissing(:a), TypedMissing(:a)])
    @test isequal(Array{Union{Int,TypedMissing}}(TypedMissing(:a), 2, 2),
                  [TypedMissing(:a) TypedMissing(:a);
                  TypedMissing(:a) TypedMissing(:a)])
end

@testset "comparison operators" begin
    @test (TypedMissing(:a) == TypedMissing(:a)) === TypedMissing(:a)
    @test (1 == TypedMissing(:a)) === TypedMissing(:a)
    @test (TypedMissing(:a) == 1) === TypedMissing(:a)
    @test (TypedMissing(:a) != TypedMissing(:a)) === TypedMissing(:a)
    @test (1 != TypedMissing(:a)) === TypedMissing(:a)
    @test (TypedMissing(:a) != 1) === TypedMissing(:a)
    @test isequal(TypedMissing(:a), TypedMissing(:a))
    @test !isequal(1, TypedMissing(:a))
    @test !isequal(TypedMissing(:a), 1)
    @test (TypedMissing(:a) < TypedMissing(:a)) === TypedMissing(:a)
    @test (TypedMissing(:a) < 1) === TypedMissing(:a)
    @test (1 < TypedMissing(:a)) === TypedMissing(:a)
    @test (TypedMissing(:a) <= TypedMissing(:a)) === TypedMissing(:a)
    @test (TypedMissing(:a) <= 1) === TypedMissing(:a)
    @test (1 <= TypedMissing(:a)) === TypedMissing(:a)
    @test !isless(TypedMissing(:a), TypedMissing(:a))
    @test !isless(TypedMissing(:a), 1)
    @test isless(1, TypedMissing(:a))
    @test (TypedMissing(:a) ≈ TypedMissing(:a)) === TypedMissing(:a)
    @test isapprox(TypedMissing(:a), 1.0, atol=1e-6) === TypedMissing(:a)
    @test isapprox(1.0, TypedMissing(:a), rtol=1e-6) === TypedMissing(:a)

    @test (TypedMissing(:a) == TypedMissing(:b)) === TypedMissing()
    @test (TypedMissing(:a) != TypedMissing(:b)) === TypedMissing()
    @test !isequal(TypedMissing(:a), TypedMissing(:b))
    @test (TypedMissing(:a) < TypedMissing(:b)) === TypedMissing()
    @test (TypedMissing(:a) <= TypedMissing(:b)) === TypedMissing()
    @test isless(TypedMissing(:a), TypedMissing(:b))
    @test (TypedMissing(:a) ≈ TypedMissing(:b)) === TypedMissing()

    @test (TypedMissing(:a) == missing) === TypedMissing()
    @test (TypedMissing(:a) != missing) === TypedMissing()
    @test !isequal(TypedMissing(:a), missing)
    @test (TypedMissing(:a) < missing) === TypedMissing()
    @test (TypedMissing(:a) <= missing) === TypedMissing()
    @test isless(missing, TypedMissing(:a))
    @test (TypedMissing(:a) ≈ missing) === TypedMissing()
    @test (missing ≈ TypedMissing(:a)) === TypedMissing()

    @test isequal(missing, TypedMissing())
    @test isequal(TypedMissing(), missing)
    @test !isless(TypedMissing(), missing)
    @test !isless(missing, TypedMissing())
    @test isless(TypedMissing(), TypedMissing(:a))

    @test !any(T -> T === Union{TypedMissing,Bool}, Base.return_types(isequal, Tuple{Any,Any}))
end

@testset "arithmetic operators" begin
    arithmetic_operators = [+, -, *, /, ^, Base.div, Base.mod, Base.fld, Base.rem]

    # All unary operators return missing of the same kind when evaluating missing
    for f in [!, ~, +, -, *, &, |, xor, nand, nor]
        @test f(TypedMissing(:a)) === TypedMissing(:a)
    end

    # All arithmetic operators return missing when operating on two missing's
    # All arithmetic operators return missing when operating on a scalar and an missing
    # All arithmetic operators return missing when operating on an missing and a scalar
    # The kind depends on whether all missings have the same kind
    for f in arithmetic_operators
        @test f(TypedMissing(:a), TypedMissing(:a)) === TypedMissing(:a)
        @test f(TypedMissing(:a), TypedMissing(:b)) === TypedMissing()
        @test f(TypedMissing(:a), missing) === TypedMissing()
        @test f(missing, TypedMissing(:a)) === TypedMissing()
        @test f(1, TypedMissing(:a)) === TypedMissing(:a)
        @test f(TypedMissing(:a), 1) === TypedMissing(:a)
    end

    @test min(TypedMissing(:a), TypedMissing(:a)) === TypedMissing(:a)
    @test max(TypedMissing(:a), TypedMissing(:a)) === TypedMissing(:a)
    @test min(TypedMissing(:a), TypedMissing(:b)) === TypedMissing()
    @test max(TypedMissing(:a), TypedMissing(:b)) === TypedMissing()
    @test min(TypedMissing(:a), missing) === TypedMissing()
    @test max(TypedMissing(:a), missing) === TypedMissing()
    @test min(missing, TypedMissing(:a)) === TypedMissing()
    @test max(missing, TypedMissing(:a)) === TypedMissing()
    for f in [min, max]
        for arg in ["", "a", 1, -1.0, [2]]
            @test f(TypedMissing(:a), arg) === TypedMissing(:a)
            @test f(arg, TypedMissing(:a)) === TypedMissing(:a)
        end
    end
end

@testset "two-argument functions" begin
    two_argument_functions = [atan, hypot, log]

    # All two-argument functions return missing when operating on two missing's
    # All two-argument functions return missing when operating on a scalar and an missing
    # All two-argument functions return missing when operating on an missing and a scalar
    # The kind depends on whether all missings have the same kind
    for f in two_argument_functions
        @test f(TypedMissing(:a), TypedMissing(:a)) === TypedMissing(:a)
        @test f(TypedMissing(:a), TypedMissing(:b)) === TypedMissing()
        @test f(TypedMissing(:a), missing) === TypedMissing()
        @test f(missing, TypedMissing(:a)) === TypedMissing()
        @test f(1, TypedMissing(:a)) === TypedMissing(:a)
        @test f(TypedMissing(:a), 1) === TypedMissing(:a)
    end
end

@testset "bit operators" begin
    bit_operators = [&, |, ⊻]

    # All bit operators return missing when operating on two missing's
    for f in bit_operators
        @test f(TypedMissing(:a), TypedMissing(:a)) === TypedMissing(:a)
        @test f(TypedMissing(:a), TypedMissing(:b)) === TypedMissing()
        @test f(TypedMissing(:a), missing) === TypedMissing()
        @test f(missing, TypedMissing(:a)) === TypedMissing()    end
end

@testset "boolean operators" begin
    @test (TypedMissing(:a) & true) === TypedMissing(:a)
    @test (true & TypedMissing(:a)) === TypedMissing(:a)
    @test !(TypedMissing(:a) & false)
    @test !(false & TypedMissing(:a))
    @test (TypedMissing(:a) | false) === TypedMissing(:a)
    @test (false | TypedMissing(:a)) === TypedMissing(:a)
    @test TypedMissing(:a) | true
    @test true | TypedMissing(:a)
    @test (xor(TypedMissing(:a), true)) === TypedMissing(:a)
    @test (xor(true, TypedMissing(:a))) === TypedMissing(:a)
    @test (xor(TypedMissing(:a), false)) === TypedMissing(:a)
    @test (xor(false, TypedMissing(:a))) === TypedMissing(:a)
    @test (nand(TypedMissing(:a), true)) === TypedMissing(:a)
    @test (nand(true, TypedMissing(:a))) === TypedMissing(:a)
    @test nand(TypedMissing(:a), false)
    @test nand(false, TypedMissing(:a))
    @test (⊼(TypedMissing(:a), true)) === TypedMissing(:a)
    @test (⊼(true, TypedMissing(:a))) === TypedMissing(:a)
    @test ⊼(TypedMissing(:a), false)
    @test ⊼(false, TypedMissing(:a))
    @test !nor(TypedMissing(:a), true)
    @test !nor(true, TypedMissing(:a))
    @test (nor(TypedMissing(:a), false)) === TypedMissing(:a)
    @test (nor(false, TypedMissing(:a))) === TypedMissing(:a)
    @test !⊽(TypedMissing(:a), true)
    @test !⊽(true, TypedMissing(:a))
    @test (⊽(TypedMissing(:a), false)) === TypedMissing(:a)
    @test (⊽(false, TypedMissing(:a))) === TypedMissing(:a)

    @test (TypedMissing(:a) & 1) === TypedMissing(:a)
    @test (1 & TypedMissing(:a)) === TypedMissing(:a)
    @test (TypedMissing(:a) | 1) === TypedMissing(:a)
    @test (1 | TypedMissing(:a)) === TypedMissing(:a)
    @test (xor(TypedMissing(:a), 1)) === TypedMissing(:a)
    @test (xor(1, TypedMissing(:a))) === TypedMissing(:a)
    @test (nand(TypedMissing(:a), 1)) === TypedMissing(:a)
    @test (nand(1, TypedMissing(:a))) === TypedMissing(:a)
    @test (⊼(TypedMissing(:a), 1)) === TypedMissing(:a)
    @test (⊼(1, TypedMissing(:a))) === TypedMissing(:a)
    @test (nor(TypedMissing(:a), 1)) === TypedMissing(:a)
    @test (nor(1, TypedMissing(:a))) === TypedMissing(:a)
    @test (⊽(TypedMissing(:a), 1)) === TypedMissing(:a)
    @test (⊽(1, TypedMissing(:a))) === TypedMissing(:a)

    @test (TypedMissing(:a) & TypedMissing(:b)) === TypedMissing()
    @test (TypedMissing(:a) | TypedMissing(:b)) === TypedMissing()
    @test (xor(TypedMissing(:a), TypedMissing(:b))) === TypedMissing()
    @test (nand(TypedMissing(:a), TypedMissing(:b))) === TypedMissing()
    @test (⊼(TypedMissing(:a), TypedMissing(:b))) === TypedMissing()
    @test (nor(TypedMissing(:a), TypedMissing(:b))) === TypedMissing()
    @test (⊽(TypedMissing(:a), TypedMissing(:b))) === TypedMissing()

    @test (TypedMissing(:a) & missing) === TypedMissing()
    @test (missing & TypedMissing(:a)) === TypedMissing()
    @test (TypedMissing(:a) | missing) === TypedMissing()
    @test (missing | TypedMissing(:a)) === TypedMissing()
    @test (xor(TypedMissing(:a), missing)) === TypedMissing()
    @test (xor(missing, TypedMissing(:a))) === TypedMissing()
    @test (nand(TypedMissing(:a), missing)) === TypedMissing()
    @test (nand(missing, TypedMissing(:a))) === TypedMissing()
    @test (⊼(TypedMissing(:a), missing)) === TypedMissing()
    @test (⊼(missing, TypedMissing(:a))) === TypedMissing()
    @test (nor(TypedMissing(:a), missing)) === TypedMissing()
    @test (nor(missing, TypedMissing(:a))) === TypedMissing()
    @test (⊽(TypedMissing(:a), missing)) === TypedMissing()
    @test (⊽(missing, TypedMissing(:a))) === TypedMissing()
end

@testset "* string/char concatenation" begin
    @test "a" * TypedMissing(:a) === TypedMissing(:a)
    @test 'a' * TypedMissing(:a) === TypedMissing(:a)
    @test TypedMissing(:a) === TypedMissing(:a) * "a"
    @test TypedMissing(:a) === TypedMissing(:a) * 'a'
end

# Emulate a unitful type such as Dates.Minute
struct Unit
    value::Int
end
Base.zero(::Type{Unit}) = Unit(0)
Base.one(::Type{Unit}) = 1

@testset "elementary functions" begin
    elementary_functions = [abs, abs2, sign, real, imag,
                            acos, acosh, asin, asinh, atan, atanh, sin, sinh,
                            conj, cos, cosh, tan, tanh,
                            exp, exp2, expm1, log, log10, log1p, log2,
                            exponent, sqrt,
                            identity, zero, one, oneunit,
                            iseven, isodd, ispow2,
                            isfinite, isinf, isnan, iszero,
                            isinteger, isreal, transpose, adjoint, float, complex, inv]

    # All elementary functions return missing when evaluating missing
    for f in elementary_functions
        @test f(TypedMissing(:a)) === TypedMissing(:a)
    end

    @test clamp(TypedMissing(:a), 1, 2) === TypedMissing(:a)

    for T in (Int, Float64)
        @test zero(Union{T, TypedMissing}) === T(0)
        @test one(Union{T, TypedMissing}) === T(1)
        @test oneunit(Union{T, TypedMissing}) === T(1)
        @test float(Union{T, TypedMissing}) === Union{float(T), TypedMissing}
        @test complex(Union{T, TypedMissing}) === Union{complex(T), TypedMissing}
    end

    @test_throws MethodError zero(Union{Symbol, TypedMissing})
    @test_throws MethodError one(Union{Symbol, TypedMissing})
    @test_throws MethodError oneunit(Union{Symbol, TypedMissing})
    @test_throws MethodError float(Union{Symbol, TypedMissing})
    @test_throws MethodError complex(Union{Symbol, TypedMissing})

    for T in (Unit,)
        @test zero(Union{T, TypedMissing}) === T(0)
        @test one(Union{T, TypedMissing}) === 1
        @test oneunit(Union{T, TypedMissing}) === T(1)
    end

    @test zero(TypedMissing) === TypedMissing()
    @test one(TypedMissing) === TypedMissing()
    @test oneunit(TypedMissing) === TypedMissing()
    @test float(TypedMissing) === TypedMissing
    @test complex(TypedMissing) === TypedMissing

    @test_throws MethodError zero(Any)
    @test_throws MethodError one(Any)
    @test_throws MethodError oneunit(Any)
    @test_throws MethodError float(Any)
    @test_throws MethodError complex(Any)

    @test_throws MethodError zero(String)
    @test_throws MethodError zero(Union{String, TypedMissing})
end

@testset "rounding functions" begin
    # All rounding functions return missing when evaluating missing as first argument

    # Check that the RoundingMode argument is passed on correctly
    @test round(Union{Int, TypedMissing}, 0.9) === round(Int, 0.9)
    @test round(Union{Int, TypedMissing}, 0.9, RoundToZero) ===
        round(Int, 0.9, RoundToZero)

    # Test elementwise on mixed arrays to ensure signature of TypedMissing methods
    # matches that of Float methods
    test_array = [1.0, TypedMissing(:a)]

    @test isequal(round.(test_array, RoundNearest), test_array)
    @test isequal(round.(Union{Int, TypedMissing}, test_array, RoundNearest), test_array)

    rounding_functions = [ceil, floor, round, trunc]
    for f in rounding_functions
        @test_throws MissingException f(Int, TypedMissing(:a))
        @test isequal(f.(test_array), test_array)
        @test isequal(f.(test_array, digits=0, base=10), test_array)
        @test isequal(f.(test_array, sigdigits=1, base=10), test_array)
        @test isequal(f.(Union{Int, TypedMissing}, test_array), test_array)
    end
end

@testset "printing" begin
    @test sprint(show, TypedMissing(:a)) == "TypedMissing(:a)"
    @test sprint(show, TypedMissing(:a), context=:compact => true) == "TypedMissing(:a)"
    @test sprint(show, TypedMissing()) == "TypedMissing()"
    @test sprint(show, TypedMissing(), context=:compact => true) == "TypedMissing()"

    @test sprint(show, [TypedMissing(:a)]) == "TypedMissing[TypedMissing(:a)]"
    @test sprint(show, [1 TypedMissing(:a)]) == "$(Union{Int, TypedMissing})[1 TypedMissing(:a)]"
    @test sprint(show, [TypedMissing()]) == "TypedMissing[TypedMissing()]"
    @test sprint(show, [1 TypedMissing()]) == "$(Union{Int, TypedMissing})[1 TypedMissing()]"

    b = IOBuffer()
    display(TextDisplay(b), [TypedMissing(:a)])
    @test String(take!(b)) == "1-element Vector{TypedMissing}:\n TypedMissing(:a)"
    b = IOBuffer()
    display(TextDisplay(b), [1 TypedMissing(:a)])
    @test String(take!(b)) == "1×2 Matrix{Union{Int64, TypedMissing}}:\n 1  TypedMissing(:a)"
    b = IOBuffer()
    display(TextDisplay(b), [TypedMissing()])
    @test String(take!(b)) == "1-element Vector{TypedMissing}:\n TypedMissing()"
    b = IOBuffer()
    display(TextDisplay(b), [1 TypedMissing()])
    @test String(take!(b)) == "1×2 Matrix{Union{Int64, TypedMissing}}:\n 1  TypedMissing()"
end

@testset "arrays with missing values" begin
    x = convert(Vector{Union{Int, TypedMissing}}, [1.0, TypedMissing(:a)])
    @test isa(x, Vector{Union{Int, TypedMissing}})
    @test isequal(x, [1, TypedMissing(:a)])
    x = convert(Vector{Union{Int, Missing}}, [1.0, TypedMissing(:a)])
    @test isa(x, Vector{Union{Int, Missing}})
    @test isequal(x, [1, missing])
    x = convert(Vector{Union{Int, TypedMissing}}, [1.0, missing])
    @test isa(x, Vector{Union{Int, TypedMissing}})
    @test isequal(x, [1, missing])
    x = convert(Vector{Union{Int, TypedMissing}}, [1.0])
    @test isa(x, Vector{Union{Int, TypedMissing}})
    @test x == [1]
    x = convert(Vector{Union{Int, TypedMissing}}, [TypedMissing(:a)])
    @test isa(x, Vector{Union{Int, TypedMissing}})
    @test isequal(x, [TypedMissing(:a)])
    x = convert(Vector{Union{Int, TypedMissing}}, [missing])
    @test isa(x, Vector{Union{Int, TypedMissing}})
    @test isequal(x, [TypedMissing()])

    @test isequal(adjoint([1, TypedMissing(:a)]), [1 TypedMissing(:a)])
    @test eltype(adjoint([1, TypedMissing(:a)])) == Union{Int, TypedMissing}
    # issue JuliaLang/julia#32777
    let a = [0, nothing, 0.0, TypedMissing(:a)]
        @test a[1] === 0.0
        @test a[2] === nothing
        @test a[3] === 0.0
        @test a[4] === TypedMissing(:a)
        @test a isa Vector{Union{TypedMissing, Nothing, Float64}}
    end
end

@testset "== and != on arrays" begin
    @test ismissing([1, TypedMissing(:a)] == [1, TypedMissing(:a)])
    @test ismissing(["a", TypedMissing(:a)] == ["a", TypedMissing(:a)])
    @test ismissing(Any[1, TypedMissing(:a)] == Any[1, TypedMissing(:a)])
    @test ismissing(Any[TypedMissing(:a)] == Any[TypedMissing(:a)])
    @test ismissing([TypedMissing(:a)] == [TypedMissing(:a)])
    @test ismissing(Any[TypedMissing(:a), 2] == Any[1, TypedMissing(:a)])
    @test ismissing([TypedMissing(:a), false] == BitArray([true, false]))
    @test ismissing(Any[TypedMissing(:a), false] == BitArray([true, false]))

    @test_broken ([1, TypedMissing(:a)] == [1, TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (["a", TypedMissing(:a)] == ["a", TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (Any[1, TypedMissing(:a)] == Any[1, TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (Any[TypedMissing(:a)] == Any[TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken ([TypedMissing(:a)] == [TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (Any[TypedMissing(:a), 2] == Any[1, TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken ([TypedMissing(:a), false] == BitArray([true, false])) === TypedMissing(:a)
    @test_broken (Any[TypedMissing(:a), false] == BitArray([true, false])) === TypedMissing(:a)

    @test ismissing([1, TypedMissing(:a)] == [1, TypedMissing(:b)])
    @test ismissing(["a", TypedMissing(:a)] == ["a", TypedMissing(:b)])
    @test ismissing(Any[1, TypedMissing(:a)] == Any[1, TypedMissing(:b)])
    @test ismissing(Any[TypedMissing(:a)] == Any[TypedMissing(:b)])
    @test ismissing([TypedMissing(:a)] == [TypedMissing(:b)])
    @test ismissing(Any[TypedMissing(:a), 2] == Any[1, TypedMissing(:b)])

    @test_broken ([1, TypedMissing(:a)] == [1, TypedMissing(:b)]) === TypedMissing()
    @test_broken (["a", TypedMissing(:a)] == ["a", TypedMissing(:b)]) === TypedMissing()
    @test_broken (Any[1, TypedMissing(:a)] == Any[1, TypedMissing(:b)]) === TypedMissing()
    @test_broken (Any[TypedMissing(:a)] == Any[TypedMissing(:b)]) === TypedMissing()
    @test_broken ([TypedMissing(:a)] == [TypedMissing(:b)]) === TypedMissing()
    @test_broken (Any[TypedMissing(:a), 2] == Any[1, TypedMissing(:b)]) === TypedMissing()

    @test ([1, TypedMissing(:a)] == [1, missing]) === missing
    @test (["a", TypedMissing(:a)] == ["a", missing]) === missing
    @test (Any[1, TypedMissing(:a)] == Any[1, missing]) === missing
    @test (Any[TypedMissing(:a)] == Any[missing]) === missing
    @test ([TypedMissing(:a)] == [missing]) === missing
    @test (Any[TypedMissing(:a), 2] == Any[1, missing]) === missing
    @test ([TypedMissing(:a), false] == BitArray([true, false])) === missing
    @test (Any[TypedMissing(:a), false] == BitArray([true, false])) === missing

    @test Union{Int, TypedMissing}[1] ==
        Union{Float64, TypedMissing}[1.0] ==
        Union{Float64, Missing}[1.0]
    @test Union{Int, TypedMissing}[1] == [1.0]
    @test Union{Bool, TypedMissing}[true] == BitArray([true])
    @test !([TypedMissing(:a), 1] == [TypedMissing(:a), 2])
    @test !([TypedMissing(:a), 1] == [TypedMissing(), 2])
    @test !([TypedMissing(:a), 1] == [missing, 2])
    @test !(Union{Int, TypedMissing}[1] == [2])
    @test !([1] == Union{Int, TypedMissing}[2])
    @test !(Union{Int, TypedMissing}[1] == Union{Int, TypedMissing}[2])

    @test ismissing([1, TypedMissing(:a)] != [1, TypedMissing(:a)])
    @test ismissing(["a", TypedMissing(:a)] != ["a", TypedMissing(:a)])
    @test ismissing(Any[1, TypedMissing(:a)] != Any[1, TypedMissing(:a)])
    @test ismissing(Any[TypedMissing(:a)] != Any[TypedMissing(:a)])
    @test ismissing([TypedMissing(:a)] != [TypedMissing(:a)])
    @test ismissing(Any[TypedMissing(:a), 2] != Any[1, TypedMissing(:a)])
    @test ismissing([TypedMissing(:a), false] != BitArray([true, false]))
    @test ismissing(Any[TypedMissing(:a), false] != BitArray([true, false]))

    @test_broken ([1, TypedMissing(:a)] != [1, TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (["a", TypedMissing(:a)] != ["a", TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (Any[1, TypedMissing(:a)] != Any[1, TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (Any[TypedMissing(:a)] != Any[TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken ([TypedMissing(:a)] != [TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (Any[TypedMissing(:a), 2] != Any[1, TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken ([TypedMissing(:a), false] != BitArray([true, false])) === TypedMissing(:a)
    @test_broken (Any[TypedMissing(:a), false] != BitArray([true, false])) === TypedMissing(:a)

    @test ismissing([1, TypedMissing(:a)] != [1, TypedMissing(:b)])
    @test ismissing(["a", TypedMissing(:a)] != ["a", TypedMissing(:b)])
    @test ismissing(Any[1, TypedMissing(:a)] != Any[1, TypedMissing(:b)])
    @test ismissing(Any[TypedMissing(:a)] != Any[TypedMissing(:b)])
    @test ismissing([TypedMissing(:a)] != [TypedMissing(:b)])
    @test ismissing(Any[TypedMissing(:a), 2] != Any[1, TypedMissing(:b)])

    @test_broken ([1, TypedMissing(:a)] != [1, TypedMissing(:b)]) === TypedMissing()
    @test_broken (["a", TypedMissing(:a)] != ["a", TypedMissing(:b)]) === TypedMissing()
    @test_broken (Any[1, TypedMissing(:a)] != Any[1, TypedMissing(:b)]) === TypedMissing()
    @test_broken (Any[TypedMissing(:a)] != Any[TypedMissing(:b)]) === TypedMissing()
    @test_broken ([TypedMissing(:a)] != [TypedMissing(:b)]) === TypedMissing()
    @test_broken (Any[TypedMissing(:a), 2] != Any[1, TypedMissing(:b)]) === TypedMissing()

    @test ismissing([1, TypedMissing(:a)] != [1, missing])
    @test ismissing(["a", TypedMissing(:a)] != ["a", missing])
    @test ismissing(Any[1, TypedMissing(:a)] != Any[1, missing])
    @test ismissing(Any[TypedMissing(:a)] != Any[missing])
    @test ismissing([TypedMissing(:a)] != [missing])
    @test ismissing(Any[TypedMissing(:a), 2] != Any[1, missing])

    @test_broken ([1, TypedMissing(:a)] != [1, missing]) === TypedMissing()
    @test_broken (["a", TypedMissing(:a)] != ["a", missing]) === TypedMissing()
    @test_broken (Any[1, TypedMissing(:a)] != Any[1, missing]) === TypedMissing()
    @test_broken (Any[TypedMissing(:a)] != Any[missing]) === TypedMissing()
    @test_broken ([TypedMissing(:a)] != [missing]) === TypedMissing()
    @test_broken (Any[TypedMissing(:a), 2] != Any[1, missing]) === TypedMissing()

    @test !(Union{Int, TypedMissing}[1] != Union{Float64, TypedMissing}[1.0])
    @test !(Union{Int, TypedMissing}[1] != Union{Float64, Missing}[1.0])
    @test !(Union{Int, TypedMissing}[1] != [1.0])
    @test !(Union{Bool, TypedMissing}[true] != BitArray([true]))
    @test [TypedMissing(:a), 1] != [TypedMissing(:a), 2]
    @test [TypedMissing(:a), 1] != [missing, 2]
    @test Union{Int, TypedMissing}[1] != [2]
    @test [1] != Union{Int, TypedMissing}[2]
    @test Union{Int, TypedMissing}[1] != Union{Int, TypedMissing}[2]
    @test Union{Int, TypedMissing}[1] != Union{Int, Missing}[2]
end

@testset "== and != on tuples" begin
    @test ismissing((1, TypedMissing(:a)) == (1, TypedMissing(:a)))
    @test ismissing(("a", TypedMissing(:a)) == ("a", TypedMissing(:a)))
    @test ismissing((TypedMissing(:a),) == (TypedMissing(:a),))
    @test ismissing((TypedMissing(:a), 2) == (1, TypedMissing(:a)))
    @test !((TypedMissing(:a), 1) == (TypedMissing(:a), 2))

    @test_broken ((1, TypedMissing(:a)) == (1, TypedMissing(:a))) === TypedMissing(:a)
    @test_broken (("a", TypedMissing(:a)) == ("a", TypedMissing(:a))) === TypedMissing(:a)
    @test_broken ((TypedMissing(:a),) == (TypedMissing(:a),)) === TypedMissing(:a)
    @test_broken ((TypedMissing(:a), 2) == (1, TypedMissing(:a))) === TypedMissing(:a)
    @test_broken !((TypedMissing(:a), 1) == (TypedMissing(:a), 2)) === TypedMissing(:a)

    @test ismissing((1, TypedMissing(:a)) == (1, TypedMissing()))
    @test ismissing(("a", TypedMissing(:a)) == ("a", TypedMissing()))
    @test ismissing((TypedMissing(:a),) == (TypedMissing(),))
    @test ismissing((TypedMissing(:a), 2) == (1, TypedMissing()))
    @test !((TypedMissing(:a), 1) == (TypedMissing(), 2))

    @test_broken ((1, TypedMissing(:a)) == (1, TypedMissing())) === TypedMissing()
    @test_broken (("a", TypedMissing(:a)) == ("a", TypedMissing())) === TypedMissing()
    @test_broken ((TypedMissing(:a),) == (TypedMissing(),)) === TypedMissing()
    @test_broken ((TypedMissing(:a), 2) == (1, TypedMissing())) === TypedMissing()
    @test_broken !((TypedMissing(:a), 1) == (TypedMissing(), 2)) === TypedMissing()

    @test ismissing((1, TypedMissing(:a)) == (1, missing))
    @test ismissing(("a", TypedMissing(:a)) == ("a", missing))
    @test ismissing((TypedMissing(:a),) == (missing,))
    @test ismissing((TypedMissing(:a), 2) == (1, missing))
    @test !((TypedMissing(:a), 1) == (missing, 2))

    @test_broken ((1, TypedMissing(:a)) == (1, missing)) === TypedMissing()
    @test_broken (("a", TypedMissing(:a)) == ("a", missing)) === TypedMissing()
    @test_broken ((TypedMissing(:a),) == (missing,)) === TypedMissing()
    @test_broken ((TypedMissing(:a), 2) == (1, missing)) === TypedMissing()
    @test_broken !((TypedMissing(:a), 1) == (missing, 2)) === TypedMissing()

    longtuple = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
    @test ismissing((longtuple...,17,TypedMissing(:a)) == (longtuple...,17,18))
    @test ismissing((longtuple...,TypedMissing(:a),18) == (longtuple...,17,18))
    @test_broken ((longtuple...,17,TypedMissing(:a)) == (longtuple...,17,18)) === TypedMissing(:a)
    @test_broken ((longtuple...,TypedMissing(:a),18) == (longtuple...,17,18)) === TypedMissing(:a)
    @test !((longtuple...,17,TypedMissing(:a)) == (longtuple...,-17,18))
    @test !((longtuple...,TypedMissing(:a),18) == (longtuple...,17,-18))

    @test ismissing((1, TypedMissing(:a)) != (1, TypedMissing(:a)))
    @test ismissing(("a", TypedMissing(:a)) != ("a", TypedMissing(:a)))
    @test ismissing((TypedMissing(:a),) != (TypedMissing(:a),))
    @test ismissing((TypedMissing(:a), 2) != (1, TypedMissing(:a)))
    @test (TypedMissing(:a), 1) != (TypedMissing(:a), 2)

    @test_broken ((1, TypedMissing(:a)) != (1, TypedMissing(:a))) === TypedMissing(:a)
    @test_broken (("a", TypedMissing(:a)) != ("a", TypedMissing(:a))) === TypedMissing(:a)
    @test_broken ((TypedMissing(:a),) != (TypedMissing(:a),)) === TypedMissing(:a)
    @test_broken ((TypedMissing(:a), 2) != (1, TypedMissing(:a))) === TypedMissing(:a)
    @test_broken (TypedMissing(:a), 1) != (TypedMissing(:a), 2) === TypedMissing(:a)

    @test ismissing((1, TypedMissing(:a)) != (1, TypedMissing(:b)))
    @test ismissing(("a", TypedMissing(:a)) != ("a", TypedMissing(:b)))
    @test ismissing((TypedMissing(:a),) != (TypedMissing(:b),))
    @test ismissing((TypedMissing(:a), 2) != (1, TypedMissing(:b)))
    @test (TypedMissing(:a), 1) != (TypedMissing(:b), 2)

    @test_broken ((1, TypedMissing(:a)) != (1, TypedMissing(:b))) === TypedMissing()
    @test_broken (("a", TypedMissing(:a)) != ("a", TypedMissing(:b))) === TypedMissing()
    @test_broken ((TypedMissing(:a),) != (TypedMissing(:b),)) === TypedMissing()
    @test_broken ((TypedMissing(:a), 2) != (1, TypedMissing(:b))) === TypedMissing()
    @test_broken (TypedMissing(:a), 1) != (TypedMissing(:b), 2) === TypedMissing()

    @test ismissing((1, TypedMissing(:a)) != (1, missing))
    @test ismissing(("a", TypedMissing(:a)) != ("a", missing))
    @test ismissing((TypedMissing(:a),) != (missing,))
    @test ismissing((TypedMissing(:a), 2) != (1, missing))
    @test (TypedMissing(:a), 1) != (missing, 2)

    @test_broken ((1, TypedMissing(:a)) != (1, missing)) === TypedMissing()
    @test_broken (("a", TypedMissing(:a)) != ("a", missing)) === TypedMissing()
    @test_broken ((TypedMissing(:a),) != (missing,)) === TypedMissing()
    @test_broken ((TypedMissing(:a), 2) != (1, missing)) === TypedMissing()
    @test_broken (TypedMissing(:a), 1) != (missing, 2) === TypedMissing()

    @test ismissing((longtuple...,17,TypedMissing(:a)) != (longtuple...,17,18))
    @test ismissing((longtuple...,TypedMissing(:a),18) != (longtuple...,17,18))
    @test_broken ((longtuple...,17,TypedMissing(:a)) != (longtuple...,17,18)) === TypedMissing(:a)
    @test_broken ((longtuple...,TypedMissing(:a),18) != (longtuple...,17,18)) === TypedMissing(:a)
    @test (longtuple...,17,TypedMissing(:a)) != (longtuple...,-17,18)
    @test (longtuple...,TypedMissing(:a),18) != (longtuple...,17,-18)
end

@testset "< and isless on tuples" begin
    @test ismissing((1, TypedMissing(:a)) < (1, 3))
    @test ismissing((1, TypedMissing(:a)) < (1, TypedMissing(:a)))
    @test ismissing((TypedMissing(:a), 1) < (TypedMissing(:a), 2))
    @test ismissing((1, 2) < (1, TypedMissing(:a)))
    @test ismissing((1, TypedMissing(:a)) < (1, 2))
    @test ismissing((TypedMissing(:a),) < (TypedMissing(:a),))
    @test ismissing((1,) < (TypedMissing(:a),))

    @test_broken ((1, TypedMissing(:a)) < (1, 3)) === TypedMissing(:a)
    @test_broken ((1, TypedMissing(:a)) < (1, TypedMissing(:a))) === TypedMissing(:a)
    @test_broken ((TypedMissing(:a), 1) < (TypedMissing(:a), 2)) === TypedMissing(:a)
    @test_broken ((1, 2) < (1, TypedMissing(:a))) === TypedMissing(:a)
    @test_broken ((1, TypedMissing(:a)) < (1, 2)) === TypedMissing(:a)
    @test_broken ((TypedMissing(:a),) < (TypedMissing(:a),)) === TypedMissing(:a)
    @test_broken ((1,) < (TypedMissing(:a),)) === TypedMissing(:a)

    @test ismissing((1, TypedMissing(:a)) < (1, TypedMissing(:b)))
    @test ismissing((TypedMissing(:a), 1) < (TypedMissing(:b), 2))
    @test ismissing((TypedMissing(:a),) < (TypedMissing(:b),))

    @test_broken ((1, TypedMissing(:a)) < (1, TypedMissing(:b))) === TypedMissing()
    @test_broken ((TypedMissing(:a), 1) < (TypedMissing(:b), 2)) === TypedMissing()
    @test_broken ((TypedMissing(:a),) < (TypedMissing(:b),)) === TypedMissing()

    @test ismissing((1, TypedMissing(:a)) < (1, missing))
    @test ismissing((TypedMissing(:a), 1) < (missing, 2))
    @test ismissing((TypedMissing(:a),) < (missing,))

    @test_broken ((1, TypedMissing(:a)) < (1, missing)) === TypedMissing()
    @test_broken ((TypedMissing(:a), 1) < (missing, 2)) === TypedMissing()
    @test_broken ((TypedMissing(:a),) < (missing,)) === TypedMissing()

    @test () < (TypedMissing(:a),)
    @test (1,) < (2, TypedMissing(:a))
    @test (1, TypedMissing(:a),) < (2, TypedMissing(:a))
    @test (1, TypedMissing(:a),) < (2, TypedMissing(:b))

    @test !isless((1, TypedMissing(:a)), (1, 3))
    @test !isless((1, TypedMissing(:a)), (1, TypedMissing(:a)))
    @test isless((TypedMissing(:a), 1), (TypedMissing(:a), 2))
    @test isless((1, 2), (1, TypedMissing(:a)))
    @test !isless((1, TypedMissing(:a)), (1, 2))
    @test !isless((TypedMissing(:a),), (TypedMissing(:a),))
    @test isless((1,), (TypedMissing(:a),))
    @test isless((), (TypedMissing(:a),))
    @test isless((1,), (2, TypedMissing(:a)))
    @test isless((1, TypedMissing(:a),), (2, TypedMissing(:a)))

    @test isless((1, TypedMissing(:a)), (1, TypedMissing(:b)))
    @test isless((1, missing), (1, TypedMissing(:a)))
    @test isless((1, TypedMissing(:a)), (1, TypedMissing(:b)))
    @test isless((TypedMissing(:a), 1), (TypedMissing(:a), 2))
    @test !isless((TypedMissing(:b),), (TypedMissing(:a),))
    @test isless((1, TypedMissing(:a),), (2, TypedMissing(:b)))
end

@testset "any & all" begin
    @test any([true, TypedMissing(:a)])
    @test any(x -> x == 1, [1, TypedMissing(:a)])

    @test ismissing(any([false, TypedMissing(:a)]))
    @test ismissing(any(x -> x == 1, [2, TypedMissing(:a)]))
    @test ismissing(all([true, TypedMissing(:a)]))
    @test ismissing(all(x -> x == 1, [1, TypedMissing(:a)]))

    @test_broken (any([false, TypedMissing(:a)])) === TypedMissing(:a)
    @test_broken (any(x -> x == 1, [2, TypedMissing(:a)])) === TypedMissing(:a)
    @test_broken (all([true, TypedMissing(:a)])) === TypedMissing(:a)
    @test_broken (all(x -> x == 1, [1, TypedMissing(:a)])) === TypedMissing(:a)

    @test !all([false, TypedMissing(:a)])
    @test !all(x -> x == 1, [2, TypedMissing(:a)])
    @test 1 in [1, TypedMissing(:a)]

    @test ismissing(2 in [1, TypedMissing(:a)])
    @test ismissing(TypedMissing(:a) in [1, TypedMissing(:a)])
    @test ismissing(TypedMissing(:a) in [1, TypedMissing(:b)])
    @test ismissing(missing in [1, TypedMissing(:a)])
    @test ismissing(TypedMissing(:a) in [1, missing])

    @test_broken (2 in [1, TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (TypedMissing(:a) in [1, TypedMissing(:a)]) === TypedMissing(:a)
    @test_broken (TypedMissing(:a) in [1, TypedMissing(:b)]) === TypedMissing()
    @test_broken (missing in [1, TypedMissing(:a)]) === TypedMissing()
    @test_broken (TypedMissing(:a) in [1, missing]) === TypedMissing()
end

@testset "float" begin
    @test isequal(float([1, TypedMissing(:a)]), [1, TypedMissing(:a)])
    @test float([1, TypedMissing(:a)]) isa Vector{Union{Float64, TypedMissing}}
    @test isequal(float(Union{Int, TypedMissing}[TypedMissing(:a)]), [TypedMissing(:a)])
    @test float(Union{Int, TypedMissing}[TypedMissing(:a)]) isa Vector{Union{Float64, TypedMissing}}
    @test float(Union{Int, TypedMissing}[1]) == [1]
    @test float(Union{Int, TypedMissing}[1]) isa Vector{Union{Float64, TypedMissing}}
    @test isequal(float([TypedMissing(:a)]), [TypedMissing(:a)])
    @test float([TypedMissing(:a)]) isa Vector{TypedMissing}
end

# TODO: test mixing Missing and TypedMissing
@testset "skipmissing" begin
    x = skipmissing([1, 2, TypedMissing(:a), 4])
    @test eltype(x) === Int
    @test collect(x) == [1, 2, 4]
    @test collect(x) isa Vector{Int}

    x = skipmissing([1  2; TypedMissing(:a) 4])
    @test eltype(x) === Int
    @test collect(x) == [1, 2, 4]
    @test collect(x) isa Vector{Int}

    x = skipmissing([TypedMissing(:a)])
    @test eltype(x) === Union{}
    @test isempty(collect(x))
    @test collect(x) isa Vector{Union{}}

    x = skipmissing(Union{Int, TypedMissing}[])
    @test eltype(x) === Int
    @test isempty(collect(x))
    @test collect(x) isa Vector{Int}

    x = skipmissing([TypedMissing(:a), TypedMissing(:a), 1, 2, TypedMissing(:a),
                     4, TypedMissing(:a), TypedMissing(:a)])
    @test eltype(x) === Int
    @test collect(x) == [1, 2, 4]
    @test collect(x) isa Vector{Int}

    x = skipmissing(v for v in [TypedMissing(:a), 1, TypedMissing(:a), 2, 4])
    @test eltype(x) === Any
    @test collect(x) == [1, 2, 4]
    @test collect(x) isa Vector{Int}

    @testset "indexing" begin
        x = skipmissing([1, TypedMissing(:a), 2, TypedMissing(:a), TypedMissing(:a)])
        @test collect(eachindex(x)) == collect(keys(x)) == [1, 3]
        @test x[1] === 1
        @test x[3] === 2
        @test_throws MissingException x[2]
        @test_throws BoundsError x[6]
        @test findfirst(==(2), x) == 3
        @test findall(==(2), x) == [3]
        @test argmin(x) == 1
        @test findmin(x) == (1, 1)
        @test argmax(x) == 3
        @test findmax(x) == (2, 3)

        x = skipmissing([TypedMissing(:a) 2; 1 TypedMissing(:a)])
        @test collect(eachindex(x)) == [2, 3]
        @test collect(keys(x)) == [CartesianIndex(2, 1), CartesianIndex(1, 2)]
        @test x[2] === x[2, 1] === 1
        @test x[3] === x[1, 2] === 2
        @test_throws MissingException x[1]
        @test_throws MissingException x[1, 1]
        @test_throws BoundsError x[5]
        @test_throws BoundsError x[3, 1]
        @test findfirst(==(2), x) == CartesianIndex(1, 2)
        @test findall(==(2), x) == [CartesianIndex(1, 2)]
        @test argmin(x) == CartesianIndex(2, 1)
        @test findmin(x) == (1, CartesianIndex(2, 1))
        @test argmax(x) == CartesianIndex(1, 2)
        @test findmax(x) == (2, CartesianIndex(1, 2))

        x = skipmissing([])
        @test isempty(collect(eachindex(x)))
        @test isempty(collect(keys(x)))
        @test_throws BoundsError x[3]
        @test_throws BoundsError x[3, 1]
        @test findfirst(==(2), x) === nothing
        @test isempty(findall(==(2), x))
        @test_throws ArgumentError("reducing over an empty collection is not allowed") argmin(x)
        @test_throws ArgumentError("reducing over an empty collection is not allowed") findmin(x)
        @test_throws ArgumentError("reducing over an empty collection is not allowed") argmax(x)
        @test_throws ArgumentError("reducing over an empty collection is not allowed") findmax(x)

        x = skipmissing([TypedMissing(:a), TypedMissing(:a)])
        @test isempty(collect(eachindex(x)))
        @test isempty(collect(keys(x)))
        @test_throws BoundsError x[3]
        @test_throws BoundsError x[3, 1]
        @test findfirst(==(2), x) === nothing
        @test isempty(findall(==(2), x))
        @test_throws ArgumentError("reducing over an empty collection is not allowed") argmin(x)
        @test_throws ArgumentError("reducing over an empty collection is not allowed") findmin(x)
        @test_throws ArgumentError("reducing over an empty collection is not allowed") argmax(x)
        @test_throws ArgumentError("reducing over an empty collection is not allowed") findmax(x)
    end

    @testset "mapreduce" begin
        # Vary size to test splitting blocks with several configurations of missing values
        for T in (Int, Float64),
            A in (rand(T, 10), rand(T, 1000), rand(T, 10000))
            if T === Int
                @test sum(A) === @inferred(sum(skipmissing(A))) ===
                    @inferred(reduce(+, skipmissing(A))) ===
                    @inferred(mapreduce(identity, +, skipmissing(A)))
            else
                @test sum(A) ≈ @inferred(sum(skipmissing(A))) ===
                    @inferred(reduce(+, skipmissing(A))) ===
                    @inferred(mapreduce(identity, +, skipmissing(A)))
            end
            @test mapreduce(cos, *, A) ≈
                @inferred(mapreduce(cos, *, skipmissing(A)))

            B = Vector{Union{T,TypedMissing}}(A)
            replace!(x -> rand(Bool) ? x : TypedMissing(:a), B)
            if T === Int
                @test sum(collect(skipmissing(B))) ===
                    @inferred(sum(skipmissing(B))) ===
                    @inferred(reduce(+, skipmissing(B))) ===
                    @inferred(mapreduce(identity, +, skipmissing(B)))
            else
                @test sum(collect(skipmissing(B))) ≈ @inferred(sum(skipmissing(B))) ===
                    @inferred(reduce(+, skipmissing(B))) ===
                    @inferred(mapreduce(identity, +, skipmissing(B)))
            end
            @test mapreduce(cos, *, collect(skipmissing(A))) ≈
                @inferred(mapreduce(cos, *, skipmissing(A)))

            # Test block full of missing values
            B[1:length(B)÷2] .= TypedMissing(:a)
            if T === Int
                @test sum(collect(skipmissing(B))) == sum(skipmissing(B)) ==
                    reduce(+, skipmissing(B)) == mapreduce(identity, +, skipmissing(B))
            else
                @test sum(collect(skipmissing(B))) ≈ sum(skipmissing(B)) ==
                    reduce(+, skipmissing(B)) == mapreduce(identity, +, skipmissing(B))
            end

            @test mapreduce(cos, *, collect(skipmissing(A))) ≈ mapreduce(cos, *, skipmissing(A))
        end

        # Patterns that exercize code paths for inputs with 1 or 2 non-missing values
        @test sum(skipmissing([1, TypedMissing(:a), TypedMissing(:a), TypedMissing(:a)])) === 1
        @test sum(skipmissing([TypedMissing(:a), TypedMissing(:a), TypedMissing(:a), 1])) === 1
        @test sum(skipmissing([1, TypedMissing(:a), TypedMissing(:a), TypedMissing(:a), 2])) === 3
        @test sum(skipmissing([TypedMissing(:a), TypedMissing(:a), TypedMissing(:a), 1, 2])) === 3

        for n in 0:3
            itr = skipmissing(Vector{Union{Int,TypedMissing}}(fill(TypedMissing(:a), n)))
            if n == 0
                @test sum(itr) == reduce(+, itr) == mapreduce(identity, +, itr) === 0
            else
                @test sum(itr) == reduce(+, itr) == mapreduce(identity, +, itr) === 0
            end
            @test_throws ArgumentError("reducing over an empty collection is not allowed") reduce(x -> x/2, itr)
            @test_throws ArgumentError("reducing over an empty collection is not allowed") mapreduce(x -> x/2, +, itr)
        end

        # issue JuliaLang/julia#35504
        nt = NamedTuple{(:x, :y),Tuple{Union{TypedMissing, Int},Union{TypedMissing, Float64}}}(
            (TypedMissing(:a), TypedMissing(:a)))
        @test sum(skipmissing(nt)) === 0

        # issues JuliaLang/julia#38627 and JuliaLang/julia#124
        @testset for len in [1, 2, 15, 16, 1024, 1025]
            v = repeat(Union{Int,TypedMissing}[1], len)
            oa = OffsetArray(v, typemax(Int)-length(v))
            sm = skipmissing(oa)
            @test sum(sm) == len

            v = repeat(Union{Int,TypedMissing}[TypedMissing(:a)], len)
            oa = OffsetArray(v, typemax(Int)-length(v))
            sm = skipmissing(oa)
            @test sum(sm) == 0
        end
    end

    @testset "filter" begin
        allmiss = Vector{Union{Int,TypedMissing}}(TypedMissing(:a), 10)
        @test isempty(filter(isodd, skipmissing(allmiss))::Vector{Int})
        twod1 = [1.0f0 TypedMissing(:a); 3.0f0 TypedMissing(:a)]
        @test filter(x->x > 0, skipmissing(twod1))::Vector{Float32} == [1, 3]
        twod2 = [1.0f0 2.0f0; 3.0f0 4.0f0]
        @test filter(x->x > 0, skipmissing(twod2)) == reshape(twod2, (4,))
    end
end

@testset "coalesce" begin
    @test coalesce(TypedMissing(:a)) === missing
    @test coalesce(TypedMissing(:a), 1) === 1
    @test coalesce(1, TypedMissing(:a)) === 1
    @test coalesce(TypedMissing(:a), TypedMissing(:b)) === missing
    @test coalesce(TypedMissing(:a), missing) === missing
    @test coalesce(missing, TypedMissing(:a)) === missing
    @test coalesce(TypedMissing(:a), 1, 2) === 1
    @test coalesce(1, TypedMissing(:a), 2) === 1
    @test coalesce(TypedMissing(:a), TypedMissing(:a), 2) === 2
    @test coalesce(TypedMissing(:a), TypedMissing(:b), TypedMissing(:c)) === missing
    @test coalesce(TypedMissing(:a), TypedMissing(:b), missing) === missing

    @test coalesce(nothing, TypedMissing(:a)) === nothing
    @test coalesce(TypedMissing(:a), nothing) === nothing
end

@testset "@coalesce" begin
    @test @coalesce(TypedMissing(:a)) === missing
    @test @coalesce(TypedMissing(:a), 1) === 1
    @test @coalesce(1, TypedMissing(:a)) === 1
    @test @coalesce(TypedMissing(:a), TypedMissing(:b)) === missing
    @test @coalesce(TypedMissing(:a), missing) === missing
    @test @coalesce(missing, TypedMissing(:a)) === missing
    @test @coalesce(TypedMissing(:a), 1, 2) === 1
    @test @coalesce(1, TypedMissing(:a), 2) === 1
    @test @coalesce(TypedMissing(:a), TypedMissing(:a), 2) === 2
    @test @coalesce(TypedMissing(:a), TypedMissing(:b), TypedMissing(:c)) === missing
    @test @coalesce(TypedMissing(:a), TypedMissing(:b), missing) === missing

    @test @coalesce(nothing, TypedMissing(:a)) === nothing
    @test @coalesce(TypedMissing(:a), nothing) === nothing

    @test @coalesce(1, error("failed")) === 1
    @test_throws ErrorException @coalesce(TypedMissing(:a), error("failed"))
end

mutable struct Obj; x; end
@testset "weak references" begin
    @noinline function mk_wr(r, wr)
        x = Obj(1)
        push!(r, x)
        push!(wr, WeakRef(x))
        nothing
    end
    ref = []
    wref = []
    mk_wr(ref, wref)
    @test (wref[1] == TypedMissing(:a)) === TypedMissing(:a)
    @test (TypedMissing(:a) == wref[1]) === TypedMissing(:a)
end

# TODO: test various kinds of TypedMissing in the same vector
@testset "sort and sortperm with $(eltype(X))" for (X, P, RP) in
    (([2, TypedMissing(:a), -2, 5, TypedMissing(:a)], [3, 1, 4, 2, 5], [2, 5, 4, 1, 3]),
     ([NaN, TypedMissing(:a), 5, -0.0, NaN, TypedMissing(:a), Inf, 0.0, -Inf],
      [9, 4, 8, 3, 7, 1, 5, 2, 6], [2, 6, 1, 5, 7, 3, 8, 4, 9]),
     ([TypedMissing(:a), "a", "c", TypedMissing(:a), "b"], [2, 5, 3, 1, 4], [1, 4, 3, 5, 2]))
    @test sortperm(X) == P
    @test sortperm(X, alg=QuickSort) == P
    @test sortperm(X, alg=MergeSort) == P

    XP = X[P]
    @test isequal(sort(X), XP)
    @test isequal(sort(X, alg=QuickSort), XP)
    @test isequal(sort(X, alg=MergeSort), XP)

    @test sortperm(X, rev=true) == RP
    @test sortperm(X, alg=QuickSort, rev=true) == RP
    @test sortperm(X, alg=MergeSort, rev=true) == RP

    XRP = X[RP]
    @test isequal(sort(X, rev=true), XRP)
    @test isequal(sort(X, alg=QuickSort, rev=true), XRP)
    @test isequal(sort(X, alg=MergeSort, rev=true), XRP)
end

sortperm(reverse([NaN, TypedMissing(:a), NaN, TypedMissing(:a)]))
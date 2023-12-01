module TypedMissings

export TypedMissing, MissingKind, MissingKinds
module MissingKinds
    export MissingKind

    @enum MissingKind::UInt8 begin
        NI=0    # No information
        INV=1   # Invalid
        OTH=2   # Other
        NINF=3  # Negative infinity
        PINF=4  # Positive infinity
        UNC=5   # Unencoded
        DER=6   # Derived
        UNK=7   # Unknown
        ASKU=8  # Asked but unknown
        NAV=9   # Temporarily not available
        NAVU=10 # Not available
        NASK=11 # Not asked
        QS=12   # Sufficient quantity
        TRC=13  # Trace
        MSK=14  # Masked
        NA=15   # Not applicable

        a=101
        b=102
        c=103
        d=104
        e=105
        f=106
        g=107
        h=108
        i=109
        j=110
        k=111
        l=112
        m=113
        n=114
        o=115
        p=116
        q=117
        r=118
        s=119
        t=120
        u=121
        v=122
        w=123
        x=124
        y=125
        z=126
    end
end
using .MissingKinds

"""
    TypedMissing(kind::MissingKind=MissingKinds.NI)

A type similar to `Missing` that allows representing multiple kinds
of missing values. The default kind is `MissingKinds.NI` ("No information"),
which is equivalent to `missing`.
Values of type `TypedMissing` behave identically to `missing`, except that
`isequal(TypedMissing(kind), missing)` is `true` if and only if
`kind == MissingKinds.NI`.

`TypedMissing` values propagate their `kind` across operations
which involve only missing values of the same `kind` or non-missing
values. Operations mixing `TypedMissing` values of different `kind`s
fall back to the `MissingKinds.NI` kind.

If provided, `kind` must be an instance of the `MissingKind` enum.
Supported `kind`s are `NI`, lowercase letters from a to z, and "null flavour"
values [defined by FHIR HL7 v3](https://www.hl7.org/fhir/v3/NullFlavor/cs.html)/
ISO 21090:
- NI: No information
- INV: Invalid
- OTH: Other
- NINF: Negative infinity
- PINF: Positive infinity
- UNC: Unencoded
- DER: Derived
- UNK: Unknown
- ASKU: Asked but unknown
- NAV: Temporarily not available
- NAVU: Not available
- QS: Sufficient quantity
- NASK: Not asked
- TRC: Trace
- MSK: Masked
- NA: Not applicable

# Examples
```jldoctest
julia> TypedMissing()
TypedMissing()

julia> TypedMissing(MissingKinds.NI)
TypedMissing()

julia> TypedMissing() + 1
TypedMissing()

julia> TypedMissing(MissingKinds.a) + 1
TypedMissing(MissingKinds.a)

julia> TypedMissing(MissingKinds.NASK) + 1
TypedMissing(MissingKinds.NASK)

julia> TypedMissing(MissingKinds.NASK) + TypedMissing(MissingKinds.INV)
TypedMissing()

julia> TypedMissing(MissingKinds.NASK) + missing
TypedMissing()

julia> TypedMissing(MissingKinds.NASK) + 1
TypedMissing(MissingKinds.NASK)

julia> isequal(TypedMissing(MissingKinds.NASK), TypedMissing(MissingKinds.NASK))
true

julia> isequal(TypedMissing(MissingKinds.NASK), TypedMissing(MissingKinds.INV))
false

julia> isequal(TypedMissing(MissingKinds.NASK), missing)
false

julia> isequal(TypedMissing(MissingKinds.NI), missing)
true
```
"""
struct TypedMissing
    kind::MissingKind
end
TypedMissing() = TypedMissing(MissingKinds.NI)

function Base.show(io::IO, x::TypedMissing)
    show(io, TypedMissing)
    print(io, '(')
    if x.kind != MissingKinds.NI
        print(io, "MissingKinds.", x.kind)
    end
    print(io, ')')
end

function Base.show(io::IO, ::MIME"text/plain", x::TypedMissing)
    show(io, TypedMissing)
    print(io, '(')
    if x.kind != MissingKinds.NI
        get(io, :compact, false) || print(io, "MissingKinds.")
        print(io, x.kind)
    end
    print(io, ')')
end

Base.ismissing(::TypedMissing) = true
Base.coalesce(x::TypedMissing, y...) = coalesce(y...)

# Semi type piracy
Base.nonmissingtype(::Type{T}) where {T >: TypedMissing} =
    Base.typesplit(T, Union{Missing, TypedMissing})

Base.broadcastable(x::TypedMissing) = Ref(x)

Base.promote_rule(T::Type{TypedMissing}, S::Type) = Union{S, TypedMissing}
Base.promote_rule(T::Type{TypedMissing}, S::Type{Missing}) = TypedMissing
Base.promote_rule(T::Type{Missing}, S::Type{TypedMissing}) = TypedMissing
Base.promote_rule(T::Type{Union{Nothing, TypedMissing}}, S::Type) = Union{S, Nothing, TypedMissing}

function _promote_nothing(T::Type, S::Type)
    R = Base.nonnothingtype(T)
    R >: T && return Any
    T = R
    R = nonmissingtype(T)
    R >: T && return Any
    T = R
    R = promote_type(T, S)
    return Union{R, Nothing, TypedMissing}
end
Base.promote_rule(T::Type{>:Union{Nothing, TypedMissing}}, S::Type) = _promote_nothing(T, S)
Base.promote_rule(T::Type{>:Union{Nothing, Missing, TypedMissing}}, S::Type) =
    _promote_nothing(T, S)
Base.promote_rule(T::Type{>:Union{Nothing, Missing, TypedMissing}}, S::Type{>:TypedMissing}) =
    _promote_nothing(T, S)
Base.promote_rule(T::Type{>:Union{Nothing, Missing}}, S::Type{>:TypedMissing}) =
    _promote_nothing(T, S)

function _promote_missing(T::Type, S::Type)
    R = nonmissingtype(T)
    R >: T && return Any
    T = R
    R = promote_type(T, nonmissingtype(S))
    return Union{R, TypedMissing}
end
Base.promote_rule(T::Type{>:TypedMissing}, S::Type) =
    _promote_missing(T, S)
Base.promote_rule(T::Type{>:Union{Missing, TypedMissing}}, S::Type) =
    _promote_missing(T, S)
Base.promote_rule(T::Type{>:Missing}, S::Type{>:TypedMissing}) =
    _promote_missing(T, S)
Base.promote_rule(T::Type{>:Union{Missing, TypedMissing}}, S::Type{>:TypedMissing}) =
    _promote_missing(T, S)
Base.promote_rule(T::Type{Missing}, S::Type{>:TypedMissing}) =
    _promote_missing(T, S)

Base.convert(::Type{T}, x::T) where {T>:TypedMissing} = x
Base.convert(::Type{T}, x::T) where {T>:Union{TypedMissing, Nothing}} = x
Base.convert(::Type{T}, x::T) where {T>:Union{TypedMissing, Missing}} = x
Base.convert(::Type{T}, x) where {T>:TypedMissing} =
    convert(Base.nonmissingtype_checked(T), x)
Base.convert(::Type{T}, x::Missing) where {T>:TypedMissing} =
    TypedMissing()
Base.convert(::Type{T}, x) where {T>:Union{TypedMissing, Nothing}} =
    convert(Base.nonmissingtype_checked(Base.nonnothingtype_checked(T)), x)
Base.convert(::Type{T}, x) where {T>:Union{TypedMissing, Missing}} =
    convert(Base.nonmissingtype_checked(T), x)
Base.convert(::Type{T}, x) where {T>:Union{TypedMissing, Missing, Nothing}} =
    convert(Base.nonmissingtype_checked(Base.nonnothingtype_checked(T)), x)
Base.convert(::Type{T}, x::Missing) where {T>:Union{TypedMissing, Nothing}} =
    TypedMissing()
Base.convert(::Type{T}, x::Missing) where {T>:Union{TypedMissing, Missing}} =
    missing
Base.convert(::Type{T}, x::Missing) where {T>:Union{TypedMissing, Missing, Nothing}} =
    missing
Base.convert(::Type{Any}, x::Missing) = missing
Base.convert(::Type{T}, x::TypedMissing) where {T>:Union{TypedMissing, Missing, Nothing}} =
    x
Base.convert(::Type{Any}, x::TypedMissing) = x
Base.convert(::Type{T}, x::TypedMissing) where {T>:Missing} = missing
Base.convert(::Type{T}, x::TypedMissing) where {T>:Union{Missing, Nothing}} = missing

Array{T,N}(x::TypedMissing, d...) where {T,N} = fill!(Array{T,N}(undef, d...), x)
Array{T}(x::TypedMissing, d...) where {T} = fill!(Array{T}(undef, d...), x)

# Comparison operators
Base.:(==)(x::TypedMissing, y::TypedMissing) =
    x === y ? x : TypedMissing()
Base.:(==)(x::TypedMissing, ::Any) = x
Base.:(==)(::Any, y::TypedMissing) = y
# To fix ambiguity
Base.:(==)(x::TypedMissing, ::WeakRef) = x
Base.:(==)(::WeakRef, y::TypedMissing) = y
Base.:(==)(x::TypedMissing, ::Missing) = TypedMissing()
Base.:(==)(::Missing, y::TypedMissing) = TypedMissing()

Base.isequal(x::TypedMissing, y::TypedMissing) = x === y
Base.isequal(x::TypedMissing, y::Missing) = x.kind == MissingKinds.NI
Base.isequal(x::Missing, y::TypedMissing) = y.kind == MissingKinds.NI
Base.isequal(x::TypedMissing, y::Any) = false
Base.isequal(x::Any, y::TypedMissing) = false

Base.:(<)(x::TypedMissing, y::TypedMissing) =
    x === y ? x : TypedMissing()
Base.:(<)(x::TypedMissing, ::Any) = x
Base.:(<)(::Any, y::TypedMissing) = y
Base.:(<)(x::TypedMissing, ::Missing) = TypedMissing()
Base.:(<)(::Missing, y::TypedMissing) = TypedMissing()

Base.isless(x::TypedMissing, y::TypedMissing) =
    isless(x.kind, y.kind)
Base.isless(x::TypedMissing, y::Missing) = false
Base.isless(x::Missing, y::TypedMissing) = y.kind != MissingKinds.NI
Base.isless(x::TypedMissing, ::Any) = false
Base.isless(::Any, y::TypedMissing) = true

Base.isapprox(x::TypedMissing, y::TypedMissing; kwargs...) =
    x === y ? x : TypedMissing()
Base.isapprox(x::TypedMissing, ::Any; kwargs...) = x
Base.isapprox(::Any, y::TypedMissing; kwargs...) = y
Base.isapprox(x::TypedMissing, ::Missing; kwargs...) = TypedMissing()
Base.isapprox(::Missing, y::TypedMissing; kwargs...) = TypedMissing()

# Unary operators/functions
for f in (:(!), :(~), :(+), :(-), :(*), :(&), :(|), :(xor),
          :(zero), :(one), :(oneunit),
          :(isfinite), :(isinf), :(isodd),
          :(isinteger), :(isreal), :(isnan),
          :(iszero), :(transpose), :(adjoint), :(float), :(complex), :(conj),
          :(abs), :(abs2), :(iseven), :(ispow2),
          :(real), :(imag), :(sign), :(inv))
    @eval (Base.$f)(x::TypedMissing) = x
end
for f in (:(Base.zero), :(Base.one), :(Base.oneunit))
    @eval ($f)(x::Type{TypedMissing}) = TypedMissing()
    @eval function $(f)(::Type{Union{T, TypedMissing}}) where T
        T === Any && throw(MethodError($f, (Any,)))  # To prevent StackOverflowError
        $f(T)
    end
end
for f in (:(Base.float), :(Base.complex))
    @eval $f(::Type{TypedMissing}) = TypedMissing
    @eval function $f(::Type{Union{T, TypedMissing}}) where T
        T === Any && throw(MethodError($f, (Any,)))  # To prevent StackOverflowError
        Union{$f(T), TypedMissing}
    end
end

# Binary operators/functions
for f in (:(+), :(-), :(*), :(/), :(^), :(mod), :(rem))
    @eval begin
        # Scalar with missing
        (Base.$f)(x::TypedMissing, y::TypedMissing) =
            x === y ? x : TypedMissing()
        (Base.$f)(x::TypedMissing, ::Number) = x
        (Base.$f)(::Number, y::TypedMissing) = y
        (Base.$f)(x::TypedMissing, ::Missing) = TypedMissing()
        (Base.$f)(::Missing, y::TypedMissing) = TypedMissing()
    end
end

Base.div(x::TypedMissing, y::TypedMissing, r::RoundingMode) =
    x === y ? x : TypedMissing()
Base.div(x::TypedMissing, ::Number, r::RoundingMode) = x
Base.div(::Number, y::TypedMissing, r::RoundingMode) = y
Base.div(x::TypedMissing, ::Missing, r::RoundingMode) = TypedMissing()
Base.div(::Missing, y::TypedMissing, r::RoundingMode) = TypedMissing()

Base.min(x::TypedMissing, y::TypedMissing) =
    x === y ? x : TypedMissing()
Base.min(x::TypedMissing, ::Any) = x
Base.min(::Any, y::TypedMissing) = y
Base.min(x::TypedMissing, ::Missing) = TypedMissing()
Base.min(::Missing, y::TypedMissing) = TypedMissing()
Base.max(x::TypedMissing, y::TypedMissing) =
    x === y ? x : TypedMissing()
Base.max(x::TypedMissing, ::Any)     = x
Base.max(::Any,     y::TypedMissing) = y
Base.max(x::TypedMissing, ::Missing) = TypedMissing()
Base.max(::Missing, y::TypedMissing) = TypedMissing()

# Math functions
for f in (:sin, :cos, :tan, :asin, :atan, :acos,
          :sinh, :cosh, :tanh, :asinh, :acosh, :atanh,
          :exp, :exp2, :exp10, :expm1, :log, :log2, :log10, :log1p,
          :exponent, :sqrt, :cbrt)
    @eval Base.$(f)(x::TypedMissing) = x
end

for f in (:atan, :hypot, :log)
    @eval Base.$(f)(x::TypedMissing, y::Missing) = TypedMissing()
    @eval Base.$(f)(x::Missing, y::TypedMissing) = TypedMissing()
    @eval Base.$(f)(x::TypedMissing, y::TypedMissing) =
        x === y ? x : TypedMissing()
    @eval Base.$(f)(x::Number, y::TypedMissing) = y
    @eval Base.$(f)(x::TypedMissing, y::Number) = x
end

Base.clamp(x::TypedMissing, lo, hi) = x

# Rounding and related functions
Base.round(x::TypedMissing, ::RoundingMode=RoundNearest;
           sigdigits::Integer=0, digits::Integer=0, base::Integer=0) = x
Base.round(::Type{>:TypedMissing}, x::TypedMissing, ::RoundingMode=RoundNearest) = x
Base.round(::Type{T}, ::TypedMissing, ::RoundingMode=RoundNearest) where {T} =
    throw(MissingException("cannot convert a typed missing value to type $T: use Union{$T, TypedMissing} instead"))
Base.round(::Type{T}, x::Any, r::RoundingMode=RoundNearest) where {T>:TypedMissing} =
    round(Base.nonmissingtype_checked(T), x, r)
# to fix ambiguities
Base.round(::Type{T}, x::Rational{Tr}, r::RoundingMode=RoundNearest) where {T>:TypedMissing,Tr} =
    round(Base.nonmissingtype_checked(T), x, r)
Base.round(::Type{T}, x::Rational{Bool}, r::RoundingMode=RoundNearest) where {T>:TypedMissing} =
    round(Base.nonmissingtype_checked(T), x, r)

# Handle ceil, floor, and trunc separately as they have no RoundingMode argument
for f in (:(ceil), :(floor), :(trunc))
    @eval begin
        (Base.$f)(x::TypedMissing; sigdigits::Integer=0, digits::Integer=0, base::Integer=0) = x
        (Base.$f)(::Type{>:TypedMissing}, x::TypedMissing) = x
        (Base.$f)(::Type{T}, ::TypedMissing) where {T} =
            throw(MissingException("cannot convert a typed missing value to type $T: use Union{$T, TypedMissing} instead"))
        (Base.$f)(::Type{T}, x::Any) where {T>:TypedMissing} =
            $f(Base.nonmissingtype_checked(T), x)
        # to fix ambiguities
        (Base.$f)(::Type{T}, x::Rational) where {T>:TypedMissing} =
            $f(Base.nonmissingtype_checked(T), x)
    end
end

# to avoid ambiguity warnings
Base.:(^)(x::TypedMissing, ::Integer) = x

# Bit operators
Base.:(&)(x::TypedMissing, y::TypedMissing) =
    x === y ? x : TypedMissing()
Base.:(&)(x::TypedMissing, y::Missing) = TypedMissing()
Base.:(&)(x::Missing, y::TypedMissing) = TypedMissing()
Base.:(&)(x::TypedMissing, y::Bool) = ifelse(y, x, false)
Base.:(&)(x::Bool, y::TypedMissing) = ifelse(x, y, false)
Base.:(&)(x::TypedMissing, ::Integer) = x
Base.:(&)(::Integer, y::TypedMissing) = y
Base.:(|)(x::TypedMissing, y::TypedMissing) =
    x === y ? x : TypedMissing()
Base.:(|)(x::TypedMissing, y::Missing) = TypedMissing()
Base.:(|)(x::Missing, y::TypedMissing) = TypedMissing()
Base.:(|)(x::TypedMissing, y::Bool) = ifelse(y, true, x)
Base.:(|)(x::Bool, y::TypedMissing) = ifelse(x, true, y)
Base.:(|)(x::TypedMissing, ::Integer) = x
Base.:(|)(::Integer, y::TypedMissing) = y
Base.xor(x::TypedMissing, y::TypedMissing) =
    x === y ? x : TypedMissing()
Base.xor(x::TypedMissing, y::Missing) = TypedMissing()
Base.xor(x::Missing, y::TypedMissing) = TypedMissing()
Base.xor(x::TypedMissing, y::Bool) = x
Base.xor(x::Bool, y::TypedMissing) = y
Base.xor(x::TypedMissing, ::Integer) = x
Base.xor(::Integer, y::TypedMissing) = y

Base.:(*)(d::TypedMissing, x::Union{AbstractString,AbstractChar}) = d
Base.:(*)(d::Union{AbstractString,AbstractChar}, x::TypedMissing) = x

function Base.float(A::AbstractArray{Union{T, TypedMissing}}) where {T}
    U = typeof(float(zero(T)))
    convert(AbstractArray{Union{U, TypedMissing}}, A)
end
Base.float(A::AbstractArray{TypedMissing}) = A


## Redefinition of Base methods
## (temporary type piracy to use `ismissing` instead of `=== missing`,
## needed until Base is fixed to do the same)

# This method is the only one which cannot be submitted to Base as-is,
# an extensible mechanism would be needed instead (JuliaLang/julia#38241)
Base._promote_typesubtract(@nospecialize(a)) =
    Base.typesplit(a, Union{Nothing, Missing, TypedMissing})

function Base.iterate(itr::Base.SkipMissing, state...)
    y = iterate(itr.x, state...)
    y === nothing && return nothing
    item, state = y
    while ismissing(item)
        y = iterate(itr.x, state)
        y === nothing && return nothing
        item, state = y
    end
    item, state
end

Base.eachindex(itr::Base.SkipMissing) =
    Iterators.filter(i -> !ismissing(@inbounds(itr.x[i])), eachindex(itr.x))
Base.keys(itr::Base.SkipMissing) =
    Iterators.filter(i -> !ismissing(@inbounds(itr.x[i])), keys(itr.x))
Base.@propagate_inbounds function Base.getindex(itr::Base.SkipMissing, I...)
    v = itr.x[I...]
    ismissing(v) && throw(MissingException("the value at index $I is missing"))
    v
end

Base.mapreduce(f, op, itr::Base.SkipMissing{<:AbstractArray}) =
    Base._mapreduce(f, op, IndexStyle(itr.x),
                    eltype(itr.x) >: Missing || eltype(itr.x) >: TypedMissing ? itr : itr.x)

function Base._mapreduce(f, op, ::IndexLinear, itr::Base.SkipMissing{<:AbstractArray})
    A = itr.x
    # TODO: use TypedMissing() when applicable?
    ai = missing
    inds = LinearIndices(A)
    i = first(inds)
    ilast = last(inds)
    for outer i in i:ilast
        @inbounds ai = A[i]
        !ismissing(ai) && break
    end
    ismissing(ai) && return Base.mapreduce_empty(f, op, eltype(itr))
    a1::eltype(itr) = ai
    i == typemax(typeof(i)) && return Base.mapreduce_first(f, op, a1)
    i += 1
    ai = missing
    for outer i in i:ilast
        @inbounds ai = A[i]
        !ismissing(ai) && break
    end
    ismissing(ai) && return Base.mapreduce_first(f, op, a1)
    # We know A contains at least two non-missing entries: the result cannot be nothing
    something(Base.mapreduce_impl(f, op, itr, first(inds), last(inds)))
end

# Returns nothing when the input contains only missing values, and Some(x) otherwise
@noinline function Base.mapreduce_impl(f, op, itr::Base.SkipMissing{<:AbstractArray},
                                       ifirst::Integer, ilast::Integer, blksize::Int)
    A = itr.x
    if ifirst > ilast
        return nothing
    elseif ifirst == ilast
        @inbounds a1 = A[ifirst]
        if ismissing(ai)
            return nothing
        else
            return Some(Base.mapreduce_first(f, op, a1))
        end
    elseif ilast - ifirst < blksize
        # sequential portion
        # TODO: use TypedMissing() when applicable?
        ai = missing
        i = ifirst
        for outer i in i:ilast
            @inbounds ai = A[i]
            !ismissing(ai) && break
        end
        ismissing(ai) && return nothing
        a1 = ai::eltype(itr)
        i == typemax(typeof(i)) && return Some(Base.mapreduce_first(f, op, a1))
        i += 1
        ai = missing
        for outer i in i:ilast
            @inbounds ai = A[i]
            !ismissing(ai) && break
        end
        ismissing(ai) && return Some(Base.mapreduce_first(f, op, a1))
        a2 = ai::eltype(itr)
        i == typemax(typeof(i)) && return Some(op(f(a1), f(a2)))
        i += 1
        v = op(f(a1), f(a2))
        @simd for i = i:ilast
            @inbounds ai = A[i]
            if !ismissing(ai)
                v = op(v, f(ai))
            end
        end
        return Some(v)
    else
        # pairwise portion
        imid = ifirst + (ilast - ifirst) >> 1
        v1 = Base.mapreduce_impl(f, op, itr, ifirst, imid, blksize)
        v2 = Base.mapreduce_impl(f, op, itr, imid+1, ilast, blksize)
        if v1 === nothing && v2 === nothing
            return nothing
        elseif v1 === nothing
            return v2
        elseif v2 === nothing
            return v1
        else
            return Some(op(something(v1), something(v2)))
        end
    end
end

function Base.filter(f, itr::Base.SkipMissing{<:AbstractArray})
    y = similar(itr.x, eltype(itr), 0)
    for xi in itr.x
        if !ismissing(xi) && f(xi)
            push!(y, xi)
        end
    end
    y
end

if VERSION >= v"1.7.0"
    import Base: @coalesce
    macro coalesce(args...)
        expr = :(missing)
        for arg in reverse(args)
            expr = :((val = $arg); !ismissing(val) ? val : $expr)
        end
        return esc(:(let val; $expr; end))
    end
end

end # module TypedMissings

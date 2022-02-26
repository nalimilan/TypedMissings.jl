module TypedMissings

export TypedMissing

abstract type AbstractMissing end

export TypedMissing

struct TypedMissing <: AbstractMissing
    kind::Symbol
end
TypedMissing() = TypedMissing(Symbol(""))

function Base.show(io::IO, x::TypedMissing)
    print(io, "TypedMissing(")
    x.kind !== Symbol("") && show(io, x.kind)
    print(io, ')')
end

Base.ismissing(::TypedMissing) = true
Base.coalesce(x::TypedMissing, y...) = coalesce(y...)
Base.coalesce(x::TypedMissing) = x

# Semi type piracy
Base.nonmissingtype(::Type{T}) where {T >: TypedMissing} =
    Base.typesplit(T, Union{Missing, TypedMissing})

Base.broadcastable(x::TypedMissing) = Ref(x)

Base.promote_rule(T::Type{TypedMissing}, S::Type) = Union{S, TypedMissing}
Base.promote_rule(T::Type{TypedMissing}, S::Type{Missing}) = TypedMissing
Base.promote_rule(T::Type{Missing}, S::Type{TypedMissing}) = TypedMissing
Base.promote_rule(T::Type{Union{Nothing, TypedMissing}}, S::Type) = Union{S, Nothing, TypedMissing}
function Base.promote_rule(T::Type{>:Union{Nothing, TypedMissing}}, S::Type)
    R = Base.nonnothingtype(T)
    R >: T && return Any
    T = R
    R = nonmissingtype(T)
    R >: T && return Any
    T = R
    R = promote_type(T, S)
    return Union{R, Nothing, TypedMissing}
end
function Base.promote_rule(T::Type{>:TypedMissing}, S::Type)
    R = nonmissingtype(T)
    R >: T && return Any
    T = R
    R = promote_type(T, nonmissingtype(S))
    return Union{R, TypedMissing}
end
function Base.promote_rule(T::Type{>:Union{Missing, TypedMissing}}, S::Type)
    R = nonmissingtype(T)
    R >: T && return Any
    T = R
    R = promote_type(T, S)
    return Union{R, TypedMissing}
end
function Base.promote_rule(T::Type{>:Union{Nothing, Missing, TypedMissing}}, S::Type)
    R = Base.nonnothingtype(T)
    R >: T && return Any
    T = R
    R = nonmissingtype(T)
    R >: T && return Any
    T = R
    R = promote_type(T, S)
    return Union{R, Nothing, TypedMissing}
end

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
Base.isequal(x::TypedMissing, y::Missing) = x.kind === Symbol("")
Base.isequal(x::Missing, y::TypedMissing) = y.kind === Symbol("")
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
Base.isless(x::Missing, y::TypedMissing) = y.kind !== Symbol("")
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

end # module TypedMissings

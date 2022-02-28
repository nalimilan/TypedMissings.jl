# TypedMissings.jl

[![CI](https://github.com/nalimilan/TypedMissings.jl/workflows/CI/badge.svg)](https://github.com/JuliaData/Missings.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/nalimilan/TypedMissings.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaData/Missings.jl)

*Support for different kinds of missing values in Julia*

TypedMissings.jl provides the `TypedMissing` type which is
similar to `Missing` but allows representing multiple kinds
of missing values.
Values of type `TypedMissing` behave identically to `missing`, except that
`isequal(TypedMissing(kind), missing)` is `true` if and only if
`kind == MissingKinds.NI`. The default kind is `MissingKinds.NI` ("No information"),
which is equivalent to `missing`.

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
- NASK:Not asked
- TRC: Trace
- MSK: Masked
- NA: Not applicable

# Examples
```julia
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
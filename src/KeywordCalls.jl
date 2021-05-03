module KeywordCalls

using NestedTuples
using MLStyle: @match
using GeneralizedGenerated

export kwcall, @kwcall

const TypeLevel = GeneralizedGenerated.TypeLevel


"""
From @thautwarm

we use this to avoid introduce static type parameters
for generated functions
"""
_unwrap_type(a::Type{<:Type}) = a.parameters[1]

"""
    kwcallperm(f, keys::Tuple{Symbol})

Compute the permutation required to get `keys` to the ordering declared by `@kwcall`
"""
function kwcallperm(f, keys)
    sortedkeys = Tuple(sort(collect(keys)))

    # The permutation to get from sorted order to the preferred order
    π = baseperm[(f, sortedkeys)]

    # The permutation to get from the call order to the sorted order
    σ = sortperm(collect(keys))

    # Composing the permutations
    return σ[π]
end

# `baseperm[(f, sortedargs::Tuple{Symbol})]` gives the permutation 
# from sorted arguments to the ordering declared with @kwcall 
const baseperm = Dict()

"""
    @kwcall f(b,a,d)

Declares that any call `f(::NamedTuple{N})` with `sort(N) == (:a,:b,:d)`
should be dispatched to the method already defined on `f(::NamedTuple{(:b,:a,:d)})`
"""
macro kwcall(call)
    esc(_kwcall(__module__, call))
end

function _kwcall(m::Module, call)
    @match call begin
        :($f($(args...))) => begin
            π = invperm(sortperm(collect(args)))
            sargs = Tuple(sort(args))
            targs = Tuple(args)
            M = to_type(m)
            quote
                KeywordCalls.baseperm[($f, $sargs)] = $π

                $f(nt::NamedTuple) = kwcall($M, $f, nt)

                $f(; kwargs...) = $f(kwargs.data)

                # $f($(args...)) = $f(NamedTuple{$targs}($(args...)))
            end
        end 
        _ => @error "`@kwcall` declaration must be of the form `@kwcall f(b,a,d)`"
    end
end

"""
    kwcall(f, ::NamedTuple)

Dispatch to the permuted `f(::NamedTuple)` call declared using `@kwcall`
"""
@gg function kwcall(M::Type{<:TypeLevel}, ::Type{F}, nt::NamedTuple{N}) where {F,N}
    π = Tuple(kwcallperm(F, N))
    Nπ = Tuple((N[p] for p in π))
    @under_global from_type(_unwrap_type(M)) quote 
        let M
            v = values(nt)
            valind(n) = @inbounds v[n]
            $F(NamedTuple{$Nπ}(Tuple(valind.($π))))
        end
    end
end

@gg function kwcall(M::Type{<:TypeLevel}, ::F, nt::NamedTuple{N}) where {F<:Function,N}
    f = F.instance
    π = Tuple(kwcallperm(f, N))
    Nπ = Tuple((N[p] for p in π))
    @under_global from_type(_unwrap_type(M)) quote 
        let M
            v = values(nt)
            valind(n) = @inbounds v[n]
            $f(NamedTuple{$Nπ}(Tuple(valind.($π))))
        end
    end
end

end # module

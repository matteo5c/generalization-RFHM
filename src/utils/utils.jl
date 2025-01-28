# Utility functions for Hopfield.jl

# Still havent decided the best way to deal with even elements sums
signb(x::T) where {T} = x == zero(T) ? one(T) : sign(x)
signr(x::T) where {T} = x == zero(T) ? rand([-one(T), one(T)]) : sign(x)

# much faster than for loops, assumes J[i,i] = 0
function energy(J, s)
    return -dot(J * s, s) / (2*length(s))
end 

# given (s0, q) returns a conf. s with overlap q with s0
function close_conf(s0, q)
    N = length(s0)
    s = copy(s0)
    ndiff = round(Int, 0.5 * (1-q) * N)
    for i in randperm(N)[1:ndiff]
        s[i] *= -1
    end
    return s
end

# print info for retrieval exps 
function print_ret_info(s, x; nlines=10)
    m = [abs(dot(x[μ], s) / length(s)) for μ = 1:length(x)]
    idx_m = sortperm(m; rev=true)
    for i = 1:min(length(x), nlines)
        println("#$(idx_m[i]): \t q = $(m[idx_m[i]])")
    end
end
function get_ret_info(s, x)
    m = [abs(dot(x[μ], s) / length(s)) for μ = 1:length(x)]
    idx_m = sortperm(m; rev=true)
    return m, idx_m
end

# Coupling Matrix 'J'
# assumes x is a vector of P vectors with length N
# 'norm' flag should be useful for (highly) correlated data
function create_couplings(x; norm=false)
    N = length(first(x))
    X = hcat(x...)
    if norm # TODO: check if true
        m = mean.(x); X .-= m
    end 
    J = X * transpose(X) ./ N 
    J[diagind(J)] .= 0f0
    return J
end

# ±1 Patterns (iid) 
function create_patterns(N::Int, P::Int; seed=-1)
    seed > 0 && Random.seed!(seed)
    x = [rand([-1f0,1f0], N) for _ = 1:P]
    return x
end
function create_patterns(N::Int, α::T; 
                         seed=-1) where {T <: AbstractFloat}
    P = round(Int, N * α) # P can be even
    x = create_patterns(N, P; seed)
    return x
end

# ±1 Patterns (hmm w/ or wo/ sparse coeff.) 
# L can be int (number of non-zero coeff per patt)
# or float (fraction of non-zero coeff per patt)
function create_patterns(N::Int, D::Int, P::Int; Z=nothing,
                         binary_coeff=false, L=-1, seed=-1, 
                         return_views_X=true, return_views_Z=true)
    seed > 0 && Random.seed!(seed)
    Z===nothing && (Z = rand([-1f0, 1f0], D, N) )
    
    if L > 0
        @assert L <= D
        L == D && @warn "L = D --> standard Hopfield"
        # C = sparse_mat(P, D, L)
        C = sparse_gauss_mat(P, D, L)
    else 
        # C = rand([-1f0, 1f0], P, D)
        C = randn(Float32, P, D)
    end 
    binary_coeff && (C = sign.(C)) # changed this from signr. to sign. to work with sparse_gauss_mat
    X = signb.(C * Z)

    return_views_X ? x=(@views [X[μ,:] for μ = 1:P]) : x=X
    return_views_Z ? z=(@views [Z[k,:] for k = 1:D]) : z=Z    

    return x, z, C
end

function create_patterns(N::Int, d::Union{Int,T}, p::Union{Int,T}; Z=nothing, 
                         binary_coeff=false, 
                         L=-1, seed=-1, 
                         return_views_X=true, return_views_Z=true) where {T <: AbstractFloat}

    P = (typeof(p) <: AbstractFloat ? round(Int, N * p) : p);  
    D = (typeof(d) <: AbstractFloat ? round(Int, N * d) : d);  
    # binary_coeff && (D += iseven(D))
    x, z, C = create_patterns(N, D, P; Z, binary_coeff, L, seed, 
                                    return_views_X, return_views_Z)
    return x, z, C
end 

function sparse_mat(P::Int, D::Int, l::T) where {T <: AbstractFloat}
    C = Matrix{Float32}(rand(P,D) .< l) # fraction of l non-zero
    return C
end
function sparse_mat(P::Int, D::Int, L::Int) 
    C = zeros(Float32, P, D)
    @inbounds for i = 1:P
        idx = randperm(D)[1:L] 
        @views C[i,idx] .= 1f0 
    end
    return C
end
function sparse_gauss_mat(P::Int, D::Int, L::Int) 
    C = zeros(Float32, P, D)
    @inbounds for i = 1:P
        idx = randperm(D)[1:L] 
        @views C[i,idx] .= randn(Float32,L)
    end
    return C
end
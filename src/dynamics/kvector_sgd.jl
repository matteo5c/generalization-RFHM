# Relaxed SGD (k-vector model) on SK-like Ising model

function vec_energy(J, s)
    N = size(J, 1)
    @assert length(s) == N
    E = 0
    @inbounds for i = 1:N
        for j = (i+1):N
            E += J[i,j] * dot(s[i], s[j])
        end
    end
    return -E / N
end

function unit_project(v)
    z = sqrt(sum(abs2, v))
    return v ./ z
end
function unit_project!(v)
    z = sqrt(sum(abs2, v))
    v .= v ./ z
end

function one_dim_projection(s)
    N = length(s)
    K = length(first(s))
    z = unit_project(randn(Float32, K))
    sbin = zeros(Float32, N)
    for i = 1:N
        q = dot(s[i], z)
        sbin[i] = (q > 0 ? 1f0 : -1f0)
    end
    return sbin
end

function init_vecs(N, K)
    s = [unit_project(randn(Float32, K)) for _ = 1:N]
    return s
end

function gradient_step(J, s; lr=0.1)
    N = size(J, 1)
    for i in randperm(N)
        ∇E = zeros(Float32, length(s[i]))
        for j = 1:N 
            j == i && continue
            @. ∇E -= J[i,j] * s[j]
        end
        # @. s[i] -= lr * ∇E / N
        @. s[i] -= lr * ∇E
        unit_project!(s[i])
    end
    return s
end

function mean_norm(s)
    N = length(s)
    z = 0
    for i = 1:N
        z += sqrt(sum(abs2, s[i]))
    end
    return z / N
end

function IPR(v)
    n = length(v)
    return 1.0 / (sum(x -> x^4, v) * n)
end

function ksgd(J::Matrix, K::Int;
               seed=-1, 
               epochs=100, lr=0.1, tol=1e-12, estop=false,
               proj_samples=10,
               verbose=true, more_info=true)

    seed > 0 && Random.seed!(seed)
    N = size(J, 1)

    s = init_vecs(N, K)
    oldE = vec_energy(J, s)
    for epoch = 1:epochs
        s = gradient_step(J, s; lr)
        E  = vec_energy(J, s)
        out = @sprintf("ep=%i \t norm=%g \t E=%g",
                        epoch, mean_norm(s), E)
        verbose  && print(out * "\r")
        estop && (abs((E - oldE) / oldE)) < tol && break
        oldE = E
    end
   
    return one_dim_projection(s)
end # ksgd
#
# MCMC and replicated-MCMC on SK-like Ising model
# the couplings J are assumed to be a symmetric square matrix
#

# TODO: accept func to avoid ops when T=0
function mc(J; s0=nothing,
               sweeps=100, verbose=false,
               seed=-1, β=0, dβ=0)

    seed > 0 && Random.seed!(seed)
    N = size(J, 1) 
    if s0 !== nothing
        @assert length(s0) == N
        s = copy(s0)
    else
        s = rand([-1f0,1f0], N)
    end
    
    conv = false
    it = 0
    h = zeros(Float32, N)
    E = energy(J, s)
    for _ = 1:sweeps
        count = 0
        for i = randperm(N)
            # h[i] = @views dot(J[i,:], s) # !!ATTENZIONE!!
            h[i] = @views dot(J[:,i], s)
            s[i] == signb(h[i]) && (count += 1)
            ΔE = 2 * h[i] * s[i]
            if ΔE <= 0 || (rand() < exp(-β * ΔE))
                s[i] *= -1
                E += ΔE/N 
            end
        end
        it += 1
        conv = (count == N)
        out = @sprintf("it=%i E=%.3f β=%.2E qt=%.2f", 
                        it, E, β, count/N)
        s0 !== nothing && (out *= @sprintf(" q0=%.3f", dot(s0,s)/N))
        verbose && print("\r"*out)
        β += dβ
        (β == Inf || dβ > 0) && conv && break
    end
    return conv, it, s
end

function rmc(J; s0=nothing, 
                sweeps=200, β=Inf, dβ=0, 
                nrep=11, γ=1e-3, dγ=1e-3, p0=0.0,
                verbose=false, seed=-1)

    seed > 0 && Random.seed!(seed)
    N = size(J, 1)
    @assert isodd(nrep)
    if s0 !== nothing
        @assert length(s0) == N
        sr = [copy(s0) for _ = 1:nrep]
    else
        sr = [rand([-1f0,1f0], N) for _ = 1:nrep]
    end
    sc = sign.(mean(sr))

    conv = false
    it = 0
    hr = [zeros(Float32, N) for _ = 1:nrep]
    for _ = 1:sweeps
        for a = 1:nrep
            for i = randperm(N)
                # hr[a][i] = @views dot(J[i,:], sr[a])
                hr[a][i] = @views dot(J[:,i], sr[a])
                ΔE = 2 * sr[a][i] * (hr[a][i] + γ * sc[i])
                if ΔE <= 0 || (rand() < exp(-β * ΔE))
                    sr[a][i] *= -1
                end
            end
        end
        sc = sign.(mean(sr))
        qrep = mean([dot(sr[a], sc) / N for a = 1:nrep])
        conv = (qrep == 1.0)

        E = energy(J, sc)
        out = @sprintf("it=%i γ=%.1E β=%.2E E=%.3f ⟨qrep⟩=%.2f", 
                        it, γ, β, E, qrep)
        verbose && print("\r"*out)

        it += 1
        γ *= (1.0 + dγ)
        β += dβ
        conv && break
    end
    verbose && print("\n")

    return conv, it, sc
end

# Pinning MC (removed for now)
# ...
# ReLU energy MC (removed for now)
# ...
# Tapping MC (removed for now)
# ...
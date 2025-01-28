module Hopfield

using Statistics, Random, LinearAlgebra
using Printf, DelimitedFiles

const RType = Union{Vector, AbstractRange}

include("utils/utils.jl")
include("dynamics/dynamics.jl")

# d and p can be integers (interpreted as D, P)
# or floats (interpreted as αd, α)
# seed is for the dynamics and seedx is for the patterns
function run(N::Int, d, p; qx=0, qz=0, s0=nothing, L=-1, 
             seed=23, seedx=23, algo=:mc, binary_coeff=false,
             # mc kws
             β=Inf, dβ=0, nrep=-1, γ=1e-2, dγ=1e-2, sweeps=100,
             # ksgd kws
             epochs=200, K=2, lr=1.0, 
             verbose=false, more_info=true, save_conf=false, 
             conf_file="tmp.dat", exp_folder="../data/conf/")
    
    @assert algo ∈ [:mc, :rmc, :ksgd]
    save_conf && @assert !isempty(conf_file) # TODO: use file_tmpl()
    
    # create patterns and couplings 
    if d > 0 
        x, z, C = create_patterns(N, d, p; binary_coeff, L, seed=seedx)
    else 
        x = create_patterns(N, p; binary_coeff, seed=seedx)
    end 
    J = create_couplings(x; norm=false)
  
    # init spin conf 
    if s0 === nothing 
        if qx > 0 
            @assert qz <= 0
            s0 = close_conf(x[1], qx)
        elseif qz > 0
            @assert qx == 0; @assert d > 0 
            s0 = close_conf(z[1], qz)
        else 
            @assert qx == 0 && qz <= 0
            s0 = rand([-1f0, 1f0], N)
        end
    end 

    # run dynamics 
    if algo == :mc
        conv, it, s =  mc(J; seed, s0, sweeps, verbose, β, dβ)
    elseif algo == :rmc 
        conv, it, s = rmc(J; seed, s0, sweeps, verbose, β, dβ, 
                             nrep, γ, dγ)
    elseif algo == :ksgd 
        s = ksgd(J, K; seed, epochs, lr, verbose)
    end

    # print more info about patterns/features mag.
    if more_info
        println("\n\nMost correlated patterns:")
        print_ret_info(s, x)
        if d > 0 
            println("\nMost correlated features:")
            print_ret_info(s, z)
        end 
    end

    # save final spin conf.
    if save_conf
        !ispath(exp_folder) && mkpath(exp_folder) 
        @info "Writing conf in $(exp_folder * conf_file)..."
        writedlm(exp_folder * conf_file, s)
    end 

    d > 0 && return s, J, x, z, C
    return s, J, x
end

# drange and prange are range/vectors (both!)
# their elements can be int or floats as in run()
function run(N::Int, drange::RType, prange::RType; 
             outfile="", print_info=true,  
             args...)

    !isempty(outfile) && (f = open(outfile, "w"))
    info_out = @sprintf("| N | αd | α | mx | mx1 |")
    all(d -> d > 0, drange) && (info_out *= @sprintf(" mz | mz1 |"))
    print_info && println(info_out)

    for p in prange, d in drange
        if d > 0   
            s, J, x, z, C = run(N, d, p; args...)
            D = length(z)
            mz = maximum([abs(dot(z[k], s) / N) for k = 1:D])
            # mz1 = abs(dot(z[1], s) / N) 
            mz1 = dot(z[1], s) / N 
        else
            s, J, x = run(N, d, p; args...)
        end 
        P = length(x)
        mx = maximum([abs(dot(x[μ], s) / N) for μ = 1:P])
        # mx1 = abs(dot(x[1], s) / N) 
        mx1 = dot(x[1], s) / N 
        out = @sprintf("%i %g %g %g %g", 
                        N, D/N, P/N, mx, mx1)
        d > 0 && (out *= @sprintf(" %g %g", mz, mz1))
        !isempty(outfile) && println(f, out)
        print_info && println(out)
    end 
    !isempty(outfile) && close(f)
end 

end # module

## lunch this script with:
## julia -p <NCORES> prun_sparse_gen_manyL.jl

using Distributed
using DelimitedFiles
############################################################################

global outdir = "../results"
mkpath(outdir)

outname = "output_sparse_gen_manyL.txt"
outfile = joinpath(outdir,outname)

############################################################################
@everywhere begin

    include("../src/Hopfield.jl"); H=Hopfield

    using Random, Statistics, LinearAlgebra, DelimitedFiles
    const RType = Union{Vector, AbstractRange}

    N_list=[8000]
    d_list=[0.005]
    p_list=[10^a for a in -2.0:0.05:0.0]
    L_list=[3] ## this is L_train
    nsamples = 10
    sample_index_list = collect(1:nsamples)

    ############################################################################
    function local_stability(s0, J)
        conv, it, s = H.mc(J; s0=s0,
                            seed=-1, sweeps=500, 
                            verbose=false, β=Inf, dβ=0.0)

        return dot(s, s0) / length(s)
    end

    ############################################################################
    function run_experiment(param_set)
        N, d, p, L_train, sample_index = param_set

        P = Int(round(p*N))
        P_test = 1
        use_P_examples = min(P,P_test)

        x_train, Z, C = H.create_patterns(N::Int, d, p; 
                                    binary_coeff=true, L=L_train, seed=-1,
                                    return_views_X=true, return_views_Z=false)
                                                        ## ^ this is the place where I need a matrix for Z
                                                        ## because I pass Z to H.create_patterns

        J = H.create_couplings(x_train; norm=false)

        x_test_L3, Z, C = H.create_patterns(N::Int, d, P_test; Z=Z, 
                                        binary_coeff=true, L=3, seed=-1,
                                        return_views_X=true, return_views_Z=false)
        x_test_L5, Z, C = H.create_patterns(N::Int, d, P_test; Z=Z, 
                                        binary_coeff=true, L=5, seed=-1,
                                        return_views_X=true, return_views_Z=false)
        x_test_L7, Z, C = H.create_patterns(N::Int, d, P_test; Z=Z, 
                                        binary_coeff=true, L=7, seed=-1,
                                        return_views_X=true, return_views_Z=false)              
        #################################################
        m_train   = 0
        m_test_L3 = 0
        m_test_L5 = 0
        m_test_L7 = 0
        mu        = 0

        for ν in 1:use_P_examples
            m_train   += local_stability(x_train[ν]  ,J)/use_P_examples
            m_test_L3 += local_stability(x_test_L3[ν],J)/use_P_examples
            m_test_L5 += local_stability(x_test_L5[ν],J)/use_P_examples
            m_test_L7 += local_stability(x_test_L7[ν],J)/use_P_examples
            mu        += local_stability(Z[ν,:],      J)/use_P_examples 
        end
 
        println("$N $d $p $L_train $sample_index $mu $m_train $m_test_L3 $m_test_L5 $m_test_L7")

        return N, d, p, L_train, sample_index, mu, m_train, m_test_L3, m_test_L5, m_test_L7
    end
    ############################################################################
    param_set_list = [(N, d, p, L, index)  for index in collect(1:nsamples)
                                        for N in N_list
                                        for d in d_list
                                        for L in L_list  
                                        for p in p_list  
                                        ]
    ############################################################################
end

############################################################################

output = pmap(run_experiment, param_set_list)

writedlm(outfile, output)

############################################################################
include("saddle_point_equations_RF_mixtures.jl")

αD=1e-8:1e-6:0.035
n = 3 # numbers of features in the mixture

α= 30:-1:10
P.find_alphac(α=α, αD=αD, dq = 0.5, qh = 0.2, 
                dp = 0.5, dph = 0.2, 
                pd = 0.5, dpdh = 0.2, 
                m = 0.5, mh = 0.3, n = n,
                σ=:sign, ψ=0.7, verb = 0
                )

α= 9.5:-0.5:1
P.find_alphac(α=α, αD=αD, dq = 0.5, qh = 0.2, 
                dp = 0.5, dph = 0.2, 
                pd = 0.5, dpdh = 0.2, 
                m = 0.5, mh = 0.3, n = n,
                σ=:sign, ψ=0.7, verb = 0
                )

α= 0.95:-0.05:0.1
P.find_alphac(α=α, αD=αD, dq = 0.5, qh = 0.2, 
                dp = 0.5, dph = 0.2, 
                pd = 0.5, dpdh = 0.2, 
                m = 0.5, mh = 0.3, n = n,
                σ=:sign, ψ=0.7, verb = 0
                )

α= 0.095:-0.005:0.01
P.find_alphac(α=α, αD=αD, dq = 0.5, qh = 0.2, 
                dp = 0.5, dph = 0.2, 
                pd = 0.5, dpdh = 0.2, 
                m = 0.5, mh = 0.3, n = n,
                σ=:sign, ψ=0.7, verb = 0
                )

α= 0.0095:-0.0005:0.001
P.find_alphac(α=α, αD=αD, dq = 0.5, qh = 0.2, 
                dp = 0.5, dph = 0.2, 
                pd = 0.5, dpdh = 0.2, 
                m = 0.5, mh = 0.3, n = n,
                σ=:sign, ψ=0.7, verb = 0
                )

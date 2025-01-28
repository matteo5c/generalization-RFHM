#%%
import numpy as np
import matplotlib.pyplot as plt
##################################################

def extract_results(data, filter_col, filter_val, 
                          x_col, y_col,
                          N=2000,
                          L_filter_val=3, L_filter_col=3,
                          alpha_lim=5.0):
    
    ## filter for the value of N
    mask_N = data[:,0] == N
    data_N = data[mask_N, :]

    ## filter for the value of L
    mask_L = data_N[:,L_filter_col] == L_filter_val
    data_L = data_N[mask_L, :]

    ## filter for the value of alpha
    mask_a = data_L[:,2] < alpha_lim
    data_a = data_L[mask_a, :]    

    ## filter for a custom value of anything
    mask_1 = data_a[:,filter_col] == filter_val
    data_1 = data_a[mask_1, :]

    x_samples = data_1[:,x_col]
    y_samples = data_1[:,y_col]

    x_values, x_counts  = np.unique(x_samples, return_counts=True)
    nx = x_values.shape[0]
    y_values = np.zeros(nx)
    y_err = np.zeros(nx)

    for i in range(nx):
        mask = x_samples == x_values[i]
        y_values[i] = np.mean(y_samples[mask])
        y_err[i] = np.std(y_samples[mask])/np.sqrt(x_counts[i])
    
    # print(x_counts[i])

    return x_values, y_values, y_err

##################################################
##################################################

plt.figure(figsize=(6,3))
# N, d, p, L, sample_index, m_train, m_test, mu
# 0  1  2  3  4             5        6       7

# N, d, p, L_train, sample_index, mu, m_train, m_test_L3, m_test_L5, m_test_L7
# 0  1  2  3        4             5   6        7          8          9

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
# CHECK N SCALING OF GENERALIZATION TRANSITION (TRAIN)
plt.subplot(1,1,1)
data = np.loadtxt("../results/output_sparse_gen_manyL.txt")
N=8000
alpha_lim=3.5
for i,aD in enumerate([0.005]):

    x_values, y_values, y_err = extract_results(data, 1, aD, 2, 5, N=N, alpha_lim=alpha_lim)
    plt.plot(x_values, y_values, label=r"$f$")
    plt.fill_between(x_values, y_values-y_err, y_values+y_err, alpha=0.3)

    x_values, y_values, y_err = extract_results(data, 1, aD, 2, 6, N=N, alpha_lim=alpha_lim)
    plt.plot(x_values, y_values, label=r"$\xi_\mathrm{train}$ (L=3)")
    plt.fill_between(x_values, y_values-y_err, y_values+y_err, alpha=0.3)

    x_values, y_values, y_err = extract_results(data, 1, aD, 2, 7, N=N, alpha_lim=alpha_lim)
    plt.plot(x_values, y_values, label=r"$\xi_\mathrm{test}$ (L=3)")
    plt.fill_between(x_values, y_values-y_err, y_values+y_err, alpha=0.3)

    x_values, y_values, y_err = extract_results(data, 1, aD, 2, 8, N=N, alpha_lim=alpha_lim)
    plt.plot(x_values, y_values, label=r"$\xi_\mathrm{test}$ (L=5)")
    plt.fill_between(x_values, y_values-y_err, y_values+y_err, alpha=0.3)

    x_values, y_values, y_err = extract_results(data, 1, aD, 2, 9, N=N, alpha_lim=alpha_lim)
    plt.plot(x_values, y_values, label=r"$\xi_\mathrm{test}$ (L=7)")
    plt.fill_between(x_values, y_values-y_err, y_values+y_err, alpha=0.3)

plt.xscale("log")
# plt.xlim(left=-0.1, right=3.1)
plt.xlabel(r"model capacity $\alpha$")
plt.ylabel(r"magnetization $m$")

plt.legend(title=r"$\alpha_D=$"+f"{aD}")
####################################################################################################
####################################################################################################

##################################################
plt.tight_layout()

plt.savefig("fig_sparse_gen_manyL.pdf")

# %%

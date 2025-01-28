import numpy as np
import matplotlib.pyplot as plt

filenames = [
"spinodal_n=3.txt",
"spinodal_n=5.txt",
"spinodal_n=7.txt",
"spinodal_n=9.txt",
]

labels=[
    "L=3",
    "L=5",
    "L=7",
    "L=9",
]

plt.figure(figsize=(4,3))

for i,filename in enumerate(filenames):
    data = np.loadtxt(filename, usecols=(0,1),comments="#")
    plt.plot(data[:,1],data[:,0], label=labels[i])

plt.xlabel(r"$\alpha_D=D/N$")
plt.ylabel(r"$\alpha=P/N$")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("spinodal_mixtures.pdf")
# Generalization in Random-Features Hopfield Model 

Code needed to reproduce the results of the paper
_"Random Features Hopfield Networks generalize retrieval to previously unseen examples"_ [[arxiv](https://arxiv.org/abs/2407.05658)]

This project contains three folders.
- `src` contains the code to run a zero-temperature asynchronous dynamics on a Random-Features Hopfield Model and measure magnetisations. 
- `run` contains scripts to run the dynamics and reproduce the numerical results of the manuscript.
- `saddle_point` contains the codes to solve the saddle point equations described in the manuscript.

## Reproduce Numerical Results

The folder `run` contains a julia script to reproduce the basic simulation of the paper, where we measure magnetization as funcion of $\alpha$ for features, train examples, and test examples for various values of $L_\mathrm{test}$. To run the simulation, type in the terminal:
```bash
cd run
julia -p $NCORES prun_sparse_gen_manyL.jl
```
where `NCORES` is the number of cores to use in parallel to cycle on the parameter lists. The parameter lists are defined inside `prun_sparse_gen_manyL.jl` and are preset to meaningful values. Increase `N` and `nsamples` for better precision. This script produces an output file `output_sparse_gen_manyL.txt` inside the directory `results`. 

To plot the results:
```bash
cd plot
python plot_prun_sparse_gen_manyL.py
```
which produces `fig_sparse_gen_manyL.pdf` in the directory `plot`. Notice that the script `plot_prun_sparse_gen_manyL.py` filters the data for lines with specific values of $N$ and $\alpha_D$, so those need to be changed accordingly if they are changed in `prun_sparse_gen_manyL.jl`.

## Saddle-point solver

There is one folder for the equations of the leaning phase, called "factor", and one folder for the storage phase, called "pattern". Their structure is the same and is explained below.

The file `saddle_point_equations_RF_mixtures.jl` contains the definition of the saddle-point equations and the essential functions to solve them, in a module called `P`. The functions relevant to the user are the following: 
- `converge(...)` takes in input values $\alpha$, $\alpha_D$, and an initial condition of the order parameters and finds a fixed point by iterating the saddle-point equations.
- `span(...)` runs `converge(...)` for a given interval of $\alpha$ or $\alpha_D$, using the fixed point at the previous run as the initial condition of the next one. It prints all the fixed points on a file.
- `find_alphac(...)` runs `converge(...)` for a given interval of $\alpha$ or $\alpha_D$ and prints to a file only when the magnetisation changes abruptly, signaling the phase transition.



The folder includes the script `findalphac.jl` reproduce the spinodal line of a mixture of a given order $n$. The script is designed to be used within the julia REPL. It should be run as follows:
```julia
julia> include("findalphac.jl");
```
`findalphac.jl` produces output files called `spinodal_n=$n.txt` for `n=3,5,7,9`, which can be plotted with `plot_spinodal.py`. The output files and the plot are already included. 


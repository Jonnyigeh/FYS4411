# RBM specifics
n_visible_nodes = 12  # number of particle spins
n_hidden_nodes = 12  # number of hidden spins
temperature = 1.0

# RNG specifics
seed = 0

# MCMC specifics
n_samples = int(2  ** 14)
n_procs = 1 # how many subprocess
sampling_method = "metro" # Metro-hastings does not make sense for ising system

# Hamiltonian specifics
J = 1.0
mu = 1.0
nv = 3.0

# Optimizer specifics
n_cycles = 25
learning_rate = 0.1

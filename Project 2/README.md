### Repository for project 2 in FYS4411

## Description
In this project we've developed a functioning Restricted Boltzmann Machine (RBM) that solves the ground state problem using the variational method
of the 1D Ising Hamiltonian. This system is a spin-lattice chain with periodic boundary conditions, which a known analytical solution exists.

## Source Code
The program is easily ran using the main.py script, found in src/main.pdf - with helper functions located in src/utils, and the program is changed by changing the parameters in src/utils/config.py. At the moment (12.06.24) the parallelisation is faulty, and should be avoided.

All dependencies are easily managed in the pyproject.toml file - and it is our recommendation that this is built using Poetry. To do so, one can clone the repository and run >> poetry install while in the src/ folder. 
For users without Poetry, dependencies must be installed manually (but they are easily found in the .toml file).

## Report
The report can be found in full in doc/main.pdf, and all sections/figures/references are also found in the doc folder

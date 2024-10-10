# libs
import numpy as np
import matplotlib.pyplot as plt

# project files
import utils.config as config
from utils.variational_state import RBM
from utils.sampler import Metropolis
from utils.hamiltonian_helper import onedim_ising_model
from utils.optimizer import sgd


def visualize(y, tag):
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel("Iterations")
    plt.ylabel(tag)
    plt.title(tag)
    plt.show()


if __name__ == "__main__":
    import pickle

    # Set the random seed
    np.random.seed(config.seed)
    # Set the state of the RBM
    state = RBM(
        n_visible=config.n_visible_nodes,
        n_hidden=config.n_hidden_nodes,
        temperature=config.temperature,
    )

    # Set the hamiltonian for our system
    hamiltonian = onedim_ising_model(
        nparticles=config.n_visible_nodes, J=config.J, mu=config.mu
    )
    # breakpoint()

    # Choose, and initialize sampler
    sampler = Metropolis(n_procs=config.n_procs, hamiltonian=hamiltonian)

    # Initialize the optimizer
    sgd_optimizer = sgd(
        state=state,
        n_cycles=config.n_cycles,
        sampler=sampler,
        learning_rate=config.learning_rate,
    )

    # Run sampling process
    # res, *things = sampler.sampler(nsteps=config.n_samples,
    #                 state=state,
    #                 )

    # Run the optimization
    # le, var, stde, res, spin_configs = sgd_optimizer.run_optimizer()
    # visualize(le, 'Local Energy')
    # breakpoint()

    le = {}
    var = {}
    stde = {}
    res = {}
    for nv in [3.0, 5.0, 10, 12]:
        config.nv = nv
        state.reset_params()
        (
            le["n_v = " + str(nv)],
            var["T= " + str(nv)],
            stde["T= " + str(nv)],
            res["T= " + str(nv)],
            *_,
        ) = sgd_optimizer.run_optimizer()

    breakpoint()
    # breakpoint()
    # Save the runs
    # with open("runs/le_12nodes_deltanv.pkl", "wb") as f:
    #     pickle.dump(le, f)

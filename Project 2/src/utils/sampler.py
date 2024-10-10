# libs
import numpy as np
from tqdm import tqdm
from joblib import delayed
from joblib import Parallel  # instead of from pathos.pools import ProcessPool

# proj files
import utils.config as config
from utils.some_utils import generate_seed_sequence


class Sampler:
    def __init__(self, hamiltonian, n_procs=1):
        """
        Parent class for a sampler object

        """
        self.n_procs = n_procs
        self.hamiltonian = hamiltonian
        # Set the sampler to be the multiproc_sampler if n_procs > 1
        if self.n_procs > 1:
            self.sampler = self.multiproc_sampler

        # Generate seeds for the PRNG
        self.seeds = generate_seed_sequence(user_seed=config.seed, pool_size=n_procs)

    def sampler(self, nsteps, state, squared_wavefunction=None):
        """
        Helper function that runs the sampler, whatever it may be
        """
        res, energies, spin_configs, E_grad_a, E_grad_b, E_grad_W = self._sampler(
            nsteps=nsteps, state=state, squared_wavefunction=squared_wavefunction
        )

        return res, energies, spin_configs, E_grad_a, E_grad_b, E_grad_W

    def multiproc_sampler(self, nsteps, state, squared_wavefunction=None):
        """
        Multi-proc sampling
        """
        nsteps = [
            nsteps
        ] * self.n_procs  # Distribute the number of steps to each process
        procs_id = list(
            range(self.n_procs)
        )  # This is necessary for nice output of progress bars
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        def compute(i):
            return self._sampler(
                nsteps=nsteps[i], state=state, seed=self.seeds[i], procs_id=procs_id[i]
            )

        output = Parallel(n_jobs=-1, backend="loky")(
            delayed(compute)(i) for i in range(self.n_procs)
        )
        res_tuple, energies, spin_configs, E_grad_a, E_grad_b, E_grad_W = zip(*output)
        # Average the results from the different processes, and collect
        E_grad_a = np.mean(np.array(E_grad_a), axis=0)
        E_grad_b = np.mean(np.array(E_grad_b), axis=0)
        E_grad_W = np.mean(np.array(E_grad_W), axis=0)
        energies = np.mean(np.array(energies), axis=0)

        # Create a new res dict, averaging from all procs
        accepted_steps = sum(
            res_tuple[i]["accepted_swaps"] for i in range(config.n_procs)
        )
        std_err = np.mean(
            np.array([res_tuple[i]["std_error"] for i in range(config.n_procs)])
        )
        var = np.mean(
            np.array([res_tuple[i]["variance"] for i in range(config.n_procs)])
        )
        e_loc = np.mean(
            np.array([res_tuple[i]["local_energy"] for i in range(config.n_procs)])
        )
        res = {
            "acceptance_rate": accepted_steps / (config.n_procs * config.n_samples),
            "std_error": std_err,  # .item(),
            "variance": var,  # .item(),
            "local_energy": e_loc,  # .item()
        }

        return res, energies, spin_configs, E_grad_a, E_grad_b, E_grad_W

    def _sampler(self):
        """
        Actual sampling function. To be implemented by childrenclass
        """
        raise NotImplementedError

    def generate_new_spin_config(self, size_visible, visible_layer):
        """
        Generate a new configuration of the spins in the visible layer.
            - Here a single (randomly chosen) spin in the visible layer is flipped (by multiplication of a factor -1), and returned.
        """
        n_spins = size_visible
        flipped_idx = np.random.randint(0, n_spins)
        bool_array = np.arange(n_spins) == flipped_idx
        flipped_visible_layer = np.where(bool_array, -visible_layer, visible_layer)

        return flipped_visible_layer


class Metropolis(Sampler):
    def __init__(self, hamiltonian, n_procs=1):
        """
        Metropolis sampler
        """
        super().__init__(n_procs=n_procs, hamiltonian=hamiltonian)
        self.tmp_energy = 1000

    def _sampler(
        self, nsteps, state, seed=config.seed, procs_id=0, squared_wavefunction=None
    ):
        """
        Sample the wavefunction using the Metropolis algorithm

        args:
            nsteps: int - number of steps in the sampling process
            state: class object - state of the system (visible and hidden layers in the RBM, etc.)
            wavefunction: func - the marginal probability of the Boltzmann machine (i.e the wavefunction ansatz)
        """
        # Set the seed for the PRNG
        np.random.seed(seed)
        # Set the wavefunction to the marginal probability if not specified
        if squared_wavefunction is None:
            squared_wavefunction = state.marginal_probability_squared
        initial_configuration = (
            state.spin_configuration
        )  # Extract the initial configuration, should this be the visible/hidden layer spins?

        # Set up lists for storing values needed for the gradients of the RBM params
        energies = []
        spin_configs = []
        grad_a = []
        grad_b = []
        grad_W = []
        prod_term_a = []
        prod_term_b = []
        prod_term_W = []
        # Reset the number of accepted swaps
        self.n_accepted = 0
        for step in range(nsteps):
            # Calculate the energy of the initial configuration
            initial_wf_probability = squared_wavefunction(initial_configuration)
            # Find a new configuration by flipping a random spin
            new_configuration = self.generate_new_spin_config(
                size_visible=state.n_visible, visible_layer=initial_configuration
            )
            # Calculate the energy of the new configuration
            new_wf_probability = squared_wavefunction(new_configuration)
            # Calculate the acceptance probability for the new configuration
            acceptance_probability = min(
                1, (new_wf_probability / initial_wf_probability) ** 2
            )  # Acceptance probability according to C6 in Carleo and Troyer 1606.02318v1
            # Accept or reject the new configuration, we can here use regular numpy randomness
            if np.random.random() < acceptance_probability:
                self.n_accepted += 1  # To calculate acceptance rate
                initial_configuration = new_configuration
                spin_configs.append(initial_configuration)  # might not be needed?

            # Find the samples needed for the expectation values to be used in calculating the gradients
            grad_ai, grad_bi, grad_Wij = state.grad_marginal_probability(
                initial_configuration
            )
            E_loc_estimate = self.hamiltonian.local_energy_long_int(
                initial_configuration, n_v=config.nv
            )

            energies.append(E_loc_estimate)
            grad_a.append(grad_ai)
            grad_b.append(grad_bi)
            grad_W.append(grad_Wij)
            prod_term_a.append(grad_ai * E_loc_estimate)
            prod_term_b.append(grad_bi * E_loc_estimate)
            prod_term_W.append(grad_Wij * E_loc_estimate)

        E_grad_a = 2 * (
            np.mean(prod_term_a, axis=0) - np.mean(energies) * np.mean(grad_a, axis=0)
        )
        E_grad_b = 2 * (
            np.mean(prod_term_b, axis=0) - np.mean(energies) * np.mean(grad_b, axis=0)
        )
        E_grad_W = 2 * (
            np.mean(prod_term_W, axis=0) - np.mean(energies) * np.mean(grad_W, axis=0)
        )
        # Update the state with the new configuration
        state.spin_configuration = initial_configuration  # TODO: Check with multiproc
        # Calculate statistical properties, and expectation value of local energy
        self.acceptance_rate = self.n_accepted / nsteps
        energies = np.array(energies)
        std_error = np.std(energies) / np.sqrt(nsteps)
        variance = np.var(energies)
        energy = np.mean(energies)
        res = {
            "acceptance_rate": self.acceptance_rate,
            "accepted_swaps": self.n_accepted,
            "std_error": std_error.item(),
            "variance": variance.item(),
            "local_energy": energy.item(),
        }

        return res, energies, spin_configs, E_grad_a, E_grad_b, E_grad_W

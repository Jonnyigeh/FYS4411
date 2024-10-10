# libs
import numpy as np
from tqdm import tqdm

# proj files
import utils.config as config


class Optimizer:
    """
    Class for the optimization of the variational parameters in the RBM
        - These parameters are the weights and biases, and there are multiple ways to train these.
    """

    def __init__(self, state, n_cycles, sampler):
        """
        Initialize the optimizer class object
        """
        # TODO: should probably put some code here
        self.state = state
        self.n_cycles = n_cycles
        self.sampler = sampler

    def run_optimizer(
        self,
    ):
        """
        Run the optimization scheme
        """

        raise NotImplementedError


class sgd(Optimizer):
    def __init__(self, state, n_cycles, sampler, learning_rate=0.01):
        super().__init__(state=state, n_cycles=n_cycles, sampler=sampler)
        self.learning_rate = learning_rate

    def update_parameters(self, W_grad, a_grad, b_grad):
        """
        Update the parameters of the RBM
        """
        W, a, b = self.state.params
        W -= self.learning_rate * W_grad
        a -= self.learning_rate * a_grad
        b -= self.learning_rate * b_grad

        return W, a, b

    def run_optimizer(self):
        """
        Run the SGD optimization scheme
        """
        print("Running optimization...")

        E_loc = 0
        # List made for plotting purposes
        local_energies = []
        variances = []
        std_errors = []
        resss = []
        spin_config = []
        with tqdm(
            total=self.n_cycles,
            desc=rf"[Optimization progress, E = {E_loc:.4f}]",
            position=0,
            colour="green",
            leave=True,
        ) as pbar:
            for n in range(self.n_cycles):
                # Run the sampler, and extract all the sampled values for the various terms.
                # They will have length equal to n_samples, and the mean will be the corresponding expectation value (of said terms)
                res, energies, spin_configs, E_grad_a, E_grad_b, E_grad_W = (
                    self.sampler.sampler(
                        nsteps=config.n_samples,
                        state=self.state,
                        squared_wavefunction=self.state.marginal_probability_squared,
                    )
                )

                # Find the updated parameters
                W, a, b = self.update_parameters(
                    W_grad=np.array(E_grad_W),
                    a_grad=np.array(E_grad_a),
                    b_grad=np.array(E_grad_b),
                )

                # Set the updated parameters in the RBM state
                self.state.params = (W, a, b)
                E_loc = res["local_energy"]
                resss.append(res)
                spin_config.append(spin_configs)
                pbar.set_description(rf"[Optimization progress, E = {E_loc:.4e}]")
                pbar.update(1)
                local_energies.append(E_loc)
                variances.append(res["variance"])
                std_errors.append(res["std_error"])

        print("Optimization done!")

        return local_energies, variances, std_errors, resss, spin_config

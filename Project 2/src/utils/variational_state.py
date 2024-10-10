import jax
import jax.numpy as np
import numpy as np


class RBM:
    """
    Class for the variational state
     - In this restricted Boltzmann machine we do not explicitly evaluate the partition function,
        as this, for all our purposes, is a constant factor that will cancel (or can be absorbed in other constants during training).
    """

    def __init__(self, n_visible, n_hidden, temperature=1.0):
        """
        Initializes the Restricted Boltzmann Machine
        """
        # Initialize the weights and biases, and the number of visible and hidden units
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # According to Carleo & Troyer, the weights are to be complex-valued, however this is not required for our purpose.
        W = 0.1 * np.random.randn(
            n_visible, n_hidden
        )
        a = 0.1 * np.random.randn(n_visible)
        b = 0.1 * np.random.randn(n_hidden)
        self.T = temperature

        # Initialize the visible and hidden layers
        visible = np.random.choice([-1, 1], size=n_visible, p=[0.5, 0.5])
        hidden = np.random.choice([-1, 1], size=n_hidden)

        # Save the initial parameters to be able to reset the parameters
        self.Wcopy = W.copy()
        self.acopy = a.copy()
        self.bcopy = b.copy()
        self.visiblecopy = visible.copy()

        # Convert to numpy arrays TODO: remove
        self._W = np.array(W)
        self._a = np.array(a)
        self._b = np.array(b)
        self._visible = np.array(visible)
        self._hidden = np.array(hidden)

    def reset_params(self):
        """
        Reset the parameters of the RBM
        """
        self._W = self.Wcopy.copy()
        self._a = self.acopy.copy()
        self._b = self.bcopy.copy()
        self._visible = self.visiblecopy.copy()

    def magnetization(self):
        """
        Compute the magnetization of the system
        """
        return np.sum(self._visible)

    def __call__(
        self,
        visible,
    ):
        """
        Evaluate the wavefunction (RBM) given the input (current spin configuration) to the visible layer
        TODO: Remove this function?
        """
        self._visible = visible
        self.forward_pass()
        self.backward_pass()
        self._energy = self.energy_function()

        return self._energy

    def forward_pass(self):
        """
        Forward pass of the RBM
        """
        input_hidden_layer = self.W @ self.visible + self.a  # Should it be v^T @ W + a
        self.hidden = self.sigmoid(input_hidden_layer)

    def backward_pass(self):
        """
        Backward pass of the RBM
        """
        input_visible_layer = (
            self.W.T @ self.hidden + self.b
        )  # Should it be h @ W^T + b
        self.visible = self.sigmoid(input_visible_layer)

    def sigmoid(self, x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def log_energy_function(self):
        """
        Evaluate the logarithm of the energy function
        """
        interaction_term = -self.visible.T @ self.W @ self.hidden
        visible_term = -self.a.T @ self.visible
        hidden_term = -self.b.T @ self.hidden

        return interaction_term + visible_term + hidden_term

    def energy_function(self):
        """
        Evaluate the energy function
        """
        interaction_term = -self._visible.T @ self._W @ self._hidden
        visible_term = -self._a.T @ self._visible
        hidden_term = -self._b.T @ self._hidden

        return np.exp(
            interaction_term + visible_term + hidden_term
        )  # Maybe use np.exp instead of np.exp?

    def log_marginal_probability(self, visible=None):
        """
        Compute the marginal probability of the visible layer (this is what represents the wavefunction)
        """
        if visible is None:
            P_v = (
                (-self._a.T @ self._visible)
                * 2
                * np.log(np.sum(np.cosh(self._b + self._W.T @ self._visible), axis=0))
            )
        else:
            P_v = (
                (-self._a.T @ visible)
                * 2
                * np.log(np.sum(np.cosh(self._b + self._W.T @ visible), axis=0))
            )

        return P_v

    def log_marginal_probability_squared(self, visible=None):
        """
        Compute the squared marginal probability of the visible layer (this is what represents the wavefunction probability distribution)
        """
        P_v = self.log_marginal_probability(visible=visible)

        return 2 * P_v

    def marginal_probability(self, visible=None):
        """
        Compute the marginal probability of the visible layer (this is what represents the wavefunction)
        """
        if visible is None:
            P_v = np.exp(1 / self.T * self._a.T @ self._visible) * np.prod(
                2 * np.cosh(1 / self.T * (self._b + self._W.T @ self._visible))
            )
        else:
            P_v = np.exp(1 / self.T * self._a.T @ visible) * np.prod(
                2 * np.cosh(1 / self.T * (self._b + self._W.T @ visible))
            )

        return P_v

    def marginal_probability_squared(self, visible=None):
        """
        Compute the squared marginal probability of the visible layer (this is what represents the wavefunction probability distribution)
        """
        P_v = self.marginal_probability(visible=visible)

        return np.abs(P_v) ** 2

    def grad_marginal_probability(self, visible=None):
        """
        Compute the gradient of the marginal probability, divided by the marginal probability
        These are according to Carleo & Troyer the gradients of the wavefunction, which in turn, will be used
        together with the local energy to compute the gradients of the RBM parameters.
        """
        if visible is None:
            visible = self._visible
        grad_a = visible / self.T
        grad_b = np.tanh(1 / self.T * (self._b + self._W.T @ visible))
        grad_W = np.outer(
            visible / self.T, np.tanh(1 / self.T * (self._b + self._W.T @ visible))
        )

        return grad_a, grad_b, grad_W

    def joint_probability(self):
        """
        Compute the joint probability of the visible and hidden layers
         as the energy function normalized by the partition function
        """
        P_rbm = self.energy_function()
        return P_rbm

    @property
    def hidden_layer(self):
        """
        Return the hidden layer
        """
        return self._hidden

    @property
    def params(self):
        """
        Return the parameters of the RBM
        """
        return self._W, self._a, self._b

    @params.setter
    def params(self, new_params):
        """
        Set the parameters of the RBM
        """
        self._W, self._a, self._b = new_params

    @property
    def spin_configuration(self):
        """
        Return the spin configuration
        """
        return self._visible

    @spin_configuration.setter
    def spin_configuration(self, configuration):
        """
        Set the spin configuration
        """
        self._visible = configuration

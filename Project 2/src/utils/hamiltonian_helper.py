import numpy as np

class hamiltonian:
    """
    Helper-class for a Hamiltonian object
    """
    def __init__(self,
                 nparticles,
                 ):
        """
        args:
            nparticles: int - number of particles
        """
        self.nparticles = nparticles
        self._energy = 0
        

    def local_energy(self,
                     wavefunction,
                     positions,
                 ):
        """
        Evaluate the Hamiltonian, and find local energy of system
         - overwritten by the child class
        """
        raise NotImplementedError
    
    @property
    def energy(self):
        """
        Return the energy of the system
        """
        return self._energy


class onedim_ising_model(hamiltonian):
    """
    Hamiltonian for the Ising model with a temperature of \beta = 1.0
    """
    def __init__(self,
                 nparticles,
                 J,
                 mu,
                 ):
        """
        args:
            nparticles: int - number of particles
            dimension: int - dimension of the system
            J: float - coupling constant
            mu: float - magnetic field
        """
        super().__init__(nparticles)
        self.J = J
        self.mu = mu

    def local_energy_long_int(self,
                              spin_configuration,
                              n_v = 5):
        """
        Evaluate the long-range interaction Hamiltonian, and find local energy

        args:
            state: class instance - the RBM class instance that represents the variational state
            spin_configuration: np.array - the current spin configuration of the state class
        
        """
        # Make the spin configuration into a periodic 1d array (we assume periodic boundary conditions)
        # That allows for the calculation of the long-range interaction
        # periodic_spin_configuration = np.append(spin_configuration, spin_configuration[0])
        # Calculate the energy by evaluating the Hamiltonian
        energy = 0
        def Jij(i, j, alpha = 1.0, n_v = 5.0):
            """The long-range interaction coupling constant between spins i and j in the lattice
            args:
                i: int - index of the first spin
                j: int - index of the second spin
            """
            dist = np.abs(i-j)
            if dist > n_v:
                return 0
            
            return self.J / (dist ** alpha)
        
        for i in range(len(spin_configuration)):
            energy += -self.mu*spin_configuration[i]
            for j in range(i+1, len(spin_configuration)):
                energy += -Jij(i, j, n_v=n_v) * spin_configuration[i] * spin_configuration[j]

        return energy


    def local_energy(self,
                     spin_configuration):
        """
        Evaluate the Hamiltonian, and find local energy of system
        
        args:
            state: class instance - the RBM class instance that represents the variational state
        """
        # Make the spin configuration into a periodic 1d array (we assume periodic boundary conditions)
        spin_configuration = np.append(spin_configuration, spin_configuration[0])
        # Calculate the energy by evaluating the Hamiltonian
        energy = 0
        
        for i in range(len(spin_configuration)-1):
            energy += -self.J*spin_configuration[i]*spin_configuration[i+1] - self.mu*spin_configuration[i]
        
        self._energy = energy

        return energy

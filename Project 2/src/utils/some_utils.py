import numpy as np


def generate_seed_sequence(user_seed=None, pool_size=None):
    """Process a user-provided seed and convert it into initial states for
    parallel pool workers.

    Parameters
    ----------
    user_seed : :obj:`int`
        User-provided seed. Default is None.
    pool_size : :obj:`int`
        The number of spawns that will be passed to child processes.

    Returns
    -------
    seeds : :obj:`list`
        Initial states for parallel pool workers.
    """
    seed_sequence = np.random.SeedSequence(user_seed)
    seeds = seed_sequence.spawn(pool_size)
    return seeds

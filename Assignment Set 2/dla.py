import numpy as np
from numba import jit, cuda


def init_diffusion_grid(size):
    """
    """


def run_dla_diff_equation(size, max_iter, omega, eta, run_GPU=False):
    """
    Performs Diffusion Limited Aggregation (DLA) using probabilities
    based on the time-independent diffusion equation (using SOR).
    arguments:
        size (int): The size of the grid.
        max_iter (int): The maximum number of iterations.
        omega (float): The relaxation factor.
        eta (float): The shape parameter for the DLA cluster.
        run_GPU (bool): Whether to run the simulation on the GPU.
    """

    # Initialize the concentration grid (linear gradient from top to bottom)
    c_grid = np.tile(np.linspace(0, 1, size), (size, 1))

    # Place initial seed
    cluster_grid = np.full((size, size), np.nan)
    occupied_indices = np.array([[size//2, 0]])
    cluster_grid[tuple(occupied_indices.T)] = 0
    c_grid = np.where(np.isnan(cluster_grid), c_grid, 0)

    # Construct neighbourhood stencil
    offsets = [[i, j] for i in range(-1, 2) for j in range(-1, 2) if abs(i) != abs(j)]

    # Growth loop
    for n in range(max_iter):
        
        # Determine growth candidates
        growth_candidate_indices = set()
        for offset in offsets:
            
            # Get neighbour indices
            nbrs = occupied_indices + np.tile(offset, (occupied_indices.shape[0], 1))

            # Disregard out-of-bounds neighbours
            mask = np.all(nbrs >= 0, axis=1) & np.all(nbrs < size, axis=1)
            nbrs = nbrs[mask]

            growth_candidate_indices = growth_candidate_indices | set(map(tuple, nbrs))
            growth_candidate_indices -= set(map(tuple, occupied_indices))
        
        # Get candidate concentrations
        growth_candidate_indices = list(growth_candidate_indices)
        c_candidates = c_grid[tuple(np.array(growth_candidate_indices).T)]

        # Determine probabilities
        growth_ps = c_candidates / np.sum(c_candidates)

        # Select growth candidate
        growth_site_index = np.random.choice(len(growth_candidate_indices), p=growth_ps)
        print(occupied_indices)
        print( np.array(growth_candidate_indices[growth_site_index]))
        occupied_indices = np.append(occupied_indices, np.array([growth_candidate_indices[growth_site_index]]), axis=0)
        print(occupied_indices)

    return


def run_dla_random_walk(size):
    """
    Performs Diffusion Limited Aggregation (DLA) using random walks.
    """

    return
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
    cluster_grid[occupied_indices.T] = 0
    print(cluster_grid)
    c_grid = np.where(np.isnan(cluster_grid), c_grid, 0)

    # Growth loop
    for n in range(max_iter):
        pass

    return


def run_dla_random_walk():
    """
    Performs Diffusion Limited Aggregation (DLA) using random walks.
    """

    return
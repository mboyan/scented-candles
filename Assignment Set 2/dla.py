import numpy as np
from numba import jit, cuda


def growth_iteration(c_grid, occupied_indices, eta, offsets):
    """
    Pick a growth site and update the indices of
    sites occupied by the DLA cluster.
    arguments:
        c_grid (ndarray): The concentration grid.
        occupied_indices (ndarray): The indices of the occupied sites.
        eta (float): The shape parameter for the DLA cluster.
        offsets (ndarray): The neighbourhood stencil.
    returns:
        occupied_indices (ndarray): The updated indices of the occupied sites.
    """

    size = c_grid.shape[0]

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
    c_candidates_powered = np.power(c_candidates, eta)
    growth_ps = c_candidates_powered / np.sum(c_candidates_powered)

    # Select growth candidate
    growth_site_index = np.random.choice(len(growth_candidate_indices), p=growth_ps)
    occupied_indices = np.append(occupied_indices, np.array([growth_candidate_indices[growth_site_index]]), axis=0)

    return occupied_indices


@cuda.jit
def diffusion_step_sor_GPU(system_A, system_B, red, cluster_grid, omega):
    """
    A GPU-parallelised Gauss-Seidel iteration function using two interlaced
    half-lattices arranged in a checkerboard pattern.
    arguments:
        system_A (DeviceNDArray) - the concentration values on the lattice before update.
        system_B (DeviceNDArray) - the concentration values on the lattice after update.
        red (bool) - determines whether to red or black squares are updated (red occupies top left corner).
        cluster_grid (DeviceNDArray) - the lattice with the DLA cluster values (NaN for unoccupied sites).
        use_objects (bool) - determines whether to use the objects array.
        omega (float) - the relaxation parameter.
    """
    i, j = cuda.grid(2) # get the position (row and column) on the grid

    # Determine whether to use objects
    if cluster_grid[i, j] == cluster_grid[i, j]: # Check for NaN
        update_toggle = False
    else:
        update_toggle = True

    # Update everything but the end rows
    if 0 < i < system_A.shape[0] - 1 and update_toggle:
        nt = system_B[i + 1, j]
        nb = system_B[i - 1, j]

        if red:
            nl = system_B[i, (j - 1) % system_A.shape[1]]
            nr = system_B[i, j]
        else:
            nl = system_B[i, j]
            nr = system_B[i, (j + 1) % system_A.shape[1]]
            
        # Update
        if omega == 1:
            system_A[i, j] = 0.25 * (nb + nt + nl + nr)
        else:
            system_A[i, j] = 0.25 * omega * (nb + nt + nl + nr) + (1 - omega) * system_A[i, j]
    
    # Set boundary conditions for top and bottom rows
    elif i == 0 or i == system_A.shape[0] - 1 or not update_toggle:
        system_A[i, j] = system_A[i, j]


def diffusion_sor(c_grid, cluster_grid, omega, delta_thresh=1e-5, max_iter=int(1e5), GPU_delta_interval=None):
    """
    Compute the diffusion of the concentration grid using the
    Successive Over-Relaxation (SOR) method.
    arguments:
        c_grid (ndarray): The concentration grid.
        cluster_grid (ndarray): The lattice with the DLA cluster values (NaN for unoccupied sites).
        omega (float): The relaxation factor.
        delta_thresh (float): The convergence threshold.
        max_iter (int): The maximum number of iterations.
        GPU_delta_interval (int): The interval at which to check for convergence on the GPU. If None, the CPU is used.
    returns:
        c_grid (ndarray): The updated concentration grid.
    """

    N = c_grid.shape[0]

    # Set sink and source rows
    c_grid[0, :] = 0
    c_grid[-1, :] = 1

    if GPU_delta_interval is not None:
        run_GPU = True
        d_c_red = cuda.to_device(np.ascontiguousarray(c_grid[:, ::2]))
        d_c_black = cuda.to_device(np.ascontiguousarray(c_grid[:, 1::2]))
    else:
        run_GPU = False

    for n in range(max_iter):

        # Compute the next iteration
        if not run_GPU:
            delta = 0
            for i in range(1, N - 1):
                for j in range(N):
                    if cluster_grid[i, j] != cluster_grid[i, j]: # Check for NaN
                        old_val = c_grid[i, j]
                        c_grid[i, j] = (1 - omega) * c_grid[i, j] + 0.25 * omega * (c_grid[(i-1)%N, j] + c_grid[(i+1)%N, j] + c_grid[i, (j-1)%N] + c_grid[i, (j+1)%N])
                        if c_grid[i, j] < 0:
                            print("Here's what happened:", i, j, c_grid[i, j], old_val)
                            print("Top neighbour:", c_grid[(i-1)%N, j])
                            print("Bottom neighbour:", c_grid[(i+1)%N, j])
                            print("Left neighbour:", c_grid[i, (j-1)%N])
                            print("Right neighbour:", c_grid[i, (j+1)%N])
                            print("Neighbour average:", 0.25 * (c_grid[(i-1)%N, j] + c_grid[(i+1)%N, j] + c_grid[i, (j-1)%N] + c_grid[i, (j+1)%N]))
                        diff = np.abs(c_grid[i, j] - old_val)
                        if diff > delta:
                            delta = diff

            # Check for convergence
            if delta < delta_thresh:
                break
        else:
            pass

    return c_grid


def run_dla_diff_equation(size, max_iter, omega, eta, GPU_delta_interval=None):
    """
    Performs Diffusion Limited Aggregation (DLA) using probabilities
    based on the time-independent diffusion equation (using SOR).
    arguments:
        size (int): The size of the grid.
        max_iter (int): The maximum number of iterations.
        omega (float): The relaxation factor.
        eta (float): The shape parameter for the DLA cluster.
        GPU_delta_interval (int): The interval at which to check for convergence on the GPU. If None, the CPU is used.
    """

    # Initialize the concentration grid (linear gradient from top to bottom)
    c_grid = np.tile(np.linspace(0, 1, size), (size, 1)).T

    # Place initial seed
    cluster_grid = np.full((size, size), np.nan)
    occupied_indices = np.array([[0, size//2]])
    cluster_grid[tuple(occupied_indices.T)] = 0
    c_grid = np.where(np.isnan(cluster_grid), c_grid, 0)

    # Construct neighbourhood stencil
    offsets = [[i, j] for i in range(-1, 2) for j in range(-1, 2) if abs(i) != abs(j)]

    # Growth loop
    for n in range(max_iter):

        # Update occupied sites
        occupied_indices = growth_iteration(c_grid, occupied_indices, eta, offsets)
        cluster_grid[tuple(occupied_indices[-1].T)] = 0
        
        # Update concentrations
        c_grid = np.where(np.isnan(cluster_grid), c_grid, 0)
        c_grid = diffusion_sor(c_grid, cluster_grid, omega, GPU_delta_interval=None)

    return c_grid


def run_dla_random_walk(size):
    """
    Performs Diffusion Limited Aggregation (DLA) using random walks.
    """

    return
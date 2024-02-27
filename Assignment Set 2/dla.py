import numpy as np
import pandas as pd
from numba import jit, cuda
from scipy.optimize import minimize_scalar

# ===== Iteration functions =====
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


def invoke_smart_kernel(size, threads_per_block=(16,16)):
    """
    Invokes kernel size parameters (number of blocks and number of threads per block).
    """
    blocks_per_grid = tuple([(s + tpb - 1) // tpb for s, tpb in zip(size, threads_per_block)])

    return blocks_per_grid, threads_per_block


@cuda.jit()
def diffusion_step_SOR_GPU(system_A, system_B, red, cluster_grid, omega):
    """
    A GPU-parallelised Gauss-Seidel iteration function using two interlaced
    half-lattices arranged in a checkerboard pattern.
    arguments:
        system_A (DeviceNDArray) - the concentration values on the lattice to update.
        system_B (DeviceNDArray) - the concentration values on the lattice to reference.
        red (bool) - determines whether the red or black squares are updated (red occupies top left corner).
        cluster_grid (DeviceNDArray) - the lattice with the DLA cluster values (NaN for unoccupied sites).
        omega (float) - the relaxation parameter.
    """
    i, j = cuda.grid(2) # get the position (row and column) on the grid

    # Determine whether to use objects
    if cluster_grid[i, j] == cluster_grid[i, j]: # Check for NaN
        update_toggle = False
    else:
        update_toggle = True

    # Update everything but the end rows and out-of-bounds squares
    if 0 < i < system_A.shape[0] - 1 and 0 <= j < system_A.shape[1] and update_toggle:
        nt = system_B[i + 1, j]
        nb = system_B[i - 1, j]

        if red:
            nl = system_B[i, (j - 1) % system_A.shape[1]]
            nr = system_B[i, j]
        else:
            nl = system_B[i, j]
            nr = system_B[i, (j + 1) % system_A.shape[1]]
            
        # Update
        system_A[i, j] = 0.25 * omega * (nb + nt + nl + nr) + (1 - omega) * system_A[i, j]

        # Clamp to zero
        system_A[i, j] = max(0, system_A[i, j])


def diffusion_SOR(c_grid, cluster_grid, omega, delta_thresh=1e-5, max_iter=int(1e5), GPU_delta_interval=None):
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

    size = c_grid.shape[0]

    # Set sink and source rows
    c_grid[0, :] = 0
    c_grid[-1, :] = 1

    # Previous grid placeholder
    c_grid_prev = np.array(c_grid)

    if GPU_delta_interval is not None:
        run_GPU = True
        d_c_red = cuda.to_device(np.ascontiguousarray(c_grid[:, ::2]))
        d_c_black = cuda.to_device(np.ascontiguousarray(c_grid[:, 1::2]))
        d_cluster_red = cuda.to_device(np.ascontiguousarray(cluster_grid[:, ::2]))
        d_cluster_black = cuda.to_device(np.ascontiguousarray(cluster_grid[:, 1::2]))
        delta = 1e6
    else:
        run_GPU = False

    
    for n in range(max_iter):

        # Compute the next iteration
        if not run_GPU:
            delta = 0
            for i in range(1, size - 1):
                for j in range(size):
                    if cluster_grid[i, j] != cluster_grid[i, j]: # Check for NaN
                        old_val = c_grid[i, j]
                        nbr_sum = (c_grid[(i-1)%size, j] + c_grid[(i+1)%size, j] + c_grid[i, (j-1)%size] + c_grid[i, (j+1)%size])
                        c_grid[i, j] = (1 - omega) * c_grid[i, j] + 0.25 * omega * nbr_sum
                        # Clamp to zero
                        c_grid[i, j] = max(0, c_grid[i, j])
                        diff = np.abs(c_grid[i, j] - old_val)
                        if diff > delta:
                            delta = diff
        else:
            # Alternate between black and red squares
            if n % 2 == 0:
                diffusion_step_SOR_GPU[invoke_smart_kernel((size, size//2))](d_c_red, d_c_black, True, d_cluster_red, omega)
            else:
                diffusion_step_SOR_GPU[invoke_smart_kernel((size, size//2))](d_c_black, d_c_red, False, d_cluster_black, omega)
            
            # Check for convergence
            if n % GPU_delta_interval == 0:
                c_grid[:, ::2] = d_c_red.copy_to_host()
                c_grid[:, 1::2] = d_c_black.copy_to_host()
                if n > 0:
                    delta = np.max(np.abs(c_grid - c_grid_prev))
            elif (n + 1) % GPU_delta_interval == 0 :
                c_grid_prev[:, ::2] = d_c_red.copy_to_host()	
                c_grid_prev[:, 1::2] = d_c_black.copy_to_host()
        
         # Check for convergence
        if (not run_GPU or (run_GPU and n % GPU_delta_interval == 0)) and delta < delta_thresh:
            break

    if n == max_iter - 1:
        print(f"Loop terminated before convergence at delta={delta}")

    return c_grid, n


def run_walker(cluster_grid, offsets):
    """
    Perform a random walk from the top of the lattice
    until the DLA cluster is reached.
    arguments:
        cluster_grid (ndarray): The lattice with the DLA cluster values (NaN for unoccupied sites).
        offsets (ndarray): The neighbourhood stencil.
    """
    
    pos = np.array([cluster_grid.shape[0] - 1, np.random.randint(0, cluster_grid.shape[1])]) # actually need to make sure there's no cluster on the top row

    max_steps = np.prod(cluster_grid.shape)

    for _ in range(max_steps):
        
        # Get neighbour indices
        nbrs = pos + offsets

        # Check for occupied neighbours
        mask = np.isnan(cluster_grid[tuple(nbrs.T)])

        # # Return if no unoccupied neighbours
        # if np.sum(mask) == 0:
        #     return

        # Add position to grid and return
        if np.prod(mask) == 0:
            cluster_grid[tuple(pos)] = 0
            return

        # nbrs = nbrs[mask]

        # Wrap around x
        nbrs[:, 1] = np.mod(nbrs[:, 1], cluster_grid.shape[1])

        # Move to a random neighbour
        pos = nbrs[np.random.choice(nbrs.shape[0])]

        # Return if beyond top or bottom row
        if pos[0] < 0 or pos[0] >= cluster_grid.shape[0]:
            return


# ===== Main DLA functions =====
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
    returns:
        c_grid (ndarray): The concentration grid.
        cluster_grid (ndarray): The DLA cluster grid.
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

    # Array for storing the number of diffusion diffusion iterations
    diff_n_maxs = np.empty(max_iter)

    # Growth loop
    for _ in range(max_iter):

        # Update occupied sites
        occupied_indices = growth_iteration(c_grid, occupied_indices, eta, offsets)
        cluster_grid[tuple(occupied_indices[-1].T)] = 0
        
        # Update concentrations
        c_grid = np.where(np.isnan(cluster_grid), c_grid, 0)
        c_grid, diff_n_maxs = diffusion_SOR(c_grid, cluster_grid, omega, GPU_delta_interval=GPU_delta_interval)

    return c_grid, cluster_grid, diff_n_maxs


def run_dla_monte_carlo(size, max_iter, ps):
    """
    Performs Diffusion Limited Aggregation (DLA) using random walks.
    """

    # Place initial seed
    cluster_grid = np.full((size, size), np.nan)
    occupied_indices = np.array([[0, size//2]])
    cluster_grid[tuple(occupied_indices.T)] = 0
    c_grid = np.where(np.isnan(cluster_grid), c_grid, 0)

    # Growth loop
    for _ in range(max_iter):

        # Random walk
        pass

    return


# ===== Simulation and analysis utilities =====
def dla_omega_optimiser(omega, size, max_iter, eta, random_seed, GPU_delta_interval):
    """
    Utility function for optimising omega with respect to the
    minimum number of diffusion iterations at each growth step.
    arguments:
        omega (float): The relaxation factor.
        size (int): The size of the grid.
        max_iter (int): The maximum number of iterations.
        eta (float): The shape parameter for the DLA cluster.
        random_seed (int): A random seed for consistent results.
        GPU_delta_interval (int): The interval at which to check for convergence on the GPU. If None, the CPU is used.
    returns:
        The norm of the vector of diffusion iteraction counts throughout the DLA growth.
    """

    np.random.seed(random_seed)

    _, _, diff_n_maxs = run_dla_diff_equation(size, max_iter, omega, eta, GPU_delta_interval)

    return np.linalg.norm(diff_n_maxs)


def dla_fractal_dimension(cluster_grid, seed):
    """
    Computes the fractal dimension of a DLA cluster by
    counting its mass within successive radii from the initial seed
    and performing a linear regression through log(r) and log(M).
    arguments:
        cluster_grid (ndarray): The lattice with the DLA cluster values (NaN for unoccupied sites).
        seed (ndarray): An array of shape (2,) containing the (x, y) coordinates of the initial seed
    returns:
    """

    assert seed.ndim == 1, 'seed must be a 1D array'
    assert seed.shape[0] == 2, 'seed must consist of 2 coordinates'

    cluster_coords = np.argwhere(np.logical_not(np.isnan(cluster_grid)))

    # Compute radii
    radii = np.linalg.norm(cluster_coords - np.tile(seed[np.newaxis, :], (cluster_coords.shape[0], 1)), axis=1)
    radii = radii[radii > 0]
    
    # Compute masses
    masses = np.empty_like(radii)
    for i, r in enumerate(radii):
        masses[i] = np.sum(np.where(radii <= r, 1, 0))
    
    # Fit line
    log_radii = np.log(radii)
    log_masses = np.log(masses)
    coeffs = np.polyfit(log_radii, log_masses, 1)

    # Slope is fractal dimension
    fract_dim = coeffs[0]

    return fract_dim


def run_dla_params(size, max_iter, eta_range, n_sims, GPU_delta_interval=50):
    """
    Runs multiple DLA simulations with different eta parameters
    using GPU acceleration.
    arguments:
        size (int): The size of the grid.
        max_iter (int): The maximum number of iterations.
        eta (float): The shape parameter for the DLA cluster.
        n_sims (int): The number of simulations per parameter value.
        GPU_delta_interval (int): The interval at which to check for convergence on the GPU. If None, the CPU is used.
    returns:
        df_sim_results (DataFrame): The results from the simulation series.
    """

    c_grids = np.empty((eta_range.shape[0], n_sims, size, size))
    cluster_grids = np.empty_like(c_grids)

    # DataFrame for saving simulation results
    df_sim_results = pd.DataFrame(columns=['sim_id', '$\eta$', '$\omega$', '$D_r$'])

    for i, eta in enumerate(eta_range):
        
        print(f"Running parameter eta = {eta}")

        for n in range(n_sims):

            sim_id = i * n_sims + n

            print(f"Running simulation {sim_id + 1}/{n_sims * eta_range.shape[0]}")

            random_seed = 11 + sim_id
            np.random.seed(random_seed)
            
            # Find optimal omega
            min_result = minimize_scalar(dla_omega_optimiser, bounds=(1.8, 1.999), args=(size, max_iter, eta, random_seed, GPU_delta_interval))
            omega = min_result.x

            # Rerun with optimal omega
            c_grid, cluster_grid, _ = run_dla_diff_equation(size, max_iter, omega, eta, GPU_delta_interval=50)
            c_grids[i, n] = np.array(c_grid)
            cluster_grids[i, n] = np.array(cluster_grid)

            # Compute fractal dimension
            fract_dim = dla_fractal_dimension(cluster_grid, np.array([0, size//2]))

            # Save results
            df_sim_results = pd.concat([df_sim_results, pd.DataFrame([{'sim_id': sim_id, '$\eta$': eta, '$\omega$': omega, '$D_r$': fract_dim}])])
    
    return df_sim_results, c_grids, cluster_grids
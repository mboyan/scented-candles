import numpy as np
import cupy as cp
import pandas as pd
from numba import jit, cuda
from scipy.optimize import minimize_scalar

# ===== Iteration functions =====
def growth_candidates(occupied_indices, offsets, size):
    """
    Determines the growth candidates for the DLA cluster.
    arguments:
        occupied_indices (ndarray): The indices of the occupied sites.
        offsets (ndarray): The neighbourhood stencil.
        size (int): The size of the grid.
    returns:
        growth_candidate_indices (ndarray): The indices of the growth candidates.
    """

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
    
    return np.array(list(growth_candidate_indices))


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
    growth_candidate_indices = growth_candidates(occupied_indices, offsets, size)
    c_candidates = c_grid[tuple(growth_candidate_indices.T)]

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


def generate_walkers(n_walkers, cluster_grid):
    """
    Generate random walker starting positions in the top row of the grid.
    arguments:
        n_walkers (int): The number of walkers to generate.
        cluster_grid (ndarray): The lattice with the DLA cluster values (NaN for unoccupied sites).
    returns:
        pos (ndarray): The starting positions of the walkers.
    """

    assert cluster_grid.shape[0] == cluster_grid.shape[1], 'grid must be square'

    # Vacant sites on top row
    # grid_indices = np.array([index for index in np.ndindex(cluster_grid.shape)])[-cluster_grid.shape[0]:]
    empty_pos = np.argwhere(np.isnan(cluster_grid))
    empty_pos_top_row = empty_pos[empty_pos[:, 0] == cluster_grid.shape[0] - 1]

    # Starting positions
    pos = empty_pos_top_row[np.random.choice(empty_pos_top_row.shape[0], n_walkers, replace=True)]

    return pos


def run_walkers(n_walkers, cluster_grid, offsets):
    """
    Perform a random walk from the top of the lattice
    until the DLA cluster is reached.
    arguments:
        n_walkers (int): The number of walkers to run.
        cluster_grid (ndarray): The lattice with the DLA cluster values (NaN for unoccupied sites).
        offsets (ndarray): The neighbourhood stencil.
    returns:
        cluster_grid (ndarray): The updated lattice with the DLA cluster values.
    """

    assert cluster_grid.shape[0] == cluster_grid.shape[1], 'grid must be square'
    assert n_walkers > 0, 'n_walkers must be greater than 0'

    # Starting positions
    pos = generate_walkers(n_walkers, cluster_grid)

    max_steps = np.prod(cluster_grid.shape) * 100
    grid_size = cluster_grid.shape[0]

    for _ in range(max_steps):
        
        # Get neighbour indices
        nbrs = pos[:, np.newaxis] + offsets

        nbrs_reshaped = nbrs.reshape(-1, 2)
        nbrs_stick = np.mod(nbrs_reshaped, grid_size)

        # Check for occupied neighbours
        occ_mask = np.logical_not(np.isnan(cluster_grid[tuple(nbrs_stick.T)].reshape(nbrs.shape[0], -1)))

        # Disregard out-of-bounds neighbours
        occ_mask = np.logical_and(occ_mask, np.logical_and(np.all(nbrs >= 0, axis=2), np.all(nbrs < grid_size, axis=2)))
        occ_mask = np.any(occ_mask, axis=1)

        # If any cluster cells detected, add first position to grid and return
        if np.any(occ_mask):
            first_occ = np.argwhere(occ_mask)[0]
            cluster_grid[tuple(pos[first_occ].T)] = 0
            return cluster_grid

        # Wrap around x
        nbrs[:, :, 1] = np.mod(nbrs[:, :, 1], grid_size)

        # Move to a random neighbour
        rand_indices = np.random.randint(0, nbrs.shape[1], pos.shape[0])
        pos = nbrs[np.arange(pos.shape[0]), rand_indices]

        # Regenerate if beyond top or bottom row
        out_of_bounds = np.logical_or(pos[:, 0] < 0, pos[:, 0] >= grid_size)
        if np.any(out_of_bounds):
            new_pos = generate_walkers(n_walkers, cluster_grid)
            pick_new = np.tile(out_of_bounds, (2, 1)).T
            pos = np.where(pick_new, new_pos, pos)
    
    return


def run_walkers_alt(n_walkers, cluster_grid, offsets, p_s=1.0):
    """
    Perform a random walk from the top of the lattice
    until the DLA cluster is reached.
    arguments:
        n_walkers (int): The number of walkers to run.
        cluster_grid (ndarray): The lattice with the DLA cluster values (NaN for unoccupied sites).
        offsets (list): The neighbourhood stencil.
        p_s (float): The probability of sticking to the cluster.
    returns:
        cluster_grid (ndarray): The updated lattice with the DLA cluster values.
    """

    assert cluster_grid.shape[0] == cluster_grid.shape[1], 'grid must be square'
    assert n_walkers > 0, 'n_walkers must be greater than 0'

    max_steps = np.prod(cluster_grid.shape)
    grid_size = cluster_grid.shape[0]

    # Starting positions
    pos = generate_walkers(n_walkers, cluster_grid)

    # Growth candidates
    grid_indices = np.array([index for index in np.ndindex(cluster_grid.shape)])
    occupied_indices = grid_indices[np.logical_not(np.isnan(cluster_grid)).flatten()]
    growth_candidate_indices = growth_candidates(occupied_indices, offsets, grid_size)
    growth_candidate_indices_flat = growth_candidate_indices[:, 0] * grid_size + growth_candidate_indices[:, 1]

    offsets = np.array(offsets)
    moves = offsets[np.random.choice(offsets.shape[0], n_walkers * max_steps)]
    moves = moves.reshape(n_walkers, max_steps, 2)
    moves_cumsum = np.cumsum(moves, axis=1)

    # Get neighbour indices
    trails = pos[:, np.newaxis] + moves_cumsum

    # Boundary conditions
    trails[:, :, 1] = np.mod(trails[:, :, 1], grid_size)
    
    # Flatten indices
    trails_flat_indices = trails[:, :, 0] * grid_size + trails[:, :, 1]
    trails_flat_indices = np.where(np.logical_or(trails[:, :, 1] < 0, trails[:, :, 1] >= grid_size), -1, trails_flat_indices)
    
    # Check for out-of-bounds
    trails_oob_mask = np.logical_or(trails[:, :, 0] < 0, trails[:, :, 0] >= grid_size).reshape(n_walkers, max_steps, 1)
    trails_oob_first_index = np.argmax(trails_oob_mask, axis=1)
    trails_oob_first_index = np.where(np.any(trails_oob_mask, axis=1), trails_oob_first_index, np.full((n_walkers, 1), max_steps))
    trails_oob_first_index = trails_oob_first_index.flatten()

    # Check for intersection with growth candidates
    intersect = np.in1d(trails_flat_indices.flatten(), growth_candidate_indices_flat).reshape(trails_flat_indices.shape)
    intersect_in_bounds = np.logical_and(intersect, np.arange(max_steps) < trails_oob_first_index[:, np.newaxis])
    intersect_first_indices = np.argmax(intersect, axis=1)
    intersect_growth_candidates = trails[np.arange(n_walkers), intersect_first_indices]
    
    intersect_first_indices = intersect_first_indices[np.any(intersect, axis=1)]
    intersect_growth_candidates = intersect_growth_candidates[np.any(intersect, axis=1)]
    trails_oob_first_index = trails_oob_first_index[np.any(intersect, axis=1)]
    intersect_growth_candidate = intersect_growth_candidates[intersect_first_indices < trails_oob_first_index]
    
    if intersect_growth_candidate.size > 0:
        gc_select = intersect_growth_candidate[0]
        cluster_grid[tuple(gc_select)] = 0
        return cluster_grid
    else:
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


def run_dla_monte_carlo(size, max_iter, p_s=1.0):
    """
    Performs Diffusion Limited Aggregation (DLA) using random walks.
    arguments:
        size (int): The size of the grid.
        max_iter (int): The maximum number of iterations.
        p_s (float): The probability of sticking to the cluster.
    """

    # Place initial seed
    cluster_grid = np.full((size, size), np.nan)
    occupied_indices = np.array([[0, size//2]])
    cluster_grid[tuple(occupied_indices.T)] = 0

    # Construct neighbourhood stencil
    offsets = [[i, j] for i in range(-1, 2) for j in range(-1, 2) if abs(i) != abs(j)]

    # Growth loop
    iter_ct = 0
    safety_ct = 0
    while iter_ct < max_iter and safety_ct < 1e6:

        # Random walk
        grid_update = run_walkers_alt(size, cluster_grid, offsets)
        if grid_update is not None:
            cluster_grid = grid_update
            iter_ct += 1
        # else:
        #     print("No attachment.")

        safety_ct += 1

    if safety_ct == 1e6:
        print("Loop terminated with incomplete results.")

    return cluster_grid


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
            c_grid, cluster_grid, _ = run_dla_diff_equation(size * size, max_iter, omega, eta, GPU_delta_interval=50)
            c_grids[i, n] = np.array(c_grid)
            cluster_grids[i, n] = np.array(cluster_grid)

            # Compute fractal dimension
            fract_dim = dla_fractal_dimension(cluster_grid, np.array([0, size//2]))

            # Save results
            df_sim_results = pd.concat([df_sim_results, pd.DataFrame([{'sim_id': sim_id, '$\eta$': eta, '$\omega$': omega, '$D_r$': fract_dim}])])
    
    return df_sim_results, c_grids, cluster_grids
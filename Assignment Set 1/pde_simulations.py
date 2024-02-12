import numpy as np
from numba import jit, cuda

def vibrating_string(u_init, t_max, c=1.0, dt=0.001, L=1):
    """
    Compute the evolution of the vibration amplitude of a string
    for a given number of timesteps, starting from an inital state
    and assuming rest at time zero (no initial velocity).
    inputs:
        u_init (numpy.ndarray) - the initial displacements along a discretised string;
        t_max (int) - a maximum number of iterations;
        c (float) - the physical constant for the string; defaults to 1.0;
        dt (float) - timestep; defaults to 0.001;
        L (float) - the length of the string; defaults to 1
    outputs:
        u_evolotion (numpy.ndarray) - the states of the string at all moments in time
    """

    assert u_init.shape[0] >= 3, 'string must be discretised to at least 3 points'
    assert u_init.ndim == 1, 'initial condition must be a 1D array'

    # Determine spatial increment
    dx = L / u_init.shape[0]

    # Determine number of frames
    n_frames = int(np.floor(t_max / dt))

    # Array for storing string states
    u_evolution = np.empty((n_frames, u_init.shape[0]))

    # Initialise previous and current state
    u_prev = np.array(u_init)
    u_curr = np.array(u_init)

    for t in range(n_frames):
        
        # Pad ends with zeros
        u_curr_padded = np.pad(u_curr, (1,1), mode='constant', constant_values=0)
        
        # Get shifted arrays
        u_curr_left = u_curr_padded[:-2]
        u_curr_right = u_curr_padded[2:]

        # Compute next state
        u_next = (c * dt) ** 2 * (u_curr_left + u_curr_right - 2*u_curr) / (dx ** 2) - u_prev + 2*u_curr

        # Set current to previous and next to current
        u_prev = np.array(u_curr)
        u_curr = np.array(u_next)

        u_evolution[t] = u_curr
    
    return u_evolution


def invoke_smart_kernel(size, threads_per_block=(16,16)):
    """
    Invokes kernel size parameters (number of blocks and number of threads per block)
    """
    blocks_per_grid = tuple([(size + tpb - 1) // tpb for tpb in threads_per_block])
    return blocks_per_grid, threads_per_block


@cuda.jit()
def diffusion_step_GPU(system_old, system_new):
    """
    A GPU-parallelised iteration function of the diffusion system
    inputs:
        system_old (GPU array) - the concentration values on the lattice before update
        system_new (GPU array) - the concentration values on the lattice after update
    """
    row, col = cuda.grid(2) # get the position (row and column) on the grid
    
    return


def diffusion_system(u_init, t_max, D=1.0, dt=0.001, L=1, run_GPU=False):
    """
    Compute the evolution of a square lattice of concentration scalars
    based on the time-dependent diffusion equation
    inputs:
        u_init (numpy.ndarray) - the initial state of the lattice;
        t_max (int) - a maximum number of iterations;
        D (float) - the diffusion constant; defaults to 1;
        dt (float) - timestep; defaults to 0.001;
        L (float) - the length of the lattice along one dimension; defaults to 1;
        run_GPU (bool) - determines whether the simulation runs on GPU
    outputs:
        u_evolotion (numpy.ndarray) - the states of the lattice at all moments in time
    """

    assert u_init.ndim == 2, 'input array must be 2-dimensional'
    assert u_init.shape[0] == u_init.shape[1], 'lattice must have equal size along each dimension'

    # Determine number of lattice intervals
    N = u_init.shape[0]

    # Determine spatial increment
    dx = L / u_init.shape[0]

    # Determine number of frames
    n_frames = int(np.floor(t_max / dt))

    # Array for storing lattice states
    u_evolution = np.empty((n_frames, N, N))

    # Initialise current state
    u_curr = np.array(u_init)

    # Send to device
    if run_GPU:
        d_u_curr = cuda.to_device(u_curr)
        d_u_next = cuda.to_device(u_curr)

    for t in range(n_frames):
        
        # Pad with 1s at the top and 0s at the bottom
        u_curr_padded = np.pad(u_curr, ((1, 1), (0, 0)), mode='constant', constant_values=((0, 1), (None, None)))
        print("Padded array:")
        print(u_curr_padded)
        print("=====")
        
        # Get shifted arrays
        u_curr_bottom = u_curr_padded[:-2]
        u_curr_top = u_curr_padded[2:]
        u_curr_left = np.roll(u_curr, -1, axis=1)
        u_curr_right = np.roll(u_curr, 1, axis=1)

        # Compute next state
        if run_GPU:
            diffusion_step_GPU[invoke_smart_kernel(N)](d_u_next, d_u_curr)

        else:
            u_next = (D * dt / (dx ** 2)) * (u_curr_bottom + u_curr_top + u_curr_left + u_curr_right - 4 * u_curr) + u_curr

        # Set current to previous and next to current
        u_curr = np.array(u_next)
        print("Resulting array:")
        print(u_curr)
        print("-----")

        u_evolution[t] = np.array(u_curr)
    
    return u_evolution
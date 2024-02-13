import numpy as np
from scipy.special import erfc
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
        L (float) - the length of the string; defaults to 1.
    outputs:
        u_evolotion (numpy.ndarray) - the states of the string at all moments in time.
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
def diffusion_step_GPU(system_old, system_new, D, dt, dx):
    """
    A GPU-parallelised iteration function of the diffusion system
    inputs:
        system_old (GPU array) - the concentration values on the lattice before update;
        system_new (GPU array) - the concentration values on the lattice after update.
    """
    row, col = cuda.grid(2) # get the position (row and column) on the grid
    size = cuda.gridsize(2)

    center = system_old[row, col]
    nt = system_old[row + 1, col]
    nb = system_old[row - 1, col]
    nl = system_old[row, col - 1]
    nr = system_old[row, col + 1]

    # Boundary conditions
    if row == 0:
        nb = 0
    elif row == size[0] - 1:
        nt = 1
    if col == 0:
        nl = system_old[row, size[0] - 1]
    elif col == size[1] - 1:
        nr = system_old[row, 0]
    
    # Update
    system_new[row, col] = (D * dt / (dx ** 2)) * (nb + nt + nl + nr - 4 * center) + center


def diffusion_system(u_init, t_max, D=1.0, dt=0.001, L=1, n_save_frames=100, run_GPU=False):
    """
    Compute the evolution of a square lattice of concentration scalars
    based on the time-dependent diffusion equation
    inputs:
        u_init (numpy.ndarray) - the initial state of the lattice;
        t_max (int) - a maximum number of iterations;
        D (float) - the diffusion constant; defaults to 1;
        dt (float) - timestep; defaults to 0.001;
        L (float) - the length of the lattice along one dimension; defaults to 1;
        n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100;
        run_GPU (bool) - determines whether the simulation runs on GPU.
    outputs:
        u_evolotion (numpy.ndarray) - the states of the lattice at all moments in time.
    """

    assert u_init.ndim == 2, 'input array must be 2-dimensional'
    assert u_init.shape[0] == u_init.shape[1], 'lattice must have equal size along each dimension'

    # Determine number of lattice intervals
    N = u_init.shape[0]

    # Determine spatial increment
    dx = L / u_init.shape[0]

    if D * dt / (dx ** 2) > 0.5:
        print("Warning: inappropriate scaling of dx and dt, may result in an unstable simulation.")

    # Determine number of frames
    n_frames = int(np.floor(t_max / dt))

    # Array for storing lattice states
    u_evolution = np.empty((n_save_frames, N, N))
    times = np.empty(n_save_frames)
    save_interval = np.floor(n_frames / n_save_frames)
    save_ct = 0

    # Initialise current state
    u_curr = np.array(u_init)

    # Send to device
    if run_GPU:
        d_u_curr = cuda.to_device(u_curr)
        d_u_next = cuda.to_device(u_curr)

    for t in range(n_frames):
        
        # Pad with 1s at the top and 0s at the bottom
        u_curr_padded = np.pad(u_curr, ((1, 1), (0, 0)), mode='constant', constant_values=((0, 1), (None, None)))
        
        # Get shifted arrays
        u_curr_bottom = u_curr_padded[:-2]
        u_curr_top = u_curr_padded[2:]
        u_curr_left = np.roll(u_curr, -1, axis=1)
        u_curr_right = np.roll(u_curr, 1, axis=1)

        # Compute next state
        if run_GPU:
            if t % 2 == 0:
                diffusion_step_GPU[invoke_smart_kernel(N)](d_u_curr, d_u_next, D, dt, dx)

                # Save frame
                if t % save_interval == 0:
                    u_evolution[save_ct] = d_u_next.copy_to_host()
                    times[save_ct] = t * dt
                    save_ct += 1
                
            else:
                diffusion_step_GPU[invoke_smart_kernel(N)](d_u_next, d_u_curr, D, dt, dx)

                # Save frame
                if t % save_interval == 0:
                    u_evolution[save_ct] = d_u_curr.copy_to_host()
                    times[save_ct] = t * dt
                    save_ct += 1
                
            
        else:
            u_next = (D * dt / (dx ** 2)) * (u_curr_bottom + u_curr_top + u_curr_left + u_curr_right - 4 * u_curr) + u_curr

            # Set current to previous and next to current
            u_curr = np.array(u_next)

            # Save frame
            if t % save_interval == 0:
                u_evolution[save_ct] = np.array(u_curr)
                times[save_ct] = t * dt
                save_ct += 1
    
    return u_evolution, times


def verify_analytical(u_frame, t, D=1.0, precision_steps=1e6):
    """Computes the error between a numerical solution of
    the diffusion equation and the analytical one based on the same parameters.
    inputs:
        u_frame (numpy.ndarray) - the numerically computed simulation frame;
        t (float) - the time for which to compute the analytical solution;
        D (float) - the diffusion constant; defaults to 1;
        precision_steps (int) - the number of steps to use in the analytical solution; defaults to 1e6.
    outputs:
        solution (numpy.ndarray) - the analytically computed simulation frame;
        error (numpy.ndarray) - the difference between the analytical solution and the numerical one.
    """

    assert u_frame.ndim == 2, 'input array must be 2-dimensional'
    assert u_frame.shape[0] == u_frame.shape[1], 'input array must have equal size along each dimension'

    N = u_frame.shape[0]

    # Initialise running sum of analytical terms
    running_sum = np.zeros(N)
    ys = np.linspace(0.5/N, 1-0.5/N, N)

    # Maximum iterations for analytical solution
    i_max = int(precision_steps)
    
    for i in range(i_max):
        running_sum += erfc((1 - ys + 2*i) / (2 * np.sqrt(D * t))) - erfc((1 + ys + 2*i) / (2 * np.sqrt(D * t)))

    # Repeat in 2D
    solution = np.tile(running_sum, (N, 1))

    # Calculate difference
    error = solution - u_frame

    return solution, error
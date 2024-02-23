import numpy as np
from scipy.special import erfc
from numba import jit, cuda

# ==== Wave Equation ====
def vibrating_string(u_init, t_max, c=1.0, dt=0.001, L=1):
    """
    Compute the evolution of the vibration amplitude of a string
    for a given number of timesteps, starting from an inital state
    and assuming rest at time zero (no initial velocity).
    inputs:
        c_init (numpy.ndarray) - the initial displacements along a discretised string;
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

# ==== Diffusion Equation: Parallelisation ====
def invoke_smart_kernel(size, threads_per_block=(16,16)):
    """
    Invokes kernel size parameters (number of blocks and number of threads per block).
    """
    # blocks_per_grid = tuple([(size + tpb - 1) // tpb for tpb in threads_per_block])
    blocks_per_grid = tuple([(s + tpb - 1) // tpb for s, tpb in zip(size, threads_per_block)])

    return blocks_per_grid, threads_per_block


@cuda.jit()
def diffusion_step_t_GPU(system_old, system_new, D, dt, dx, isolators, use_isolators):
    """
    A GPU-parallelised iteration function of the time-dependent diffusion system.
    inputs:
        system_old (DeviceNDArray) - the concentration values on the lattice before update;
        system_new (DeviceNDArray) - the concentration values on the lattice after update.
        D (float) - the diffusion constant;
        dt (float) - time interval;
        dx (float) - space interval;
        isolators (DeviceNDArray) - the lattice with initialised isolators;
        use_isolators (bool) - determines whether to use the isolators array.
    """
    i, j = cuda.grid(2) # get the position (row and column) on the grid

    # Update everything but the end rows
    if 0 < i < system_old.shape[0] - 1:
        center = system_old[i, j]
        nt = system_old[i + 1, j]
        nb = system_old[i - 1, j]
        nl = system_old[i, j - 1]
        nr = system_old[i, j + 1]

        # Boundary conditions
        if j == 0:
            nl = system_old[i, -1]
        elif j == system_old.shape[1] - 1:
            nr = system_old[i, 0]
            
        # Update
        if use_isolators:
            dtop = isolators[i + 1, j]
            dbottom = isolators[i - 1, j]
            dleft = isolators[i, j - 1]
            dright = isolators[i, j + 1]
            system_new[i, j] = (dt / (dx ** 2)) * (dtop * (nt - center)
                                                       + dbottom * (nb - center)
                                                       + dleft * (nl - center)
                                                       +dright * (nr - center)) + center
        else:
            system_new[i, j] = (D * dt / (dx ** 2)) * (nb + nt + nl + nr - 4 * center) + center
    
    # Set boundary conditions for top and bottom rows
    elif i == 0 or i == system_old.shape[0] - 1:
        system_new[i, j] = system_old[i, j]


@cuda.jit()
def diffusion_step_jacobi_GPU(system_old, system_new, objects, use_objects):
    """
    A GPU-parallelised Jacobi iteration function.
    inputs:
        system_old (DeviceNDArray) - the concentration values on the lattice before update;
        system_new (DeviceNDArray) - the concentration values on the lattice after update;
        objects (DeviceNDArray) - the lattice with initialised objects;
        use_objects (bool) - determines whether to use the objects array.
    """
    i, j = cuda.grid(2) # get the position (row and column) on the grid

    # Determine whether to use objects
    update_toggle = True
    if use_objects:
        if objects[i, j] == objects[i, j]: # Check for NaN
            update_toggle = False

    # Update everything but the end rows
    if 0 < i < system_old.shape[0] - 1 and update_toggle:
        nt = system_old[i + 1, j]
        nb = system_old[i - 1, j]
        nl = system_old[i, j - 1]
        nr = system_old[i, j + 1]

        # Boundary conditions
        if j == 0:
            nl = system_old[i, -1]
        elif j == system_old.shape[1] - 1:
            nr = system_old[i, 0]
            
        # Update
        system_new[i, j] = 0.25 * (nb + nt + nl + nr)
    
    # Set boundary conditions for top and bottom rows
    elif i == 0 or i == system_old.shape[0] - 1 or not update_toggle:
        system_new[i, j] = system_old[i, j]


@cuda.jit()
def diffusion_step_gauss_seidel_GPU(system_A, system_B, red, objects, use_objects, omega=1):
    """
    A GPU-parallelised Gauss-Seidel iteration function using two interlaced
    half-lattices arranged in a checkerboard pattern.
    inputs:
        system_A (DeviceNDArray) - the concentration values on the lattice before update;
        system_B (DeviceNDArray) - the concentration values on the lattice after update;
        red (bool) - determines whether to red or black squares are updated (red occupies top left corner)
        objects (DeviceNDArray) - the lattice with initialised objects;
        use_objects (bool) - determines whether to use the objects array;
        omega (float) - the relaxation parameter; if not 1, SOR is performed; defaults to 1;
    """
    i, j = cuda.grid(2) # get the position (row and column) on the grid

    # Determine whether to use objects
    update_toggle = True
    if use_objects:
        if objects[i, j] == objects[i, j]: # Check for NaN
            update_toggle = False

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
            system_A[i, j] = omega * (nb + nt + nl + nr) / 4 + (1 - omega) * system_A[i, j]
    
    # Set boundary conditions for top and bottom rows
    elif i == 0 or i == system_A.shape[0] - 1 or not update_toggle:
        system_A[i, j] = system_A[i, j]


# ==== Diffusion Equation =====
def diffusion_system_time_dependent(c_init, t_max, D=1.0, dt=0.001, L=1, n_save_frames=100, run_GPU=False, isolators=None):
    """
    Compute the evolution of a square lattice of concentration scalars
    based on the time-dependent diffusion equation.
    inputs:
        c_init (numpy.ndarray) - the initial state of the lattice;
        t_max (int) - a maximum number of iterations;
        D (float) - the diffusion constant; defaults to 1;
        dt (float) - timestep; defaults to 0.001;
        L (float) - the length of the lattice along one dimension; defaults to 1;
        n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100;
        run_GPU (bool) - determines whether the simulation runs on GPU;
        isolators (numpy.ndarray) - the lattice with initialised isolators; defaults to None.
    outputs:
        u_evolotion (numpy.ndarray) - the states of the lattice at all moments in time.
    """

    assert c_init.ndim == 2, 'input array must be 2-dimensional'
    assert c_init.shape[0] == c_init.shape[1], 'lattice must have equal size along each dimension'
    if isolators is not None:
        assert isolators.shape == c_init.shape, 'isolators array must have the same shape as the lattice'

    # Determine number of lattice rows/columns
    N = c_init.shape[0]

    # Determine spatial increment
    dx = L / c_init.shape[0]

    if D * dt / (dx ** 2) > 0.5:
        print("Warning: inappropriate scaling of dx and dt, may result in an unstable simulation.")

    # Determine number of frames
    n_frames = int(np.floor(t_max / dt)) + 1
    print(f"Simulation running for {n_frames} steps.")

    # Array for storing lattice states
    c_evolution = np.zeros((n_save_frames + 1, N, N))
    times = np.zeros(n_save_frames + 1)
    save_interval = np.floor(n_frames / n_save_frames)
    save_ct = 0

    # Initialise current state
    c_curr = np.array(c_init)
    c_curr[-1, :] = 1

    # Determine whether to use isolators in GPU
    if isolators is not None:
        use_isolators = True
    else:
        use_isolators = False
        isolators = np.full_like(c_curr, 1)

    # Send to device
    if run_GPU:
        d_c_curr = cuda.to_device(c_curr)
        d_c_next = cuda.to_device(c_curr)
        d_isolators = cuda.to_device(isolators)

    for t in range(n_frames):

        # Compute next state
        if run_GPU:

            if t % 2 == 0:
                diffusion_step_t_GPU[invoke_smart_kernel((N, N))](d_c_curr, d_c_next, D, dt, dx, d_isolators, use_isolators)

                # Save frame
                if t % save_interval == 0:
                    c_evolution[save_ct] = d_c_next.copy_to_host()
                    times[save_ct] = t * dt
                    save_ct += 1
                
            else:
                diffusion_step_t_GPU[invoke_smart_kernel((N, N))](d_c_next, d_c_curr, D, dt, dx, d_isolators, use_isolators)

                # Save frame
                if t % save_interval == 0:
                    c_evolution[save_ct] = d_c_curr.copy_to_host()
                    times[save_ct] = t * dt
                    save_ct += 1
            
        else:
            c_curr_center = c_curr[1:-1,:]
            c_curr_bottom = c_curr[:-2,:]
            c_curr_top = c_curr[2:,:]
            c_curr_left = np.roll(c_curr, -1, axis=1)[1:-1,:]
            c_curr_right = np.roll(c_curr, 1, axis=1)[1:-1,:]
            
            if use_isolators:
                d_bottom = isolators[:-2,:]
                d_top = isolators[2:,:]
                d_left = np.roll(isolators, -1, axis=1)[1:-1,:]
                d_right = np.roll(isolators, 1, axis=1)[1:-1,:]
                c_next = (dt / (dx ** 2)) * (d_top * (c_curr_top - c_curr_center)
                                                 + d_bottom * (c_curr_bottom - c_curr_center)
                                                 + d_left * (c_curr_left - c_curr_center)
                                                 + d_right * (c_curr_right - c_curr_center)) + c_curr_center
            else:
                c_next = (D * dt / (dx ** 2)) * (c_curr_bottom + c_curr_top + c_curr_left + c_curr_right - 4 * c_curr_center) + c_curr_center
            
            # Update current array (apart from top and bottom row)
            c_curr[1:-1, :] = np.array(c_next)

            # Save frame
            if t % save_interval == 0:
                c_evolution[save_ct] = np.array(c_curr)
                times[save_ct] = t * dt
                save_ct += 1

    return c_evolution, times


def diffusion_system_time_independent(c_init, delta_thresh=1e-5, n_max_iter=int(1e5), omega=None, gauss_seidel=False,
                                      save_interval=100, delta_interval=10, run_GPU=False, objects=None):
    """
    Compute the evolution of a square lattice of diffusing concentration scalars
    based on a time-independent Laplacian equation (Jacobi, Gauss-Seidel or SOR).
    inputs:
        c_init (numpy.ndarray) - the initial state of the lattice;
        delta_thresh (float) - the termination threshold for the maximum difference in concentrations between timesteps; defaults to 1e-5;
        n_max_iter (int) - the maximum number of iterations for the scheme; defaults to 1e5;
        omega (float) - if provided, a Successive Over-Relaxation is performed with relaxation parameter omega; defaults to None;
        gauss_seidel (bool) - determines whether to modify the lattice in place (Gauss-Seidel iteration); defaults to False;
        save_interval (int) - interval between saving frames to the output array; detaults to 100;
        delta_interval (int) - interval between calculating convergence; defaults to 10;
        run_GPU (bool) - determines whether the simulation runs on GPU;
        objects (numpy.ndarray) - the lattice with initialised objects.
    outputs:
        c_evolotion (numpy.ndarray) - the states of the lattice at all moments in time.
    """

    assert c_init.ndim == 2, 'input array must be 2-dimensional'
    assert c_init.shape[0] == c_init.shape[1], 'lattice must have equal size along each dimension'
    assert omega is None or 0 < omega <= 2, 'omega parameter must be between 0 and 2 for stability'
    assert type(n_max_iter) == int, 'n_max_iter must be an integer'
    if objects is not None:
        assert objects.shape == c_init.shape, 'objects array must have the same shape as the lattice'

    # Determine number of lattice rows/columns
    N = c_init.shape[0]

    # Array for storing lattice states
    c_evolution = np.empty((1, N, N))
    c_evolution[0] = np.array(c_init)

    # Initialise current state
    c_curr = np.array(c_init)
    c_curr[-1, :] = 1

    c_prev = np.full_like(c_curr, 1e6)

    # Determine whether to use objects in GPU
    if objects is not None:
        use_objects = True
        # Add object values to initial state
        c_curr = np.where(np.isnan(objects), c_curr, objects)
    else:
        use_objects = False
        objects = np.full_like(c_curr, np.nan)

    # Send to device
    if run_GPU:
        if not gauss_seidel:
            d_c_curr = cuda.to_device(c_curr)
            d_c_next = cuda.to_device(c_curr)
            d_objects = cuda.to_device(objects)
        else:
            # Create interlaced half-lattices
            d_c_red = cuda.to_device(np.ascontiguousarray(c_curr[:, ::2]))
            d_c_black = cuda.to_device(np.ascontiguousarray(c_curr[:, 1::2]))
            d_objects_red = cuda.to_device(np.ascontiguousarray(objects[:, ::2]))
            d_objects_black = cuda.to_device(np.ascontiguousarray(objects[:, 1::2]))

    for n in range(n_max_iter):

        if run_GPU:
            if gauss_seidel:
                # Perform parallel Gauss-Seidel Iteration
                if omega is None:
                    omega_input = 1
                else:
                    omega_input = omega
                diffusion_step_gauss_seidel_GPU[invoke_smart_kernel((N, N//2))](d_c_red, d_c_black, True, d_objects_red, use_objects, omega_input)
                diffusion_step_gauss_seidel_GPU[invoke_smart_kernel((N, N//2))](d_c_black, d_c_red, False, d_objects_black, use_objects, omega_input)
                if n % save_interval == 0:
                    c_curr[:, ::2] = d_c_red.copy_to_host()
                    c_curr[:, 1::2] = d_c_black.copy_to_host()
                elif (n + 1) % delta_interval == 0:
                    c_prev[:, ::2] = d_c_red.copy_to_host()	
                    c_prev[:, 1::2] = d_c_black.copy_to_host()
            else:
                # Perform parallel Jacobi Iteration
                if n % 2 == 0:
                    diffusion_step_jacobi_GPU[invoke_smart_kernel((N, N))](d_c_curr, d_c_next, d_objects, use_objects)
                    if n % save_interval == 0 or n % delta_interval == 0:
                        c_curr = d_c_next.copy_to_host()
                    elif (n + 1) % delta_interval == 0:
                        c_prev = d_c_next.copy_to_host()
                else:
                    diffusion_step_jacobi_GPU[invoke_smart_kernel((N, N))](d_c_next, d_c_curr, d_objects, use_objects)
                    if n % save_interval == 0 or n % delta_interval == 0:
                        c_curr = d_c_curr.copy_to_host()
                    elif (n + 1) % delta_interval == 0:
                        c_prev = d_c_curr.copy_to_host()
            # Halting condition
            delta = np.max(np.abs(c_curr - c_prev))
        else:
            if gauss_seidel:
                delta = 0
                if omega is None:
                    # Perform Gauss-Seidel Iteration
                    for i in range(1, N - 1):
                        for j in range(N):
                            old_val = c_curr[i, j]
                            c_curr[i, j] = 0.25 * (c_curr[(i-1)%N, j] + c_curr[(i+1)%N, j] + c_curr[i, (j-1)%N] + c_curr[i, (j+1)%N])
                            diff = np.abs(c_curr[i, j] - old_val)
                            if diff > delta:
                                delta = diff
                else:
                    # Perform SOR
                    for i in range(1, N - 1):
                        for j in range(N):
                            old_val = c_curr[i, j]
                            c_curr[i, j] = omega * (c_curr[(i-1)%N, j] + c_curr[(i+1)%N, j] + c_curr[i, (j-1)%N] + c_curr[i, (j+1)%N]) / 4 + (1 - omega) * c_curr[i, j]
                            diff = np.abs(c_curr[i, j] - old_val)
                            if diff > delta:
                                delta = diff
            else:
                # Perform Jacobi Iteration
                c_curr_bottom = c_curr[:-2,:]
                c_curr_top = c_curr[2:,:]
                c_curr_left = np.roll(c_curr, -1, axis=1)[1:-1,:]
                c_curr_right = np.roll(c_curr, 1, axis=1)[1:-1,:]

                c_next = 0.25 * (c_curr_bottom + c_curr_top + c_curr_left + c_curr_right)

                # Calculate max difference
                delta = np.max(np.abs(c_curr[1:-1, :] - c_next))
                
                # Update current array (apart from top and bottom row)
                c_curr[1:-1, :] = np.array(c_next)
            
            # Overwrite objects
            if objects is not None:
                c_curr = np.where(np.isnan(objects), c_curr, objects)
            
        if n % save_interval == 0:
            c_evolution = np.append(c_evolution, c_curr[np.newaxis, :, :], axis=0)

        if delta < delta_thresh:
            print(f"Terminating after {n} iterations.")
            break
    
    # Save last frame
    if n % save_interval != 0:
        c_evolution = np.append(c_evolution, c_curr[np.newaxis, :, :], axis=0)

    return c_evolution, n


def init_lattice_object(c_base, center, size, type='square', const_val=0.0):
    """
    Initialises regions on a lattice with special behaviour (sinks, sources).
    inputs:
        c_base (numpy.ndarray) - the initial state of the lattice;
        center (numpy.ndarray) - the coordinates of the center of the object;
        size (float) - the size of the object (radius for circles, half-side for rectangles);
        type (str) - the geometry of the region; can be one of the following: 'square', 'circle' or 'sierpinski carpet'; defaults to 'rectangle';
        const_val (float) - the value that will be permanently assigned to the region; defaults to 0.0 (sink);
    outputs:
        c_objects (numpy.ndarray) - the lattice with the object initialised.
    """
    # print(size)
    assert c_base.ndim == 2, 'input lattice array must be 2-dimensional'
    assert center.ndim == 1, 'center coordinates must be a 1D array'
    assert center.shape[0] == 2, 'center can only have 2 coordinates (x, y)'

    c_objects = np.where(c_base == 0, np.nan, c_base)

    if type == 'square':
        c_objects[int(center[0]) - int(size):int(center[0]) + int(size), int(center[1]) - int(size):int(center[1]) + int(size)] = const_val
    elif type == 'circle':
        # Get coordinates of all points in the lattice
        coords = np.array(np.meshgrid(np.arange(c_base.shape[0]), np.arange(c_base.shape[1]))).T
        c_objects = np.where(np.linalg.norm(coords - np.tile(center, (c_objects.shape[0], c_objects.shape[1], 1)), axis=2) <= size, const_val, c_objects)
    elif type == 'sierpinski carpet':
        if size > 18:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not (i == 0 and j == 0):
                        c_subdiv = init_lattice_object(c_objects, center + np.array([i, j]) * size / 3, size / 3, type='sierpinski carpet', const_val=const_val)
                        c_objects = np.where(np.isnan(c_subdiv), c_objects, c_subdiv)
                    else:
                        c_subdiv = init_lattice_object(c_objects, center + np.array([i, j]) * size / 3, size / 6, type='square', const_val=const_val)
                        c_objects = np.where(np.isnan(c_subdiv), c_objects, c_subdiv)
        else:
            c_objects = init_lattice_object(c_objects, center, size / 3, type='square', const_val=const_val)
    return c_objects


def init_lattice_d_params(c_base, center, size, type='square'):
    """
    Initialises regions on a lattice with insulating or conducting behaviour.
    inputs:
        c_base (numpy.ndarray) - the initial state of the lattice;
        center (numpy.ndarray) - the coordinates of the center of the object;
        size (float) - the size of the object (radius for circles, half-side for rectangles);
        type (str) - the geometry of the region; can be one of the following: 'square', 'circle' or 'sierpinski carpet'; defaults to 'rectangle';
        const_val (float) - the value that will be permanently assigned to the region; defaults to 0.0 (sink);
    outputs:
        c_objects (numpy.ndarray) - the lattice with the object initialised.
    """
    # print(size)
    assert c_base.ndim == 2, 'input lattice array must be 2-dimensional'
    assert center.ndim == 1, 'center coordinates must be a 1D array'
    assert center.shape[0] == 2, 'center can only have 2 coordinates (x, y)'

    c_objects = np.full_like(c_base, 1)

    if type == 'square':
        c_objects[int(center[0]) - int(size):int(center[0]) + int(size), int(center[1]) - int(size):int(center[1]) + int(size)] = 0
    elif type == 'circle':
        # Get coordinates of all points in the lattice
        coords = np.array(np.meshgrid(np.arange(c_base.shape[0]), np.arange(c_base.shape[1]))).T
        c_objects = np.where(np.linalg.norm(coords - np.tile(center, (c_objects.shape[0], c_objects.shape[1], 1)), axis=2) <= size, 0, c_objects)
    elif type == 'sierpinski carpet':
        if size > 18:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not (i == 0 and j == 0):
                        c_objects *= init_lattice_d_params(c_objects, center + np.array([i, j]) * size / 3, size / 3, type='sierpinski carpet')
                    else:
                        c_objects *= init_lattice_d_params(c_objects, center + np.array([i, j]) * size / 3, size / 6, type='square')
        else:
            c_objects = init_lattice_d_params(c_objects, center, size / 3, type='square')
    return c_objects


def verify_analytical_tdde(c_frame, t, D=1.0, precision_steps=1e6):
    """Computes the error between a numerical solution of
    the time-dependent diffusion equation and the analytical one based on the same parameters.
    inputs:
        c_frame (numpy.ndarray) - the numerically computed simulation frame;
        t (float) - the time for which to compute the analytical solution;
        D (float) - the diffusion constant; defaults to 1;
        precision_steps (int) - the number of steps to use in the analytical solution; defaults to 1e6.
    outputs:
        solution (numpy.ndarray) - the analytically computed simulation frame;
        error (numpy.ndarray) - the difference between the analytical solution and the numerical one.
    """

    assert c_frame.ndim == 2, 'input array must be 2-dimensional'
    assert c_frame.shape[0] == c_frame.shape[1], 'input array must have equal size along each dimension'

    N = c_frame.shape[0]

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
    error = solution - c_frame

    return solution, error
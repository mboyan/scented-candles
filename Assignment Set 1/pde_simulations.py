import numpy as np
import numba as nb

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

    # Array for storing string states
    u_evolution = np.empty((t_max, u_init.shape[0]))

    # Initialise previous and current state
    u_prev = np.copy(u_init)
    u_curr = np.copy(u_init)

    for t in range(t_max):
        
        # Pad ends with zeros
        u_curr_padded = np.pad(u_curr, (1,1), mode='constant', constant_values=0)
        
        # Get shifted arrays
        u_curr_left = u_curr_padded[:-2]
        u_curr_right = u_curr_padded[2:]

        # Compute next state
        u_next = (c * dt) ** 2 * (u_curr_left + u_curr_right - 2*u_curr) / (dx ** 2) - u_prev + 2*u_curr

        # Set current to previous and next to current
        u_prev = np.copy(u_curr)
        u_curr = np.copy(u_next)

        u_evolution[t] = u_curr
    
    return u_evolution
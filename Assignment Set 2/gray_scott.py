import numpy as np
from numba import jit, cuda
from dla import invoke_smart_kernel

@cuda.jit()
def gray_scott_step_GPU(u_old, u_new, v_old, v_new, f, k, dt, Du_dx2, Dv_dx2):
    """
    Updates the concentration of U and V using the Gray-Scott model.
    arguments:
        u (DeviceNDArray): The concentration of U.
        v (DeviceNDArray): The concentration of V.
        F (float): The feed rate.
        k (float): The kill rate.
        dt (float): The time step.
        Du_dx2 (float): The pre-computed constant Du / dx**2.
        Dv_dx2 (float): The pre-computed constant Dv / dx**2.
    """

    i, j = cuda.grid(2) # get the position (row and column) on the grid

    if i >= 0 and i < u_old.shape[0] and j >= 0 and j < u_old.shape[1]:
        # Compute the Laplacian of U and V
        lapl_u = u_old[(i+1)%u_old.shape[0], j] + u_old[(i-1)%u_old.shape[0], j] + u_old[i, (j+1)%u_old.shape[1]] + u_old[i, (j-1)%u_old.shape[1]] - 4 * u_old[i, j]
        lapl_v = v_old[(i+1)%v_old.shape[0], j] + v_old[(i-1)%v_old.shape[0], j] + v_old[i, (j+1)%v_old.shape[1]] + v_old[i, (j-1)%v_old.shape[1]] - 4 * v_old[i, j]

        # Update the concentration of U and V
        u_new[i, j] += dt * (Du_dx2 * lapl_u - u_old[i, j] * v_old[i, j]**2 + f * (1 - u_old[i, j]))
        v_new[i, j] += dt * (Dv_dx2 * lapl_v + u_old[i, j] * v_old[i, j]**2 - (f + k) * v_old[i, j])


def run_gray_scott(u0, v0, max_iter, Du, Dv, f, k, dt, dx):
    """
    Simulates the Gray-Scott reaction-diffusion model
    on a doubly periodic lattice.
    arguments:
        u0 (ndarray): The initial concentration of U.
        v0 (ndarray): The initial concentration of V.
        max_iter (int): The number of iterations to run.
        Du (float): The diffusion rate of U.
        Dv (float): The diffusion rate of V.
        F (float): The feed rate.
        k (float): The kill rate.
        dt (float): The time step.
        dx (float): The spatial step.
    returns:
        u (ndarray): The concentration of U after max_iter iterations.
        v (ndarray): The concentration of V after max_iter iterations.
    """

    assert u0.shape == v0.shape, 'u0 and v0 must have the same shape.'
    assert u0.ndim == 2, 'u0 and v0 must be 2-dimensional.'
    assert u0.shape[0] == u0.shape[1], 'u0 and v0 must be square.'

    size = u0.shape[0]

    u = u0.copy()
    v = v0.copy()

    # Initialize the GPU arrays
    d_u_a = cuda.to_device(u)
    d_u_b = cuda.to_device(u)
    d_v_a = cuda.to_device(v)
    d_v_b = cuda.to_device(v)

    # Pre-compute constants
    Du_dx2 = Du * dt / dx**2
    Dv_dx2 = Dv * dt / dx**2

    for _ in range(max_iter):
        # gray_scott_step_GPU[invoke_smart_kernel((size, size))](d_u_a, d_u_b, d_v_a, d_v_b, f, k, dt, Du_dx2, Dv_dx2)
        # gray_scott_step_GPU[invoke_smart_kernel((size, size))](d_u_b, d_u_a, d_v_b, d_v_a, f, k, dt, Du_dx2, Dv_dx2)

        u_bottom = np.roll(u, 1, axis=0)
        u_top = np.roll(u, -1, axis=0)
        u_left = np.roll(u, 1, axis=1)
        u_right = np.roll(u, -1, axis=1)

        v_bottom = np.roll(v, 1, axis=0)
        v_top = np.roll(v, -1, axis=0)
        v_left = np.roll(v, 1, axis=1)
        v_right = np.roll(v, -1, axis=1)

        lapl_u = u_bottom + u_top + u_left + u_right - 4 * u
        lapl_v = v_bottom + v_top + v_left + v_right - 4 * v

        u += dt * (Du_dx2 * lapl_u - u * v**2 + f * (1 - u))
        v += dt * (Dv_dx2 * lapl_v + u * v**2 - (f + k) * v)
    
    # Copy the results back to the CPU
    # u = d_u_a.copy_to_host()
    # v = d_v_a.copy_to_host()

    return u, v
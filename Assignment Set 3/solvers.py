import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def construct_coeff_matrix(size_x, size_y, boundary_type='rect', sparse=False):
    """
    Construct the coefficient matrix A for the eigenvalue problem
    Mv = Kv of the 2D wave equation. Uses a specific boundary shape,
    if circle is chosen, smallest of the two dimensions is used as the radius.
    Arguments:
        size_x (int): number of points in the x direction.
        size_y (int): number of points in the y direction.
        boundary_type (str): type of boundary condition. Can be 'rect' or 'circ'.
        sparse (bool): whether to return a sparse matrix.
    Returns:
        np.ndarray: the coefficient matrix M.
    """

    assert boundary_type in ['rect', 'circle'], "Invalid boundary type."

    # Matrix indices
    mat_1D_indices = np.arange(size_x * size_y)
    mat_2D_indices = np.array([i for i in np.ndindex((size_x, size_y))])

    if boundary_type == 'circle':
        radius = min(size_x, size_y) / 2
        in_circle_mask = np.linalg.norm((mat_2D_indices - np.array([(size_x - 1) / 2, (size_y - 1) / 2])).T, axis=0) <= radius
        in_circle_mask_2D = in_circle_mask.reshape(size_x, size_y)
        
        # Get neighbour indices
        shifts = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]
        offset_indices = np.array([mat_2D_indices + shift for shift in shifts])
        offset_indices = np.clip(offset_indices, 0, np.array([size_x, size_y]) - 1)

        # Mark circular boundary
        bnd_tags = np.sum(in_circle_mask_2D[tuple(offset_indices.reshape(-1, 2).T)].reshape(4, size_x, size_y), axis=0)
        bnd_tags = np.mod(bnd_tags, 4) != 0
        bnd_tags = np.logical_or(bnd_tags, np.any(mat_2D_indices == 0, axis=1).reshape(size_x, size_y))
        bnd_tags = np.logical_or(bnd_tags, np.any(mat_2D_indices == np.array([size_x - 1, size_y - 1]), axis=1).reshape(size_x, size_y))
        bnd_tags = np.where(np.logical_and(bnd_tags, in_circle_mask_2D), True, False)

        # Get boundary indices
        bnd_1D_indices = mat_1D_indices[bnd_tags.reshape(-1)]

    else:
        # Get boundary indices
        bnd_1D_indices = mat_1D_indices[(mat_2D_indices[:, 0] == 0) | (mat_2D_indices[:, 0] == size_x - 1) | (mat_2D_indices[:, 1] == 0) | (mat_2D_indices[:, 1] == size_y - 1)]

    # Get neighbour indices
    inner_off_diag_idx_left = np.array([mat_1D_indices[1:], mat_1D_indices[:-1]])
    inner_off_diag_idx_right = np.array([mat_1D_indices[:-1], mat_1D_indices[1:]])
    inner_off_diag_idx_top = np.array([mat_1D_indices[size_y:], mat_1D_indices[:-size_y]])
    inner_off_diag_idx_bottom = np.array([mat_1D_indices[:-size_y], mat_1D_indices[size_y:]])

    # Remove neighbours from boundaries
    inner_off_diag_idx_left = inner_off_diag_idx_left.T[np.logical_not(np.any(np.isin(inner_off_diag_idx_left, bnd_1D_indices), axis=0))].T
    inner_off_diag_idx_right = inner_off_diag_idx_right.T[np.logical_not(np.any(np.isin(inner_off_diag_idx_right, bnd_1D_indices), axis=0))].T
    inner_off_diag_idx_top = inner_off_diag_idx_top.T[np.logical_not(np.any(np.isin(inner_off_diag_idx_top, bnd_1D_indices), axis=0))].T
    inner_off_diag_idx_bottom = inner_off_diag_idx_bottom.T[np.logical_not(np.any(np.isin(inner_off_diag_idx_bottom, bnd_1D_indices), axis=0))].T

    # Reduce array to cirular region
    if boundary_type == 'circle':
        idx_mapping = np.where(in_circle_mask, np.arange(size_x * size_y), -1)
        idx_mapping[in_circle_mask] = np.arange(np.sum(in_circle_mask))

        inner_off_diag_idx_left = idx_mapping[inner_off_diag_idx_left]
        inner_off_diag_idx_right = idx_mapping[inner_off_diag_idx_right]
        inner_off_diag_idx_top = idx_mapping[inner_off_diag_idx_top]
        inner_off_diag_idx_bottom = idx_mapping[inner_off_diag_idx_bottom]

        inner_off_diag_idx_left = inner_off_diag_idx_left[:, np.all(inner_off_diag_idx_left != -1, axis=0)]
        inner_off_diag_idx_right = inner_off_diag_idx_right[:, np.all(inner_off_diag_idx_right != -1, axis=0)]
        inner_off_diag_idx_top = inner_off_diag_idx_top[:, np.all(inner_off_diag_idx_top != -1, axis=0)]
        inner_off_diag_idx_bottom = inner_off_diag_idx_bottom[:, np.all(inner_off_diag_idx_bottom != -1, axis=0)]
        
        mat_1D_indices = np.arange(np.sum(in_circle_mask))
        mat_2D_indices = mat_2D_indices[in_circle_mask]

    # Construct coefficient matrix
    coeffs_matrix = np.zeros((mat_1D_indices.shape[0], mat_1D_indices.shape[0]))

    # Set diagonal to -4
    coeffs_matrix[mat_1D_indices, mat_1D_indices] = -4

    # Set off-diagonals to 1
    coeffs_matrix[tuple(inner_off_diag_idx_left)] = 1
    coeffs_matrix[tuple(inner_off_diag_idx_right)] = 1
    coeffs_matrix[tuple(inner_off_diag_idx_top)] = 1
    coeffs_matrix[tuple(inner_off_diag_idx_bottom)] = 1

    # Convert to sparse matrix
    if sparse:
        coeffs_matrix = sp.dia_matrix(coeffs_matrix)

    return coeffs_matrix, mat_2D_indices


def construct_coeff_matrix_alt(size_x, size_y, boundary_type='rect', sparse=False, src_pt_idx=None):
    """
    Construct the coefficient matrix A for the eigenvalue problem
    of the 2D wave equation with a boundary fixed to zero.
    Uses a specific boundary shape, if circle is chosen,
    smallest of the two dimensions is used as the radius.
    Arguments:
        size_x (int): number of points in the x direction.
        size_y (int): number of points in the y direction.
        boundary_type (str): type of boundary condition. Can be 'rect' or 'circ'.
        sparse (bool): whether to return a sparse matrix. Default is False.
        src_pt_idx (tuple): the index of a source point. Default is None.
    Returns:
        np.ndarray: the coefficient matrix M.
    """

    assert boundary_type in ['rect', 'circle'], "Invalid boundary type."
    assert type(src_pt_idx) in [tuple, type(None)], "Invalid source point index."
    if src_pt_idx is not None:
        assert len(src_pt_idx) == 2, "Invalid number of source point coordinates."
        assert src_pt_idx[0] in range(size_x) and src_pt_idx[1] in range(size_y), "Source point index out of range."

    # Matrix indices
    mat_1D_indices = np.arange(size_x * size_y)
    system_coords = np.array([i for i in np.ndindex((size_x, size_y))])

    if boundary_type == 'circle':
        radius = min(size_x, size_y) / 2
        in_circle_mask = np.linalg.norm((system_coords - np.array([(size_x - 1) / 2, (size_y - 1) / 2])).T, axis=0) <= radius
        in_circle_mask_2D = in_circle_mask.reshape(size_x, size_y)
        
        # Get neighbour indices
        shifts = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]
        offset_indices = np.array([system_coords + shift for shift in shifts])
        offset_indices = np.clip(offset_indices, 0, np.array([size_x, size_y]) - 1)

        # Mark circular boundary
        bnd_tags = np.sum(in_circle_mask_2D[tuple(offset_indices.reshape(-1, 2).T)].reshape(4, size_x, size_y), axis=0)
        bnd_tags = np.mod(bnd_tags, 4) != 0
        bnd_tags = np.logical_or(bnd_tags, np.any(system_coords == 0, axis=1).reshape(size_x, size_y))
        bnd_tags = np.logical_or(bnd_tags, np.any(system_coords == np.array([size_x - 1, size_y - 1]), axis=1).reshape(size_x, size_y))
        bnd_tags = np.where(np.logical_and(bnd_tags, in_circle_mask_2D), True, False)

        inner_mask = np.logical_and(in_circle_mask, np.logical_not(bnd_tags.reshape(-1)))

    else:
        inner_mask = np.all((system_coords > 0) & (system_coords < np.array([size_x - 1, size_y - 1])), axis=1)
    
    # Count source point as external
    if src_pt_idx is not None:
        # Convert to flat index
        src_pt_idx = np.array(src_pt_idx)
        src_pt_idx = np.dot(src_pt_idx, np.array([size_x, 1]))
        inner_mask[src_pt_idx] = False

    # Get neighbour indices
    off_diag_idx_left = np.array([mat_1D_indices[1:], mat_1D_indices[:-1]])
    off_diag_idx_right = np.array([mat_1D_indices[:-1], mat_1D_indices[1:]])
    off_diag_idx_top = np.array([mat_1D_indices[size_y:], mat_1D_indices[:-size_y]])
    off_diag_idx_bottom = np.array([mat_1D_indices[:-size_y], mat_1D_indices[size_y:]])
    
    # Reduce array to inner region
    inner_idx_mapping = np.where(inner_mask, np.arange(size_x * size_y), -1)
    inner_idx_mapping[inner_mask] = np.arange(np.sum(inner_mask))

    if src_pt_idx is not None:
        inner_idx_mapping[src_pt_idx] = -2
        # for i in range(size_y):
        #     print(inner_idx_mapping.reshape((size_x, size_y))[i])

    tagged_off_diag_idx_left = inner_idx_mapping[off_diag_idx_left]
    tagged_off_diag_idx_right = inner_idx_mapping[off_diag_idx_right]
    tagged_off_diag_idx_top = inner_idx_mapping[off_diag_idx_top]
    tagged_off_diag_idx_bottom = inner_idx_mapping[off_diag_idx_bottom]

    # Define source neighbour indices
    src_idx_left = off_diag_idx_left[:, np.any(tagged_off_diag_idx_left == -2, axis=0) ]
    src_idx_right = off_diag_idx_right[:, np.any(tagged_off_diag_idx_right == -2, axis=0)]
    src_idx_top = off_diag_idx_top[:, np.any(tagged_off_diag_idx_top == -2, axis=0)]
    src_idx_bottom = off_diag_idx_bottom[:, np.any(tagged_off_diag_idx_bottom == -2, axis=0)]

    src_idx_left = inner_idx_mapping[src_idx_left]
    src_idx_right = inner_idx_mapping[src_idx_right]
    src_idx_top = inner_idx_mapping[src_idx_top]
    src_idx_bottom = inner_idx_mapping[src_idx_bottom]

    src_indices = np.concatenate((src_idx_left[src_idx_left >= 0],
                                  src_idx_right[src_idx_right >= 0],
                                  src_idx_top[src_idx_top >= 0],
                                  src_idx_bottom[src_idx_bottom >= 0]))

    src_indices = set(src_indices)

    # Remove irrelevant indices
    inner_off_diag_idx_left = tagged_off_diag_idx_left[:, np.all(tagged_off_diag_idx_left >= 0, axis=0)]
    inner_off_diag_idx_right = tagged_off_diag_idx_right[:, np.all(tagged_off_diag_idx_right >= 0, axis=0)]
    inner_off_diag_idx_top = tagged_off_diag_idx_top[:, np.all(tagged_off_diag_idx_top >= 0, axis=0)]
    inner_off_diag_idx_bottom = tagged_off_diag_idx_bottom[:, np.all(tagged_off_diag_idx_bottom >= 0, axis=0)]
    
    # Reduce indices and coordinates to inner region
    mat_1D_indices = np.arange(np.sum(inner_mask))
    system_coords = system_coords[inner_mask] - np.array([1, 1])

    # Construct coefficient matrix
    coeffs_matrix = np.zeros((mat_1D_indices.shape[0], mat_1D_indices.shape[0]))

    # Set diagonal to -4
    coeffs_matrix[mat_1D_indices, mat_1D_indices] = -4

    # Set off-diagonals to 1
    coeffs_matrix[tuple(inner_off_diag_idx_left)] = 1
    coeffs_matrix[tuple(inner_off_diag_idx_right)] = 1
    coeffs_matrix[tuple(inner_off_diag_idx_top)] = 1
    coeffs_matrix[tuple(inner_off_diag_idx_bottom)] = 1

    # Construct b vector
    # b_vector = np.where(coeffs_matrix == 1, 1, 0).sum(axis=1) - 4=
    b_vector = np.zeros(mat_1D_indices.shape[0])
    b_vector[list(src_indices)] = -1

    # Convert to sparse matrix
    if sparse:
        coeffs_matrix = sp.dia_matrix(coeffs_matrix)

    return coeffs_matrix, system_coords, b_vector


def solve_eigval_problem(coeff_matrix, factor, la_function=la.eigh, both_ends=False):
    """
    Solve the eigenvalue problem Mv = Kv for the 2D wave equation.
    Arguments:
        coeff_matrix (np.ndarray): the coefficient matrix M.
        factor (float): the factor to multiply the coefficient matrix with.
        la_function (function): the function to use for solving the eigenvalue problem.
        both_ends (bool): whether to solve for both ends of the spectrum. Only works for spla.eigsh.
    Returns:
        np.ndarray: the eigenvalues.
        np.ndarray: the eigenvectors.
    """

    assert la_function in [la.eigh, la.eig, spla.eigs, spla.eigsh], "Invalid LA function."
    assert type(coeff_matrix) in [np.ndarray, sp.dia_matrix], "Invalid coefficient matrix type."
    assert type(factor) in [int, float, np.float64], "Invalid factor type."
    assert (type(coeff_matrix) == sp.dia_matrix and la_function in [spla.eigs, spla.eigsh]) or \
            (type(coeff_matrix) == np.ndarray and la_function in [la.eigh, la.eig]), "Invalid LA function for sparse matrix."
    
    # Solve eigenvalue problem
    if type(coeff_matrix) == sp.dia_matrix:
        if both_ends:
            eigvals, eigvecs = la_function(factor * coeff_matrix, which='BE')
        else:    
            eigvals, eigvecs = la_function(factor * coeff_matrix, which='SM')
    else:
        eigvals, eigvecs = la_function(factor * coeff_matrix)

    return eigvals, eigvecs.T


def compute_wave_time_dependent(init_amp, init_vel, eigvec, eigval, ts):
    """
    Computes the time-dependent solution of a vibrating membrane
    given initial amplitude and velocity, an eigenvector and and eigenvalue.
    Arguments:
        init_amp (np.ndarray): the initial amplitude.
        init_vel (np.ndarray): the initial velocity.
        eigvec (np.ndarray): the eigenvector.
        eigval (float): the eigenvalue.
        ts (np.ndarray): the time steps.
    Returns:
        np.ndarray: the time-dependent solution.
    """

    assert type(init_amp) == np.ndarray, "Invalid initial amplitude type."
    assert type(init_vel) == np.ndarray, "Invalid initial velocity type."
    assert type(eigvec) == np.ndarray, "Invalid eigenvector type."
    assert type(eigval) in [float, int, np.float64], "Invalid eigenvalue type."
    assert type(ts) == np.ndarray, "Invalid time steps type."
    assert init_amp.shape == init_vel.shape == eigvec.shape, "Invalid shape of input arrays."

    # Array for storing snapshots
    u_evolution = np.empty((ts.shape[0], eigvec.shape[0]))

    # Find A and B coefficients
    v_nonzero_mask = eigvec != 0

    A_coeff = init_amp[v_nonzero_mask] / eigvec[v_nonzero_mask]
    A_coeff = set(np.round(A_coeff, 10))
    assert len(A_coeff) == 1 or (len(A_coeff) == 2 and 0 in A_coeff), "Ambiguous A coefficient."
    A_coeff = list(A_coeff)
    A_coeff = A_coeff[np.argmax(np.abs(A_coeff))]

    B_coeff = init_vel[v_nonzero_mask] / (eigvec[v_nonzero_mask] * np.sqrt(-eigval))
    B_coeff = set(np.round(B_coeff, 10))
    assert len(B_coeff) == 1 or (len(B_coeff) == 2 and 0 in B_coeff), "Ambiguous B coefficient."
    B_coeff = list(B_coeff)
    B_coeff = B_coeff[np.argmax(np.abs(B_coeff))]

    # Time-dependent solution
    for i, t in enumerate(ts):
        u_evolution[i] = A_coeff * np.cos(np.sqrt(-eigval) * t) * eigvec + B_coeff * np.sin(np.sqrt(-eigval) * t) * eigvec
    
    return u_evolution


def compute_wave_time_dependent_multiple(init_amps, init_vels, eigvecs, eigvals, ts):
    """
    Computes the time-dependent solution of multiple vibrating membranes
    given initial amplitudes and velocities, eigenvectors and eigenvalues.
    Arguments:
        init_amps (np.ndarray): the initial amplitudes.
        init_vels (np.ndarray): the initial velocities.
        eigvecs (np.ndarray): the eigenvectors.
        eigvals (np.ndarray): the eigenvalues.
        ts (np.ndarray): the time steps.
    Returns:
        np.ndarray: the time-dependent solutions.
    """

    assert type(init_amps) == list, "Invalid initial amplitudes type."
    assert type(init_vels) == list, "Invalid initial velocities type."
    assert type(eigvecs) == list, "Invalid eigenvectors type."
    assert type(eigvals) == list, "Invalid eigenvalues type."
    assert type(ts) == np.ndarray, "Invalid time steps type."
    assert len(init_amps) == len(init_vels) == len(eigvecs) == len(eigvals), "Invalid shape of input arrays."

    # Array for storing snapshots
    u_evolutions = []

    # Loop over membrane types
    for i in range(len(eigvecs)):
        u_evolutions_bndry = np.empty((eigvecs[i].shape[0], ts.shape[0], eigvecs[i].shape[1]))
        # Loop over eigenvalues
        for j in range(eigvecs[i].shape[0]):
            u_evolutions_bndry[j] = compute_wave_time_dependent(init_amps[i][j], init_vels[i][j], eigvecs[i][j], eigvals[i][j], ts)
        u_evolutions.append(u_evolutions_bndry)
    
    return u_evolutions

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

    # if boundary_type == 'circle':
    #     size_x = size_y = min(size_x, size_y)

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

    # Get boundary indices
    # bnd_1D_indices = mat_1D_indices[(mat_2D_indices[:, 0] == 0) | (mat_2D_indices[:, 0] == size_x - 1) | (mat_2D_indices[:, 1] == 0) | (mat_2D_indices[:, 1] == size_y - 1)]

    # Get neighbour indices
    off_diag_indices_left = np.array([mat_1D_indices[1:], mat_1D_indices[:-1]])
    off_diag_indices_right = np.array([mat_1D_indices[:-1], mat_1D_indices[1:]])
    off_diag_indices_top = np.array([mat_1D_indices[size_y:], mat_1D_indices[:-size_y]])
    off_diag_indices_bottom = np.array([mat_1D_indices[:-size_y], mat_1D_indices[size_y:]])

    # Remove neighbours from boundaries
    off_diag_indices_left = off_diag_indices_left.T[np.logical_not(np.any(np.isin(off_diag_indices_left, bnd_1D_indices), axis=0))].T
    off_diag_indices_right = off_diag_indices_right.T[np.logical_not(np.any(np.isin(off_diag_indices_right, bnd_1D_indices), axis=0))].T
    off_diag_indices_top = off_diag_indices_top.T[np.logical_not(np.any(np.isin(off_diag_indices_top, bnd_1D_indices), axis=0))].T
    off_diag_indices_bottom = off_diag_indices_bottom.T[np.logical_not(np.any(np.isin(off_diag_indices_bottom, bnd_1D_indices), axis=0))].T

    # Reduce array to cirular region
    if boundary_type == 'circle':
        idx_mapping = np.where(in_circle_mask, np.arange(size_x * size_y), -1)
        idx_mapping[in_circle_mask] = np.arange(np.sum(in_circle_mask))

        off_diag_indices_left = idx_mapping[off_diag_indices_left]
        off_diag_indices_right = idx_mapping[off_diag_indices_right]
        off_diag_indices_top = idx_mapping[off_diag_indices_top]
        off_diag_indices_bottom = idx_mapping[off_diag_indices_bottom]

        off_diag_indices_left = off_diag_indices_left[:, np.all(off_diag_indices_left != -1, axis=0)]
        off_diag_indices_right = off_diag_indices_right[:, np.all(off_diag_indices_right != -1, axis=0)]
        off_diag_indices_top = off_diag_indices_top[:, np.all(off_diag_indices_top != -1, axis=0)]
        off_diag_indices_bottom = off_diag_indices_bottom[:, np.all(off_diag_indices_bottom != -1, axis=0)]
        
        mat_1D_indices = np.arange(np.sum(in_circle_mask))
        mat_2D_indices = mat_2D_indices[in_circle_mask]

    # Construct coefficient matrix
    coeffs_matrix = np.zeros((mat_1D_indices.shape[0], mat_1D_indices.shape[0]))

    # Set diagonal to -4
    coeffs_matrix[mat_1D_indices, mat_1D_indices] = -4

    # Set off-diagonals to 1
    coeffs_matrix[tuple(off_diag_indices_left)] = 1
    coeffs_matrix[tuple(off_diag_indices_right)] = 1
    coeffs_matrix[tuple(off_diag_indices_top)] = 1
    coeffs_matrix[tuple(off_diag_indices_bottom)] = 1

    # Convert to sparse matrix
    if sparse:
        coeffs_matrix = sp.dia_matrix(coeffs_matrix)

    return coeffs_matrix, mat_2D_indices


def construct_coeff_matrix_alt(size_x, size_y, boundary_type='rect', sparse=False):
    """
    Construct the coefficient matrix A for the eigenvalue problem
    of the 2D wave equation with a boundary fixed to zero.
    Uses a specific boundary shape, if circle is chosen,
    smallest of the two dimensions is used as the radius.
    Arguments:
        size_x (int): number of points in the x direction.
        size_y (int): number of points in the y direction.
        boundary_type (str): type of boundary condition. Can be 'rect' or 'circ'.
        sparse (bool): whether to return a sparse matrix.
    Returns:
        np.ndarray: the coefficient matrix M.
    """

    assert boundary_type in ['rect', 'circle'], "Invalid boundary type."

    # if boundary_type == 'circle':
    #     size_x = size_y = min(size_x, size_y)

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

        # Get boundary indices
        bnd_1D_indices = mat_1D_indices[bnd_tags.reshape(-1)]

        inner_mask = np.logical_and(in_circle_mask, np.logical_not(bnd_tags.reshape(-1)))

    else:
        inner_mask = np.all((system_coords > 0) & (system_coords < np.array([size_x - 1, size_y - 1])), axis=1)
        # Get boundary indices
        bnd_1D_indices = mat_1D_indices[np.any((system_coords == 0) | (system_coords == np.array([size_x - 1, size_y - 1])), axis=1)]
    
    # Get neighbour indices
    off_diag_indices_left = np.array([mat_1D_indices[1:], mat_1D_indices[:-1]])
    off_diag_indices_right = np.array([mat_1D_indices[:-1], mat_1D_indices[1:]])
    off_diag_indices_top = np.array([mat_1D_indices[size_y:], mat_1D_indices[:-size_y]])
    off_diag_indices_bottom = np.array([mat_1D_indices[:-size_y], mat_1D_indices[size_y:]])

    # Remove neighbours from boundaries
    off_diag_indices_left = off_diag_indices_left.T[np.logical_not(np.any(np.isin(off_diag_indices_left, bnd_1D_indices), axis=0))].T
    off_diag_indices_right = off_diag_indices_right.T[np.logical_not(np.any(np.isin(off_diag_indices_right, bnd_1D_indices), axis=0))].T
    off_diag_indices_top = off_diag_indices_top.T[np.logical_not(np.any(np.isin(off_diag_indices_top, bnd_1D_indices), axis=0))].T
    off_diag_indices_bottom = off_diag_indices_bottom.T[np.logical_not(np.any(np.isin(off_diag_indices_bottom, bnd_1D_indices), axis=0))].T

    # Reduce array to cirular region
    # if boundary_type == 'circle':
    #     idx_mapping = np.where(in_circle_mask, np.arange(size_x * size_y), -1)
    #     idx_mapping[in_circle_mask] = np.arange(np.sum(in_circle_mask))

    #     off_diag_indices_left = idx_mapping[off_diag_indices_left]
    #     off_diag_indices_right = idx_mapping[off_diag_indices_right]
    #     off_diag_indices_top = idx_mapping[off_diag_indices_top]
    #     off_diag_indices_bottom = idx_mapping[off_diag_indices_bottom]

    #     off_diag_indices_left = off_diag_indices_left[:, np.all(off_diag_indices_left != -1, axis=0)]
    #     off_diag_indices_right = off_diag_indices_right[:, np.all(off_diag_indices_right != -1, axis=0)]
    #     off_diag_indices_top = off_diag_indices_top[:, np.all(off_diag_indices_top != -1, axis=0)]
    #     off_diag_indices_bottom = off_diag_indices_bottom[:, np.all(off_diag_indices_bottom != -1, axis=0)]
        
    #     mat_1D_indices = np.arange(np.sum(in_circle_mask))
    #     system_coords = system_coords[in_circle_mask]
    
    idx_mapping = np.where(inner_mask, np.arange(size_x * size_y), -1)
    idx_mapping[inner_mask] = np.arange(np.sum(inner_mask))

    off_diag_indices_left = idx_mapping[off_diag_indices_left]
    off_diag_indices_right = idx_mapping[off_diag_indices_right]
    off_diag_indices_top = idx_mapping[off_diag_indices_top]
    off_diag_indices_bottom = idx_mapping[off_diag_indices_bottom]

    off_diag_indices_left = off_diag_indices_left[:, np.all(off_diag_indices_left != -1, axis=0)]
    off_diag_indices_right = off_diag_indices_right[:, np.all(off_diag_indices_right != -1, axis=0)]
    off_diag_indices_top = off_diag_indices_top[:, np.all(off_diag_indices_top != -1, axis=0)]
    off_diag_indices_bottom = off_diag_indices_bottom[:, np.all(off_diag_indices_bottom != -1, axis=0)]
    
    mat_1D_indices = np.arange(np.sum(inner_mask))
    system_coords = system_coords[inner_mask] - np.array([1, 1])

    # Construct coefficient matrix
    coeffs_matrix = np.zeros((mat_1D_indices.shape[0], mat_1D_indices.shape[0]))

    # Set diagonal to -4
    coeffs_matrix[mat_1D_indices, mat_1D_indices] = -4

    # Set off-diagonals to 1
    coeffs_matrix[tuple(off_diag_indices_left)] = 1
    coeffs_matrix[tuple(off_diag_indices_right)] = 1
    coeffs_matrix[tuple(off_diag_indices_top)] = 1
    coeffs_matrix[tuple(off_diag_indices_bottom)] = 1

    # Convert to sparse matrix
    if sparse:
        coeffs_matrix = sp.dia_matrix(coeffs_matrix)

    return coeffs_matrix, system_coords


def solve_eigval_problem(coeff_matrix, factor, la_function=la.eigh):
    """
    Solve the eigenvalue problem Mv = Kv for the 2D wave equation.
    Arguments:
        coeff_matrix (np.ndarray): the coefficient matrix M.
        factor (float): the factor to multiply the coefficient matrix with.
        la_function (function): the function to use for solving the eigenvalue problem.
    Returns:
        np.ndarray: the eigenvalues.
        np.ndarray: the eigenvectors.
    """

    assert la_function in [la.eigh, la.eig, spla.eigs, spla.eigsh], "Invalid LA function."
    assert type(coeff_matrix) in [np.ndarray, sp.dia_matrix], "Invalid coefficient matrix type."
    assert type(factor) in [int, float], "Invalid factor type."
    
    # Solve eigenvalue problem
    eigvals, eigvecs = la_function(factor * coeff_matrix)
    # if type(coeff_matrix) == sp.dia_matrix:
    #     eigvals, eigvecs = la_function(coeff_matrix, k=coeff_matrix.shape[0] - 2)
    # else:
    #     eigvals, eigvecs = la_function(factor * coeff_matrix)

    return eigvals, eigvecs
import numpy as np

def construct_coeff_matrix(size_x, size_y, boundary_type='rect'):
    """
    Construct the coefficient matrix A for the eigenvalue problem
    Mv = Kv of the 2D wave equation.
    Arguments:
        size_x (int): number of points in the x direction.
        size_y (int): number of points in the y direction.
        boundary_type (str): type of boundary condition. Can be 'rect' or 'circ'.
    Returns:
        np.ndarray: the coefficient matrix M.
    """

    # Matrix indices
    mat_1D_indices = np.arange(size_x * size_y)
    mat_2D_indices = np.array([i for i in np.ndindex((size_x, size_y))])

    # Get boundary indices
    bnd_1D_indices = mat_1D_indices[(mat_2D_indices[:, 0] == 0) | (mat_2D_indices[:, 0] == size_x - 1) | (mat_2D_indices[:, 1] == 0) | (mat_2D_indices[:, 1] == size_y - 1)]
    
    # Construct coefficient matrix
    coeffs_matrix = np.zeros((size_x * size_y, size_x * size_y))

    # Set diagonal to -4
    coeffs_matrix[mat_1D_indices, mat_1D_indices] = -4

    # Get neighbour indices
    off_diag_indices_left = np.array([mat_1D_indices[1:], mat_1D_indices[:-1]])
    off_diag_indices_right = np.array([mat_1D_indices[:-1], mat_1D_indices[1:]])
    off_diag_indices_top = np.array([mat_1D_indices[size_x:], mat_1D_indices[:-size_x]])
    off_diag_indices_bottom = np.array([mat_1D_indices[:-size_x], mat_1D_indices[size_x:]])

    # Remove neighbours from boundaries
    off_diag_indices_left = off_diag_indices_left.T[np.logical_not(np.any(np.isin(off_diag_indices_left, bnd_1D_indices), axis=0))].T
    off_diag_indices_right = off_diag_indices_right.T[np.logical_not(np.any(np.isin(off_diag_indices_right, bnd_1D_indices), axis=0))].T
    off_diag_indices_top = off_diag_indices_top.T[np.logical_not(np.any(np.isin(off_diag_indices_top, bnd_1D_indices), axis=0))].T
    off_diag_indices_bottom = off_diag_indices_bottom.T[np.logical_not(np.any(np.isin(off_diag_indices_bottom, bnd_1D_indices), axis=0))].T

    # Set off-diagonals to 1
    coeffs_matrix[tuple(off_diag_indices_left)] = 1
    coeffs_matrix[tuple(off_diag_indices_right)] = 1
    coeffs_matrix[tuple(off_diag_indices_top)] = 1
    coeffs_matrix[tuple(off_diag_indices_bottom)] = 1

    return coeffs_matrix
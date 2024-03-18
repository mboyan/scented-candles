import numpy as np
import matplotlib.pyplot as plt

def plot_lattice_topology(size_x, size_y):
    """
    Plots the index map of a 2D lattice.
    Arguments:
        size_x (int): number of points in the x direction.
        size_y (int): number of points in the y direction.
    """

    xs = np.arange(size_x)
    ys = np.arange(size_y)

    fig, ax = plt.subplots()
    fig.set_size_inches(3, 3)
    ax.set_aspect('equal', 'box')
    
    # Plot vertical grid lines
    for x in xs[1:-1]:
        ax.plot([x, x], [ys[0], ys[1]], color='black', ls='--')
        ax.plot([x, x], [ys[1], ys[-2]], color='black')
        ax.plot([x, x], [ys[-2], ys[-1]], color='black', ls='--')

    # Plot horizontal grid lines
    for y in ys[1:-1]:
        ax.plot([xs[0], xs[1]], [y, y], color='black', ls='--')
        ax.plot([xs[1], xs[-2]], [y, y], color='black')
        ax.plot([xs[-2], xs[-1]], [y, y], color='black', ls='--')

    # Plot dots at grid nodes
    for x in xs:
        for y in ys:
            if x == 0 or x == size_x-1 or y == 0 or y == size_y-1:
                col = 'r'
            else:
                col = 'w'
            ax.scatter(x, y, color=col, marker='o', edgecolors='k')
            ax.text(x+0.25/size_x, y+0.25/size_y, f'$i={y*size_x+x}$', ha='left', va='bottom', fontsize=8)
    
    ax.axis('off')
    plt.show()


def plot_lattice_topo_from_coeff(coeff_matrix, lattice_coords, ax=None):
    """
    Plots the index map of a 2D lattice based on
    a coefficient matrix and the coordinates of the lattice points.
    Arguments:
        coeff_matrix (np.ndarray): the coefficient matrix.
        lattice_coords (np.ndarray): the coordinates of the lattice points.
        ax (matplotlib.axes.Axes, optional): the axes to plot on. If None, a new figure is created.
    """

    size_x = np.max(lattice_coords[:, 1]) + 1
    size_y = np.max(lattice_coords[:, 0]) + 1

    lattice_coords = np.flip(lattice_coords, axis=1)

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal', 'box')

    # Plot lattice points
    row_sums = np.sum(coeff_matrix, axis=1)
    # ax.scatter(lattice_coords[:, 0], lattice_coords[:, 1], color='w', edgecolors='k')
    for i, (x, y) in enumerate(lattice_coords):
        if row_sums[i] == -4:
            col = 'r'
        else:
            col = 'w'
            nbr_indices = np.where(coeff_matrix[i] == 1)[0]
            for nbr in nbr_indices:
                ax.plot([x, lattice_coords[nbr, 0]], [y, lattice_coords[nbr, 1]], color='black')
        ax.scatter(x, y, color=col, marker='o', edgecolors='k')
        ax.text(x+0.25/size_x, y+0.25/size_y, f'$i={i}$', ha='left', va='bottom', fontsize=8)
    
    ax.set_xlabel('j')
    ax.set_ylabel('k')
    ax.set_xticks(np.arange(size_x))
    ax.set_yticks(np.arange(size_y))
    ax.grid(True)
    # ax.axis('off')
    if ax is None:
        plt.show()


def plot_coeff_matrix(coeff_matrix, ax=None):
    """
    Plots the coefficient matrix.
    Arguments:
        coeff_matrix (np.ndarray): the coefficient matrix.
        ax (matplotlib.axes.Axes, optional): the axes to plot on. If None, a new figure is created.
    """

    # Get non-zero value positions
    non_zero_pos = np.array(np.where(coeff_matrix != 0))

    # Plot matrix
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal', 'box')
    ax.pcolor(np.abs(np.flip(coeff_matrix, axis=0)), cmap='coolwarm', edgecolors='k')

    # Tag non-zero values
    for pos in non_zero_pos.T:
        ax.text(pos[1] + 0.5, coeff_matrix.shape[0] - pos[0] - 0.5, f'{int(coeff_matrix[pos[0], pos[1]])}', ha='center', va='center', fontsize=8)

    # Set ticks
    ax.set_xticks(np.arange(coeff_matrix.shape[0], step=2) + 0.5)
    ax.set_yticks(np.arange(coeff_matrix.shape[1], step=2) + 0.5)
    ax.set_xticklabels(np.arange(coeff_matrix.shape[0], step=2))
    ax.set_yticklabels(np.arange(coeff_matrix.shape[1] - 1, -1, -2))
    ax.set_xlabel('q')
    ax.set_ylabel('i')
    
    if ax is None:
        plt.show()


def plot_lattice_2D(coeff_matrix, lattice_coords):
    """
    Creates a combined plot of the lattice topology and the coefficient matrix.
    Arguments:
        coeff_matrix (np.ndarray): the coefficient matrix.
        lattice_coords (np.ndarray): the coordinates of the lattice points.
    """

    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(4, 8)
    plot_lattice_topo_from_coeff(coeff_matrix, lattice_coords, ax=ax[0])
    plot_coeff_matrix(coeff_matrix, ax=ax[1])

    ax[0].set_title('Lattice Topology')
    ax[1].set_title('Coefficient Matrix')

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib import animation
from IPython.display import HTML

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


def plot_eigmode(eigmode, eigfreq, lattice_coords, ax=None, vmin=None, vmax=None):
    """
    Plots an eigenvector as a 2D scalar field.
    Arguments:
        eigmode (np.ndarray): the eigenvector.
        eigfreq (float): the eigenfrequency.
        lattice_coords (np.ndarray): the coordinates of the lattice points.
        ax (matplotlib.axes.Axes, optional): the axes to plot on. If None, a new figure is created.
        vmin (float, optional): the minimum value of the color scale.
        vmax (float, optional): the maximum value of the color scale.
    Returns:
        matplotlib.image.AxesImage: the image object.
    """

    assert eigmode.ndim == 1, 'Eigenvector must be 1D.'
    assert lattice_coords.ndim == 2, 'Lattice coordinates must be 2D.'
    assert lattice_coords.shape[1] == 2, 'Lattice coordinates must contain 2 elements.'
    assert eigmode.shape[0] == lattice_coords.shape[0], 'Eigenvector and lattice coordinates must have the same length.'

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal', 'box')

    size_x = np.max(lattice_coords[:, 0]) + 2
    size_y = np.max(lattice_coords[:, 1]) + 2

    scalar_field = np.zeros((size_x, size_y))
    for i, (x, y) in enumerate(lattice_coords):
        scalar_field[x + 1, y + 1] = eigmode[i]
    
    scalar_field = scalar_field.T
    
    if vmin is None:
        vmin = np.min(scalar_field)
    if vmax is None:
        vmax = np.max(scalar_field)
    vmin, vmax = np.min([vmin, -vmax]), np.max([-vmin, vmax])
    im = ax.imshow(scalar_field, cmap='bwr', interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
    
    if ax is None:
        ax.set_xticks(np.arange(size_x, step=max(1, size_x//5)))
        ax.set_yticks(np.arange(size_y, step=max(1, size_y//5)))
        ax.set_xlabel('j')
        ax.set_ylabel('k')
        ax.set_title(f'$K={eigfreq}$')
        plt.colorbar(im)
        plt.show()
    
    return im


def plot_eigmodes_spectrum(eigmodes, eigfreqs, lattice_coords, column_titles=None):
    """
    Plots the spectrum of eigenvectors.
    Arguments:
        eigmodes (list): a list of eigenvectors.
        eigfreqs (list): a list of eigenfrequencies.
        lattice_coords (np.ndarray): the coordinates of the lattice points.
        column_titles (list, optional): a list of titles for each column.
    """

    assert type(eigmodes) == list, 'Eigenvectors must be a list.'
    assert type(eigfreqs) == list, 'Eigenfrequencies must be a list.'
    assert len(eigmodes) == len(eigfreqs), 'Eigenvectors and eigenfrequencies must have the same length.'
    assert type(lattice_coords) == list, 'Lattice coordinates must be a list.'

    fig, axs = plt.subplots(eigmodes[0].shape[0], len(eigmodes), sharex=True)
    fig.set_size_inches(4, 4 / len(eigmodes) * eigmodes[0].shape[0])
    fig.subplots_adjust(hspace=0.5)

    vmin = min([np.min(eigmode) for eigmode in eigmodes])
    vmax = max([np.max(eigmode) for eigmode in eigmodes])

    for i, (eigmode, eigfreq, coords) in enumerate(zip(eigmodes, eigfreqs, lattice_coords)):
        for j in range(eigmode.shape[0]):
            img = plot_eigmode(eigmode[j], eigfreq[j], coords, ax=axs[j, i], vmin=vmin, vmax=vmax)
            axs[j, i].set_xticks([0, np.ceil(max(coords[:, 0]) / 2 + 1), max(coords[:, 0]) + 2])
            axs[j, i].set_yticks([0, np.ceil(max(coords[:, 1]) / 2 + 1), max(coords[:, 1]) + 2])

            if j == 0 and column_titles is not None:
                title_add = column_titles[i] + '\n'
            else:
                title_add = ''
            axs[j, i].set_title(title_add + f"$K={'{:.3f}'.format(eigfreq[j])}$", fontsize=10)
            axs[j, i].margins(y=0.01)

    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
    cbar_ax.set_title('$v(x,y)$', fontsize=10)
    fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
    # cbar_ax.text(-0.01, 0.5, '$v(x,y)$', transform=cbar_ax.transAxes, ha='right', va='center')

    # plt.tight_layout()
    plt.show()


def plot_eigvals_Ls(result_setups, eigvals):
    """
    Creates a series of plots of the eigenvalues
    as a function of the lattice size,
    for different boundary shapes.
    Arguments:
        result_setups (pd.DataFrame): the experimental setups containing the boundary shapes and Ls.
        eigvals (np.ndarray): the eigenvalues.
    """

    assert type(result_setups) == pd.DataFrame, 'Result setups must be a DataFrame.'

    Ls = result_setups['L'].unique()
    boundaries = result_setups['Boundary'].unique()

    assert eigvals.ndim == 2, 'Eigenvalues must be 2D.'

    fig, axs = plt.subplots(boundaries.shape[0], 1)
    fig.set_size_inches(3.5, 3.25 * boundaries.shape[0])

    for i, boundary in enumerate(boundaries):

        bndry_mask = result_setups['Boundary'] == boundary
        eigval_selection = -eigvals[bndry_mask]

        eigval_indices = np.arange(eigval_selection.shape[1])

        Ls_grid, eigval_indices = np.meshgrid(Ls, eigval_indices)

        df_plot = pd.DataFrame({
            '$L$': Ls_grid.T.flatten(),
            '$K$ index': eigval_indices.T.flatten(),
            '$|K|$': eigval_selection.flatten()
        })

        sns.lineplot(data=df_plot, x='$L$', y='$|K|$', hue='$K$ index', ax=axs[i])
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_title(f'Boundary: {boundary}')

    plt.tight_layout()
    plt.show()


def plot_eigvals_Ns(result_setups, eigvals):
    """
    Plots the interval between the largest and smallest
    eigenvalues as a function of the subdivision density,
    for different boundary shapes.
    Arguments:
        result_setups (pd.DataFrame): the experimental setups containing the boundary shapes and Ns.
        eigvals (np.ndarray): the eigenvalues.
    """

    assert type(result_setups) == pd.DataFrame, 'Result setups must be a DataFrame.'

    Ns = result_setups['N'].unique()
    boundaries = result_setups['Boundary'].unique()

    assert eigvals.ndim == 2, 'Eigenvalues must be 2D.'

    fig, axs = plt.subplots(boundaries.shape[0], 1)
    fig.set_size_inches(3.5, 3.25 * boundaries.shape[0])

    for i, boundary in enumerate(boundaries):

        bndry_mask = result_setups['Boundary'] == boundary
        eigval_selection = -eigvals[bndry_mask]

        axs[i].plot(Ns, np.min(eigval_selection, axis=1), label='$|K|_{min}$')
        axs[i].plot(Ns, np.max(eigval_selection, axis=1), label='$|K|_{max}$')
        # axs[i].fill_between(Ns.astype(int), np.min(eigval_selection, axis=1), np.max(eigval_selection, axis=1), alpha=0.5)
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_xlabel('$N$')
        axs[i].set_ylabel('$|K|$ range')
        axs[i].set_title(f'Boundary: {boundary}')
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def animate_membranes(frames, times, interval=10):
    """
    Creates a 3D animation of a series of vibrating membranes from the provided frames.
    input:
        frames (numpy.ndarray) - a 3D array containing the frames for the animation of each string
        times (numpy.ndarray) - an array with the corresponding timesteps
    output:
        an HTML animation
    """

    assert frames.ndim == 3, 'frames must be a 3D array'
    assert frames.shape[1] == times.shape[0], 'frames and timesteps must have matching shapes'

    n_frames = frames.shape[1]

    fig, axs = plt.subplots(1, frames.shape[0], sharex=True, sharey=True)
    fig.set_size_inches(8, 7/frames.shape[0])
    plots = [axs[i].plot(np.linspace(0, 1, frames.shape[2]), frames[i][0])[0] for i in range(frames.shape[0])]
    # plot, = ax.plot(np.linspace(0, 1, frames.shape[1]), frames[0])
    time_txt = [ax.text(0.05, 0.05, '', transform=ax.transAxes) for ax in axs]

    def init_anim():
        """
        Initialize animation
        """
        for j, plot in enumerate(plots):
            plot.set_ydata(frames[j][0])
            time_txt[j].set_text('')
            axs[j].set_xlabel('$x$')
            axs[j].set_ylabel('$u(x,t)$')
            axs[j].set_title(f'Init. condition {j+1}')
            axs[j].grid()
        return *plots, *time_txt

    def update(i):
        """
        Update animation
        """
        for j, plot in enumerate(plots):
            plot.set_ydata(frames[j][i])
            time_txt[j].set_text(f'$t={times[i]:.3f}$')
        return *plots, *time_txt

    anim = animation.FuncAnimation(fig, update, init_func=init_anim, frames=n_frames, interval=interval, blit=True)

    plt.tight_layout()
    plt.show()

    return HTML(anim.to_html5_video())


def plot_diff_eq_solution(c_vals, coords, src_idx=None):
    """
    Plots the solution of the diffusion equation.
    Arguments:
        c_vals (np.ndarray): the solution values.
        coords (np.ndarray): the corresponding coordinates of the lattice points.
        src_coords (np.ndarray, optional): the coordinates of the source point. Default is None.
    """

    assert c_vals.ndim == 1, 'Solution values must be 1D.'
    assert coords.ndim == 2, 'Lattice coordinates must be 2D.'
    assert c_vals.shape[0] == coords.shape[0], 'Solution values and lattice coordinates must have the same length.'

    size_x = np.max(coords[:, 0]) + 2
    size_y = np.max(coords[:, 1]) + 2

    scalar_field = np.zeros((size_x, size_y))
    for i, (x, y) in enumerate(coords):
        scalar_field[x + 1, y + 1] = c_vals[i]

    if src_idx is not None:
        scalar_field[src_idx[0], src_idx[1]] = 1

    scalar_field = scalar_field.T

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    ax.set_aspect('equal', 'box')

    im = ax.imshow(scalar_field, cmap='plasma', interpolation='nearest', origin='lower')

    cont = ax.contour(scalar_field, levels=np.linspace(0, 1, 6)**2, cmap='YlOrRd', alpha=0.5)
    ax.clabel(cont, inline=True, fontsize=8)

    ax.set_xticks(np.arange(size_x, step=max(1, size_x//5)))
    ax.set_yticks(np.arange(size_y, step=max(1, size_y//5)))
    ax.set_xlabel('j')
    ax.set_ylabel('k')
    ax.set_title('Diffusion Solution')
    plt.colorbar(im)

    plt.show()
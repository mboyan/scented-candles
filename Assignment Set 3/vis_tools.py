import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
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


def plot_lattice_2D(coeff_matrix, lattice_coords, horizontal=False):
    """
    Creates a combined plot of the lattice topology and the coefficient matrix.
    Arguments:
        coeff_matrix (np.ndarray): the coefficient matrix.
        lattice_coords (np.ndarray): the coordinates of the lattice points.
        horizontal (bool, optional): whether to align the plots horizontally. Default is False.
    """

    if horizontal:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(4, 2)
        pad = 10
    else:
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(4, 8)
        pad = 0
    plot_lattice_topo_from_coeff(coeff_matrix, lattice_coords, ax=ax[0])
    plot_coeff_matrix(coeff_matrix, ax=ax[1])

    ax[0].set_title('Lattice Topology', pad=pad)
    ax[1].set_title('Coefficient Matrix', pad=pad)

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

    # fig, axs = plt.subplots(boundaries.shape[0], 1)
    # fig.set_size_inches(3.5, 3.25 * boundaries.shape[0])
    fig, axs = plt.subplots(1, boundaries.shape[0])
    fig.set_size_inches(7, 6 / boundaries.shape[0])
    fig.subplots_adjust(wspace=0.3)

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

        if i == len(boundaries) - 1:
            legend = True
        else:
            legend = False

        sns.lineplot(data=df_plot, x='$L$', y='$|K|$', hue='$K$ index', ax=axs[i], legend=legend)
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_title(boundary)

        if legend:
            leg = axs[i].legend(ncol=2, loc='lower left', bbox_to_anchor=(0, 0), columnspacing=0.5)
            leg.set_title('$K$ index', prop={'size': 10})
            for t in leg.texts:
                t.set_fontsize(8)

        if i > 0:
            axs[i].set(ylabel='')

    # plt.tight_layout()
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

    # fig, axs = plt.subplots(boundaries.shape[0], 1)
    # fig.set_size_inches(3.5, 3.25 * boundaries.shape[0])
    fig, axs = plt.subplots(1, boundaries.shape[0])
    fig.set_size_inches(6, 5 / boundaries.shape[0])
    fig.subplots_adjust(wspace=0.3)

    for i, boundary in enumerate(boundaries):

        bndry_mask = result_setups['Boundary'] == boundary
        eigval_selection = -eigvals[bndry_mask]

        axs[i].plot(Ns, np.min(eigval_selection, axis=1), label='$|K|_{min}$')
        axs[i].plot(Ns, np.max(eigval_selection, axis=1), label='$|K|_{max}$')
        # axs[i].fill_between(Ns.astype(int), np.min(eigval_selection, axis=1), np.max(eigval_selection, axis=1), alpha=0.5)
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_xlabel('$N$')
        if i == 0:
            axs[i].set_ylabel('$|K|$ range')
        axs[i].set_title(boundary)
        
        if i == len(boundaries) - 1:
            axs[i].legend(fontsize=9, loc='center right', bbox_to_anchor=(1, 0.3))

    # plt.tight_layout()
    plt.show()


def plot_init_condition(init_coords, init_vals):
    """
    Plots the initial condition of a vibrating membrane.
    Arguments:
        init_coords (np.ndarray): the coordinates of the lattice points.
        init_vals (np.ndarray): the initial values.
    """

    assert init_coords.ndim == 2, 'Initial coordinates must be 2D.'
    assert init_vals.ndim == 1, 'Initial values must be 1D.'
    assert init_coords.shape[0] == init_vals.shape[0], 'Initial coordinates and values must have the same length.'

    size_x = np.max(init_coords[:, 0]) + 2
    size_y = np.max(init_coords[:, 1]) + 2

    scalar_field = np.zeros((size_x, size_y))
    for i, (x, y) in enumerate(init_coords):
        scalar_field[x + 1, y + 1] = init_vals[i]

    scalar_field = scalar_field.T

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    ax.set_aspect('equal', 'box')

    im = ax.imshow(scalar_field, cmap='plasma', interpolation='nearest', origin='lower')
    ax.set_xticks(np.arange(size_x, step=max(1, size_x//5)))
    ax.set_yticks(np.arange(size_y, step=max(1, size_y//5)))
    ax.set_xlabel('j')
    ax.set_ylabel('k')
    ax.set_title('Initial Condition')
    
    plt.colorbar(im)
    plt.show()


def plot_init_condition_3D_multiple(init_coords, init_vals, bndry_labels):
    """
    Creates a series of 3D plots of the initial conditions
    of different membrane shapes.
    Arguments:
        init_coords (list): the coordinates of the lattice points.
        init_vals (list): the initial values.
        bndry_label (str): the boundary labels.
    """

    assert type(init_coords) == list, 'Initial coordinates must be a list.'
    assert type(init_vals) == list, 'Initial values must be a list.'
    assert type(bndry_labels) == list, 'Boundary labels must be a list.'

    fig, axs = plt.subplots(len(init_vals), init_vals[0].shape[0], subplot_kw={'projection': '3d'})
    fig.set_size_inches(4.5, 6 * len(init_vals) / init_vals[0].shape[0])

    for i, (coords, bndry_label) in enumerate(zip(init_coords, bndry_labels)):
        
        size_x = np.max(coords[:, 0]) + 2
        size_y = np.max(coords[:, 1]) + 2

        if bndry_label == 'rectangle':
            xs = np.linspace(0, 0.5, size_x)
        else:
            xs = np.linspace(0, 1, size_x)
        ys = np.linspace(0, 1, size_y)

        for j in range(init_vals[i].shape[0]):

            X, Y = np.meshgrid(xs, ys)
            scalar_field = np.full((size_x, size_y), np.nan)
            for k, (x, y) in enumerate(coords):
                scalar_field[x + 1, y + 1] = init_vals[i][j][k]
            Z = np.ma.masked_where(np.isnan(scalar_field), scalar_field).T

            # col = plt.cm.summer(j / vals.shape[0])
            # face_cols = np.full((Z.shape[0], Z.shape[1], 4), col)

            axs[j, i].plot_surface(X, Y, Z, cmap='plasma', edgecolor='k', linewidth=0.05)
            axs[j, i].set_xticks([0, 0.5, 1])
            axs[j, i].set_yticks([0, 0.5, 1])
            axs[j, i].set_xlim(0, 1)
            axs[j, i].set_ylim(0, 1)
            axs[j, i].set_zlim(-1, 1)

            # axs[j, i].zaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))
            axs[j, i].tick_params(axis='x', which='major', pad=-3, labelsize=7)
            axs[j, i].tick_params(axis='y', which='major', pad=-3, labelsize=7)
            axs[j, i].tick_params(axis='z', which='major', pad=-0.5, labelsize=7)
            axs[j, i].set_title(f'{bndry_label}:\nInit. condition {j+1}', fontsize=9)
            # axs[i, j].set_xlabel('j')
            # axs[i, j].set_ylabel('k')
            # axs[i, j].set_zlabel('$u(x,y)$')

    plt.show()


def plot_time_dependent_solutions(u_evolutions, plot_times, lattice_coords, bndry_labels, eigval_labels, time_tags):
    """
    Creates a series of 3D plots of specific time steps
    of the time-dependent solutions for different membrane shapes.
    Arguments:
        u_evolutions (list): the time-dependent solutions.
        plot_times (list): the times to plot.
        lattice_coords (list): the coordinates of the lattice points.
        bndry_labels (list): the boundary labels.
        eigval_labels (list): the eigenvalue labels.
        time_tags (list): the timestamps for each time step.
    """

    assert len(bndry_labels) == len(u_evolutions) == len(lattice_coords) == len(eigval_labels), 'Boundary labels, time-dependent solutions, and lattice coordinates must have the same length.'
    assert time_tags.ndim == 1, 'Time tags must be 1D.'
    assert time_tags.shape[0] == u_evolutions[0].shape[1], 'Time tags and time-dependent solutions must have the same length.'

    fig, axs = plt.subplots(u_evolutions[0].shape[0], len(bndry_labels), subplot_kw={'projection': '3d'})
    fig.set_size_inches(4.5, 4.5 * u_evolutions[0].shape[0] / len(bndry_labels)+0.25)
    fig.subplots_adjust(hspace=0.5)

    for i, bndry_label in enumerate(bndry_labels):

        size_x = np.max(lattice_coords[i][:, 0]) + 2
        size_y = np.max(lattice_coords[i][:, 1]) + 2

        min_amp = np.min(u_evolutions[i])
        max_amp = np.max(u_evolutions[i])

        for j, u_evol_eig in enumerate(u_evolutions[i]):

            if j == 0:
                title_add = bndry_label + "\n"
            else:
                title_add = ''

            for t in plot_times:

                t_match = np.argmin(np.abs(time_tags - t))
                
                if bndry_label == 'rectangle':
                    xs = np.linspace(0, 0.5, size_x)
                else:
                    xs = np.linspace(0, 1, size_x)
                ys = np.linspace(0, 1, size_y)
                X, Y = np.meshgrid(xs, ys)

                u_full = np.full((size_x, size_y), np.nan)
                for l, (x, y) in enumerate(lattice_coords[i]):
                    u_full[x + 1, y + 1] = u_evol_eig[t_match, l]
                Z = np.ma.masked_where(np.isnan(u_full), u_full).T
                
                col = plt.cm.summer(t / np.max(plot_times))
                face_cols = np.full((Z.shape[0], Z.shape[1], 4), col)

                axs[j, i].plot_surface(X, Y, Z, facecolors=face_cols, edgecolor=col, linewidth=0.05, alpha=0.2, shade=True)
                axs[j, i].set_xticks([0, 0.5, 1])
                axs[j, i].set_yticks([0, 0.5, 1])
                axs[j, i].set_xlim(0, 1)
                axs[j, i].set_ylim(0, 1)
                # axs[j, i].set_zlim(np.min(u_evol_eig), np.max(u_evol_eig))
                axs[j, i].set_zlim(min_amp, max_amp)

                axs[j, i].zaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))
                axs[j, i].tick_params(axis='x', which='major', pad=-3, labelsize=7)
                axs[j, i].tick_params(axis='y', which='major', pad=-3, labelsize=7)
                axs[j, i].tick_params(axis='z', which='major', pad=-0.5, labelsize=7)
                axs[j, i].set_title(title_add + f'$K='+'{:.3f}'.format(eigval_labels[i][j])+'$', fontsize=9)
                # axs[j, i].set_xlabel('j')
                # axs[j, i].set_ylabel('k')
                # axs[j, i].set_zlabel('$u(x,y)$')
    
    sm = cm.ScalarMappable(cmap=plt.cm.summer, norm=mcolors.Normalize(vmin=plot_times[0], vmax=plot_times[-1]))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
    # cbar_ax.set_title('$v(x,y)$', fontsize=10)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar_ax.text(-0.01, 0.5, '$t$', transform=cbar_ax.transAxes, ha='right', va='center')

    # plt.tight_layout()
    plt.show()


def animate_membranes(u_evolutions, lattice_coords, bndry_labels, eigval_labels, time_tags, title, interval=10):
    """
    Creates a 3D animation of a series of vibrating membranes from the provided frames.
    Arguments:
        u_evolutions (list): the time-dependent solutions.
        lattice_coords (list): the coordinates of the lattice points.
        bndry_labels (list): the boundary labels.
        eigval_labels (list): the eigenvalue labels.
        time_tags (numpy.ndarray): the timestamps for each time step.
        title (str): the title of the animation.
        interval (int, optional) - the interval between frames in milliseconds. Default is 10.
    Returns:
        an HTML animation
    """

    assert len(bndry_labels) == len(u_evolutions) == len(lattice_coords) == len(eigval_labels), 'Boundary labels, time-dependent solutions, and lattice coordinates must have the same length.'
    assert time_tags.ndim == 1, 'Time tags must be 1D.'
    assert time_tags.shape[0] == u_evolutions[0].shape[1], 'Time tags and time-dependent solutions must have the same length.'
    assert len(set([u_evol.shape[:2] for u_evol in u_evolutions])) == 1, 'All time-dependent solutions must have the same length.'
    
    n_frames = time_tags.shape[0]

    fig, axs = plt.subplots(len(bndry_labels), u_evolutions[0].shape[0] + 1, subplot_kw={'projection': '3d'})
    
    fig.set_size_inches(6 * u_evolutions[0].shape[0] / len(bndry_labels) + 1, 7)

    Xs = []
    Ys = []
    plots = []
    sizes_x = np.empty(len(bndry_labels), dtype=int)
    sizes_y = np.empty(len(bndry_labels), dtype=int)
    min_amps = np.empty(len(bndry_labels))
    max_amps = np.empty(len(bndry_labels))

    # Prepare data for plots
    for i, bndry_label in enumerate(bndry_labels):

        sizes_x[i] = np.max(lattice_coords[i][:, 0]) + 2
        sizes_y[i] = np.max(lattice_coords[i][:, 1]) + 2

        min_amps[i] = np.min(u_evolutions[i])
        max_amps[i] = np.max(u_evolutions[i])

        for j in range(u_evolutions[i].shape[0] + 1):
                
            if bndry_label == 'rectangle':
                xs = np.linspace(0, 0.5, sizes_x[i])
            else:
                xs = np.linspace(0, 1, sizes_x[i])
            ys = np.linspace(0, 1, sizes_y[i])
            X, Y = np.meshgrid(xs, ys)
            
            # Sum eigenvectors for last plot
            if j < u_evolutions[i].shape[0]:
                u_evol_eig = u_evolutions[i][j]
            else:
                u_evol_eig = np.sum(u_evolutions[i], axis=0)

            u_full = np.full((sizes_x[i], sizes_y[i]), np.nan)
            for l, (x, y) in enumerate(lattice_coords[i]):
                u_full[x + 1, y + 1] = u_evol_eig[0, l]
            Z = np.ma.masked_where(np.isnan(u_full), u_full).T
            
            plots.append(axs[i, j].plot_surface(X, Y, Z, cmap='plasma', linewidth=0.05, shade=True))
            Xs.append(X)
            Ys.append(Y)
    
    def reset_ax(ax, X, Y, Z, min_amp, max_amp, bndry_label, eigval_label):
        """
        Reset axis with new data
        """

        ax.clear()

        plot = ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0.05, shade=True, vmin=min_amp, vmax=max_amp)

        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ax.set_zlim(np.min(u_evol_eig), np.max(u_evol_eig))
        ax.set_zlim(min_amp*1.5, max_amp*1.5)

        ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))
        ax.tick_params(axis='x', which='major', pad=-3, labelsize=7)
        ax.tick_params(axis='y', which='major', pad=-3, labelsize=7)
        ax.tick_params(axis='z', which='major', pad=-0.5, labelsize=7)
        
        if eigval_label is not None:
            ax.set_title(bndry_label + f'\n$K='+'{:.3f}'.format(eigval_label)+'$', fontsize=9)
        else:
            ax.set_title(bndry_label + '\nSum', fontsize=9)
        
        return plot

    def init_anim():
        """
        Initialize animation
        """
        for i, bndry_label in enumerate(bndry_labels):
            for j in range(u_evolutions[i].shape[0] + 1):

                idx = i * (u_evolutions[i].shape[0] + 1) + j

                # Sum eigenvectors for last plot
                if j < u_evolutions[i].shape[0]:
                    u_evol_eig = u_evolutions[i][j]
                else:
                    u_evol_eig = np.sum(u_evolutions[i], axis=0)

                u_full = np.full((sizes_x[i], sizes_y[i]), np.nan)
                for l, (x, y) in enumerate(lattice_coords[i]):
                    u_full[x + 1, y + 1] = u_evol_eig[0, l]
                Z = np.ma.masked_where(np.isnan(u_full), u_full).T
                
                if j < u_evolutions[i].shape[0]:
                    plots[idx] = reset_ax(axs[i, j], Xs[idx], Ys[idx], Z, min_amps[i], max_amps[i], bndry_label, eigval_labels[i][j])
                else: 
                    plots[idx] = reset_ax(axs[i, j], Xs[idx], Ys[idx], Z, min_amps[i], max_amps[i], bndry_label, None)
        
        return plots

    def update(t):
        """
        Update animation
        """
        for i in range(len(bndry_labels)):
            for j in range(u_evolutions[i].shape[0] + 1):
                
                idx = i * (u_evolutions[i].shape[0] + 1) + j

                # Sum eigenvectors for last plot
                if j < u_evolutions[i].shape[0]:
                    u_evol_eig = u_evolutions[i][j]
                else:
                    u_evol_eig = np.sum(u_evolutions[i], axis=0)

                u_full = np.full((sizes_x[i], sizes_y[i]), np.nan)
                for l, (x, y) in enumerate(lattice_coords[i]):
                    u_full[x + 1, y + 1] = u_evol_eig[t, l]
                Z = np.ma.masked_where(np.isnan(u_full), u_full).T

                if j < u_evolutions[i].shape[0]:
                    plots[idx] = reset_ax(axs[i, j], Xs[idx], Ys[idx], Z, min_amps[i], max_amps[i], bndry_label, eigval_labels[i][j])
                else:
                    plots[idx] = reset_ax(axs[i, j], Xs[idx], Ys[idx], Z, min_amps[i], max_amps[i], bndry_label, None)
        
        # Update the figure title
        fig.suptitle(title + ': t={:.2f}'.format(time_tags[t]))

        return plots

    anim = animation.FuncAnimation(fig, update, init_func=init_anim, frames=n_frames, interval=interval, blit=True)

    # plt.tight_layout()
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


def plot_harmonic_oscillator(xs, vs, ts, ks):
    """
    Plots the time series and the phase space of multiple
    harmonic oscillators with different k parameters.
    Arguments:
        xs (numpy.ndarray): the time series of the positions (one series per k value).
        vs (numpy.ndarray): the time series of the velocities (one series per k value).
        ts (numpy.ndarray): the time points.
        ks (numpy.ndarray): the k parameters.
    """

    assert xs.ndim == vs.ndim == 2, 'Time series must be 2D.'
    assert xs.shape[0] == vs.shape[0] == ks.shape[0], 'Time series and k parameters must have the same length.'

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(4, 4.2)

    for i, k in enumerate(ks):
        axs[0, 0].plot(vs[i], xs[i], label=f'$k={k}$', linewidth=1)
        axs[0, 1].plot(ts[i], xs[i], label=f'$k={k}$', linewidth=1)
        axs[1, 0].plot(vs[i], ts[i], label=f'$k={k}$', linewidth=1)
    
    # axs[0, 0].set_xlabel('$v(t)$')
    axs[0, 0].set_ylabel('$x(t)$')
    axs[0, 0].tick_params(axis='x', labelbottom=False, bottom=False)
    axs[1, 0].set_xlabel('$v(t)$')
    axs[1, 0].set_ylabel('$t$')
    axs[0, 1].set_xlabel('$t$')
    axs[0, 1].set_ylabel('$x(t)$')
    axs[0, 1].tick_params(axis='y', labelleft=False, labelright=True, left=False, right=True)
    axs[0, 1].yaxis.set_label_position('right')

    max_amp = max(np.max(np.abs(xs)), np.max(np.abs(vs))) + 0.5
    axs[0, 0].set_xlim((-max_amp, max_amp))
    axs[0, 0].set_ylim((-max_amp, max_amp))
    axs[0, 1].set_xlim((np.min(ts), np.max(ts)))
    axs[0, 1].set_ylim((-max_amp, max_amp))
    axs[1, 0].set_xlim((-max_amp, max_amp))
    axs[1, 0].set_ylim((np.max(ts), np.min(ts)))
    

    axs[0, 0].set_title('Phase Plot')
    # axs[1, 0].set_title('$v(t)$ vs. $t$')
    # axs[0, 1].set_title('$x(t)$ vs. $t$')
    axs[0, 0].grid(True)
    axs[0, 1].grid(True)
    axs[1, 0].grid(True)
    axs[1, 1].set_axis_off()

    handles, labels = axs[0, 0].get_legend_handles_labels()

    axs[1, 1].legend(handles, labels, loc='center')

    # plt.tight_layout(pad=0.5)
    plt.show()


def phase_plots(xs, vs, omegas):
    """
    Plots the phase space of multiple harmonic oscillators.
    Arguments:
        xs (numpy.ndarray): the positions.
        vs (numpy.ndarray): the velocities.
        omegas (numpy.ndarray): the angular frequencies.
    """

    assert xs.ndim == vs.ndim == 2, 'Positions and velocities must be 2D.'
    assert xs.shape[0] == vs.shape[0] == omegas.shape[0], 'Positions, velocities, and angular frequencies must have the same length.'

    n_rows = np.floor(omegas.shape[0] / 2).astype(int)

    fig, axs = plt.subplots(n_rows, 2)
    fig.set_size_inches(4, 2 * n_rows)
    fig.subplots_adjust(wspace=0.15)

    for i, ax in enumerate(axs.flatten()):
        ax.plot(vs[i], xs[i], linewidth=0.8, label='{:.3f}'.format(omegas[i]))

        ax.set_xlabel('$v(t)$')
        ax.set_ylabel('$x(t)$')
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_title(f'$\omega={omegas[i]}$')

    plt.tight_layout(pad=0.5)
    plt.show()


def plot_solution_errors(solutions, ts, labels, last=None, ax=None, legend=False):
    """
    Plots the errors between multiple numerical solutions
    of the harmonic oscillator with k=1, x0=1, v0=0
    and its analytical solution.
    Arguments:
        solutions (numpy.ndarray): the numerical solutions.
        ts (numpy.ndarray): the time points.
        labels (list): the labels of the solutions.
        last (int, optional): if provided, only this many last values will be plotted at the beginning or at the end.
        ax (matplotlib.axes.Axes, optional): the axes to plot on. If None, a new figure is created.
        legend (bool, optional): whether to show the legend. Default is False.
    """

    assert solutions.ndim == 2, 'Solutions must be 2D.'
    assert solutions.shape[1] == ts.shape[0], 'Solutions and time points must have the same length.'
    assert len(labels) == solutions.shape[0], 'Labels and solutions must have the same length.'

    x_analytical = np.cos(ts)
    errors = np.abs(solutions - x_analytical)
    # print(np.mean(errors)) 

    if last is not None:
        if last < 0:
            ax.set_xlim((ts[last], ts[-1]))
        else:
            ax.set_xlim((ts[0], ts[last]))
        # ax.set_ylim((np.mean(errors) - 0.01, np.max(errors) + 0.01))

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
    else:
        fig = ax.get_figure()

    for i, error in enumerate(errors):
        ax.plot(ts, error, label=f'Error {labels[i]}', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_yscale('log')

    if legend:
        ax.legend(fontsize='small', loc='lower right')
    else:
        ax.set_ylabel('Error')
    
    if ax is None:
        plt.show()


def plot_energy_comparison(xs, vs, ts, labels, E_base, k=1, m=1, last=None, ax=None, legend=False):
    """
    Plots the energy of multiple numerical solutions
    of the harmonic oscillator.
    Arguments:
        xs (numpy.ndarray): the solutions for the oscillator's position.
        vs (numpy.ndarray): the solutions for the oscillator's velocity.
        ts (numpy.ndarray): the time points.
        labels (list): the labels of the solutions.
        E_base (float): the baseline energy of the system.
        k (float, optional): the spring constant. Default is 1.
        m (float, optional): the mass. Default is 1.
        last (int, optional): if provided, only this many last values will be plotted at the beginning or at the end.
        ax (matplotlib.axes.Axes, optional): the axes to plot on. If None, a new figure is created.
        legend (bool, optional): whether to show the legend. Default is False.
    """

    assert xs.shape == vs.shape, 'Solutions must have the same shape.'
    assert xs.shape[1] == ts.shape[0], 'Solutions and time points must have the same length.'
    assert len(labels) == xs.shape[0], 'Labels and solutions must have the same length.'

    Es = 0.5 * m * vs**2 + 0.5 * k * xs**2

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
    else:
        fig = ax.get_figure()

    if last is not None:
        if last < 0:
            ax.set_xlim((ts[last], ts[-1]))
        else:
            ax.set_xlim((ts[0], ts[last]))
        # ax.set_ylim((np.min(Es), np.max(Es)))

    for i, E in enumerate(Es):
        ax.plot(ts, E, label=f'Energy {labels[i]}', linewidth=1)

    ax.hlines(E_base, ts[0], ts[-1], linestyles='dashed', label='Initial Energy')

    ax.set_xlabel('$t$')
    ax.set_yscale('log')

    if legend:
        ax.legend(fontsize='small', loc='lower right')
    else:
        ax.set_ylabel('Energy')
    
    if ax is None:
        plt.show()


def plot_leapfrog_rk45_comparison(xs1, vs1, xs2, vs2, ts, k=1, m=1, last=None):
    """
    Creates a combined plot of the errors and energies
    of the leapfrog and Runge-Kutta 4th order methods.
    Arguments:
        xs1 (numpy.ndarray): the positions of the leapfrog method.
        vs1 (numpy.ndarray): the velocities of the leapfrog method.
        xs2 (numpy.ndarray): the positions of the Runge-Kutta 4th order method.
        vs2 (numpy.ndarray): the velocities of the Runge-Kutta 4th order method.
        ts (numpy.ndarray): the time points.
        k (float, optional): the spring constant. Default is 1.
        m (float, optional): the mass. Default is 1.
        last (int, optional): if provided, only this many last values will be plotted.
    """

    assert xs1.shape == vs1.shape == xs2.shape == vs2.shape == ts.shape, 'Solutions and time points must have the same shape.'

    fig, axs = plt.subplots(2, 2, sharey='row')
    fig.set_size_inches(4.5, 4.5)
    
    for i in range(2):
        for j in range(2):
            axs[i, j].tick_params(axis='x', which='major', labelsize=9)
            axs[i, j].tick_params(axis='x', which='minor', labelsize=9)
            axs[i, j].tick_params(axis='y', which='major', labelsize=9)
            axs[i, j].tick_params(axis='y', which='minor', labelsize=9)

    if last is None:
        last = ts.shape[0] // 2

    E_base = 0.5 * m * vs1[0]**2 + 0.5 * k * xs1[0]**2

    # Plot starting errors
    plot_solution_errors(np.array([xs1, xs2]), ts, ['Leapfrog', 'RK4'], last=last, ax=axs[0, 0])
    # Plot starting energies
    plot_energy_comparison(np.array([xs1, xs2]), np.array([vs1, vs2]), ts,
                           ['Leapfrog', 'RK4'], E_base, k=k, m=m, last=last, ax=axs[1, 0])
    # Plot ending errors
    plot_solution_errors(np.array([xs1, xs2]), ts, ['Leapfrog', 'RK4'], last=-last, ax=axs[0, 1], legend=True)
    # Plot ending energies
    plot_energy_comparison(np.array([xs1, xs2]), np.array([vs1, vs2]), ts,
                           ['Leapfrog', 'RK4'], E_base, k=k, m=m, last=-last, ax=axs[1, 1], legend=True)
    
    plt.show()
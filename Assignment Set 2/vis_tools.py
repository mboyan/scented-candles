import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dla(cluster_grid, c_grid=None, ax=None, title=None):
    """
    Plots a DLA cluster with a diffusion gradient.
    arguments:
        cluster_grid (ndarray): The DLA cluster grid.
        c_grid (ndarray): The concentration grid. Default is None.
        ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure is created.
        title (str): The title of the plot.
    """

    if c_grid is not None:
        assert c_grid.shape == cluster_grid.shape, 'c_grid and cluster_grid must have the same shape.'
        grid_combined = np.where(np.isnan(cluster_grid), c_grid, np.nan)
    else:
        grid_combined = cluster_grid

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    img = ax.imshow(grid_combined, cmap='plasma', origin='lower', interpolation='nearest')

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('DLA Cluster with Diffusion Gradient')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if ax is None:
        plt.show()

    return img


def plot_dla_diff_sim_params(df_sim_results):
    """
    Creates two plots showing the effect of the eta parameter
    over the optimal omega for the SOR method and
    the fractal dimension of the DLA cluster.
    arguments:
        df_sim_results (pandas.DataFrame): The simulation results.
    """

    fig, ax = plt.subplots(2, 1, figsize=(4, 7), sharex=True)

    sns.scatterplot(data=df_sim_results, x='$\eta$', y='$\omega$', size=0.25, ax=ax[0], legend=False)

    sns.lineplot(data=df_sim_results, x='$\eta$', y='$\omega$', ax=ax[0])
    ax[0].set_title('Optimal Omega for SOR Method')
    ax[0].grid()

    # sns.scatterplot(data=df_sim_results, x='$\eta$', y='$D_r$', ax=ax[1])

    sns.lineplot(data=df_sim_results, x='$\eta$', y='$D_r$', ax=ax[1])
    ax[1].set_title('Fractal Dimension of DLA Cluster')
    ax[1].grid()

    plt.tight_layout()

    plt.show()


def plot_dla_param_snapshots(cluster_grids, params, c_grids=None, param_name='parameter'):
    """
    Plots a series of DLA clusters with a diffusion gradient
    for different eta values.
    arguments:
        cluster_grids (ndarray): The DLA cluster grids.
        params (ndarray): The parameter values corresponding to the snapshots.
        c_grids (ndarray): The concentration grids. Default is None.
    """

    n_plots= cluster_grids.shape[0]

    fig, ax = plt.subplots(n_plots, 1, figsize=(4, 4 * n_plots + 1), sharex=True, sharey=True)

    for i in range(n_plots):
        if c_grids is not None:
            img = plot_dla(cluster_grids[i], c_grids[i], ax=ax[i], title=f'{param_name} = {params[i]}')
        else:
            img = plot_dla(cluster_grids[i], ax=ax[i], title=f'{param_name} = {params[i]}')

    if c_grids is not None:
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
        cbar_ax.text(-0.2, 0.5, 'u(t,x,y)', transform=cbar_ax.transAxes, ha='left', va='center')

    plt.show()


def plot_dla_mc_sim_params(df_sim_results):
    """
    Creates a plot showing the effect of the p_s parameter
    on the fractal dimension of the Monte Carlo DLA cluster.
    arguments:
        df_sim_results (pandas.DataFrame): The simulation results.
    """

    fig, ax = plt.subplots(figsize=(4, 4))

    sns.lineplot(data=df_sim_results, x='$p_s$', y='$D_r$', ax=ax)
    ax.set_title('Fractal Dimension of DLA Cluster')
    ax.grid()

    plt.tight_layout()

    plt.show()


def plot_reaction_diffusion(c_grids, labels=None):
    """
    Plot multiple 2D reaction-diffusion simulation results.
    arguments:
        c_grids (ndarray): The grids of concentrations (either U or V).
        labels (list): Optional labels for the plots. Default is None
    """

    assert np.ndim(c_grids) == 3, 'input must contain an array of 2D grids'

    n_plots = c_grids.shape[0]

    if labels is not None:
        assert n_plots == len(labels), 'mismatching number of labels and plots'

    fig, axs = plt.subplots(n_plots, 1, figsize=(4, 4*n_plots + 1), sharex=True)
    if n_plots == 1:
        axs = [axs]

    for i, c in enumerate(c_grids):
        img = axs[i].imshow(c, cmap='plasma', interpolation='nearest', origin='lower')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        if labels is not None:
            axs[i].set_title(labels[i])
    
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
    fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
    cbar_ax.text(-0.2, 0.5, 'u(t,x,y)', transform=cbar_ax.transAxes, ha='left', va='center')
    
    plt.show()


def plot_gray_scott_f_k(c_grids, f_range, k_range, labels=None):
    """
    Plots the concentration patterns emerging from spatially
    varying f and k parameters.
    arguments:
        c_grids (ndarray): The grid of concentrations (either U or V).
        f_range (ndarray): A 1D array of f parameter values.
        k_range (ndarray): A 1D array of k parameter values.
    """
    
    assert np.ndim(c_grids) == 3, 'c_grids must be a collection of 2D grids'
    assert c_grids.shape[1] == c_grids.shape[2] == f_range.shape[0] == k_range.shape[0], 'c_grid must be square, potential array size mismatch'

    n_plots = c_grids.shape[0]

    if labels is not None:
        assert n_plots == len(labels), 'mismatching number of labels and plots'

    fig, axs = plt.subplots(n_plots, 1, figsize=(4, 4*n_plots + 1))
    if n_plots == 1:
        axs = [axs]

    for i, c in enumerate(c_grids):
        img = axs[i].imshow(c, cmap='plasma', interpolation='nearest', origin='lower')
        axs[i].set_xlabel('k')
        axs[i].set_ylabel('f')
        if labels is not None:
            axs[i].set_title(labels[i])

        # Use parameter ranges as ticks
        f_ticks = ["%.2f" % f for f in f_range]
        k_ticks = ["%.2f" % k for k in k_range]
        axs[i].set_xticks(np.arange(f_range.shape[0])[::f_range.shape[0] // 8], f_ticks[::f_range.shape[0] // 8])
        axs[i].set_yticks(np.arange(k_range.shape[0])[::k_range.shape[0] // 8], k_ticks[::k_range.shape[0] // 8])
    
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
    fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
    cbar_ax.text(-0.2, 0.5, 'u(t,x,y)', transform=cbar_ax.transAxes, ha='left', va='center')
    
    plt.show()
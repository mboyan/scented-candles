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

    ax.imshow(grid_combined, cmap='plasma', origin='lower', interpolation='nearest')

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('DLA Cluster with Diffusion Gradient')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if ax is None:
        plt.show()


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

    n = cluster_grids.shape[0]

    fig, ax = plt.subplots(n, 1, figsize=(4, 4 * n), sharex=True, sharey=True)

    for i in range(n):
        if c_grids is not None:
            plot_dla(cluster_grids[i], c_grids[i], ax=ax[i], title=f'{param_name} = {params[i]}')
        else:
            plot_dla(cluster_grids[i], ax=ax[i], title=f'{param_name} = {params[i]}')

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
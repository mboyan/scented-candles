import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dla_diffusion(c_grid, cluster_grid, ax=None, title=None):
    """
    Plots a DLA cluster with a diffusion gradient.
    arguments:
        c_grid (ndarray): The concentration grid.
        cluster_grid (ndarray): The DLA cluster grid.
        ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure is created.
        title (str): The title of the plot.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    grid_combined = np.where(np.isnan(cluster_grid), c_grid, np.nan)

    plt.imshow(grid_combined, cmap='plasma', origin='lower')

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('DLA Cluster with Diffusion Gradient')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
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

    sns.scatterplot(data=df_sim_results, x='$\eta$', y='$\omega$', ax=ax[0])

    sns.lineplot(data=df_sim_results, x='$\eta$', y='$\omega$', ax=ax[0])
    ax[0].set_title('Optimal Omega for SOR Method')
    ax[0].grid()

    sns.scatterplot(data=df_sim_results, x='$\eta$', y='$D_r$', ax=ax[1])

    sns.lineplot(data=df_sim_results, x='$\eta$', y='$D_r$', ax=ax[1])
    ax[1].set_title('Fractal Dimension of DLA Cluster')
    ax[1].grid()

    plt.tight_layout()

    plt.show()
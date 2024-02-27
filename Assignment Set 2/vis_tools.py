import numpy as np
import matplotlib.pyplot as plt

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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def animate_strings(frames, times, interval=10):
    """
    Creates a 2D animation of a series of vibrating strings from the provided frames.
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


def animate_diffusion(frames, times, interval=10):
    """
    Creates a 2D animation of diffusion on a square lattice.
    input:
        frames (numpy.ndarray) - a 3D array containing the frames for the animation
        times (numpy.ndarray) - an array with the corresponding timesteps
    output:
        an HTML animation
    """

    assert frames.ndim == 3, 'frames must be a 3D array'
    assert frames.shape[1] == frames.shape[2], 'lattice must be square'
    assert frames.shape[0] == times.shape[0], 'mismatch between number of frames and timesteps'

    n_frames = times.shape[0]

    fig, ax = plt.subplots()
    plot = ax.imshow(frames[0], vmin=0.0, vmax=1.0, origin='lower', cmap='plasma')
    time_txt = ax.text(0.05, 0.05, '', c='w', transform=ax.transAxes)

    def init_anim():
        """
        Initialize animation
        """
        plot.set_array(frames[0])
        time_txt.set_text('')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Numerical solution of the diffusion equation')
        return plot, time_txt
    
    def update(i):
        """
        Update animation
        """
        plot.set_array(frames[i])
        time_txt.set_text(f'$t={times[i]:.3f}$')
        return plot, time_txt
    
    anim = animation.FuncAnimation(fig, update, init_func=init_anim, frames=n_frames, interval=interval, blit=True)

    plt.tight_layout()
    plt.show()

    return HTML(anim.to_html5_video())


def plot_lattice_map(N):
    """
    Plots a schematic of a 2D lattice with a source at the top, a sink at the bottom
    and a periodic condition along the x-axis.
    inputs:
        N (int) - the size of the 
    """

    xs = np.arange(-1, N+1)
    ys = np.arange(N)

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    ax.set_aspect('equal', 'box')
    
    # Plot vertical grid lines
    for x in xs[1:-1]:
        ax.plot([x, x], [ys[0], ys[-1]], color='black')

    # Plot horizontal grid lines
    for y in ys:
        ax.plot([xs[0], xs[1]], [y, y], color='black', ls='--')
        ax.plot([xs[1], xs[-2]], [y, y], color='black')
        ax.plot([xs[-2], xs[-1]], [y, y], color='black', ls='--')

    # Plot dots at grid nodes
    for x in xs:
        for y in ys:
            if y == 0:
                col = 'b'
            elif y == N-1:
                col = 'y'
            else:
                col = 'w'
            ax.scatter(x, y, color=col, marker='o', edgecolors='k')
            ax.text(x+0.25/N, y+0.25/N, f'({x%N},{y})', ha='left', va='bottom', fontsize=8)
    
    ax.axis('off')
    # ax.set_xticks([])
    # ax.set_yticks([])


def plot_error_convergence(c_frames_list, labels):
    """
    Plots the convergence of the maximum error between the numerical solution
    of the diffusion equation on a single-periodic, source-to-sink lattice
    and the analytical solution c(y) = y for multiple iteration strategies
    inputs:
        c_frames (list of numpy.ndarray) - a list containint the series of simulation frames for each strategy
        labels (list of str) - a list of graph labels
    """

    assert type(c_frames_list) == list, 'input must be list of numpy arrays'
    assert len(labels) == len(c_frames_list), 'mismatching number of labels'
    
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    ax.set_xlabel('Iterations $n$')
    ax.set_ylabel(r'$\max_{i,j}|y - c_{i,j}^k|$')

    for i, c_frames in enumerate(c_frames_list):
        an_sol = np.tile(np.linspace(0, 1, c_frames.shape[2]), (c_frames.shape[0], c_frames.shape[1], 1))
        an_sol = np.moveaxis(an_sol, 1, 2)
        error = np.max(np.abs(c_frames - an_sol), axis=(1, 2))
        
        ax.semilogy(error, label=labels[i])

    ax.legend(fontsize='small')
    ax.grid()

    plt.show()


def plot_delta_convergence(c_frames_list, labels):
    """
    Plots the convergence of the difference between iterations
    of the diffusion equation on a single-periodic, source-to-sink lattice
    and the analytical solution c(y) = y for multiple iteration strategies
    inputs:
        c_frames (list of numpy.ndarray) - a list containint the series of simulation frames for each strategy
        labels (list of str) - a list of graph labels
    """

    assert type(c_frames_list) == list, 'input must be list of numpy arrays'
    assert len(labels) == len(c_frames_list), 'mismatching number of labels'
    
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    ax.set_xlabel('Iterations $n$')
    ax.set_ylabel('$\delta$')

    for i, c_frames in enumerate(c_frames_list):
        delta = np.max(c_frames[1:] - c_frames[:-1], axis=(1, 2))
        
        ax.semilogy(delta, label=labels[i])

    ax.legend(fontsize='small')
    ax.grid()

    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def animate_strings(frames, interval=10):
    """
    Creates a 2D animation of a series of vibrating strings from the provided frames.
    input:
        frames (numpy.ndarray) - a 3D array containing the frames for the animation of each string
    output:
        an HTML animation
    """

    assert frames.ndim == 3, 'frames must be a 3D array'

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
            axs[j].set_title(f'Init. condition {j}')
        return *plots, *time_txt

    def update(i):
        """
        Update animation
        """
        for j, plot in enumerate(plots):
            plot.set_ydata(frames[j][i])
            time_txt[j].set_text(f'$t={i}$')
        return *plots, *time_txt

    anim = animation.FuncAnimation(fig, update, init_func=init_anim, frames=n_frames, interval=interval, blit=True)

    plt.tight_layout()
    plt.show()

    return HTML(anim.to_html5_video())
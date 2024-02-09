import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def animate_frames(frames):
    """
    Creates a 2D animation of a vibrating string from the provided frames.
    input:
        frames (numpy.ndarray) - a 2D array containing the frames for the animation
    output:
        an HTML animation
    """

    assert frames.ndim == 2, 'frames must be a 3D array'

    n_frames = frames.shape[0]

    # fig = plt.figure()
    # plot,  = plt.plot(frames[0])
    ax = plt.subplot()
    fig = ax.get_figure()
    plot, = ax.plot(frames[0])

    def update(i):
        plot.set_data(np.linspace(0, 1, frames[i].shape[0]), frames[i])
        return plot,

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=10, blit=True)

    return HTML(anim.to_html5_video())